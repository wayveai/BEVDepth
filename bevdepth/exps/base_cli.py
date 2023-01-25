# Copyright (c) Megvii Inc. All rights reserved.
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from bevdepth.callbacks.ema import EMACallback
from bevdepth.exps.nuscenes.base_exp import (BEVDepthLightningModel,
                                             bevdepth_root)
from bevdepth.utils.torch_dist import all_gather_object, synchronize


def run_cli(
    model_class=BEVDepthLightningModel,
    exp_name='base_exp',
    use_ema=False,
    extra_trainer_config_args={},
):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument(
        '-e',
        '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set',
    )
    parent_parser.add_argument(
        '-p',
        '--predict',
        dest='predict',
        action='store_true',
        help='predict model on testing set',
    )
    parent_parser.add_argument('-b',
                               '--batch_size_per_device',
                               type=int,
                               default=1)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    training_artifacts_root_dir = os.path.join(bevdepth_root, 'outputs',
                                               exp_name)

    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=extra_trainer_config_args.get('epochs', 24),
        accelerator='ddp',
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        limit_val_batches=0,
        # limit_val_batches=1.0,
        # check_val_every_n_epoch=5,
        limit_train_batches=extra_trainer_config_args.get(
            'limit_train_batches', 1.0),
        enable_checkpointing=True,
        # precision=16,
        default_root_dir=training_artifacts_root_dir,
    )
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(training_artifacts_root_dir, 'checkpoints'),
            save_last=False,
            save_top_k=-1,
        )
    ]
    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        callbacks = callbacks.append(ema_callback)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    elif args.predict:
        predict_step_outputs = trainer.predict(model, ckpt_path=args.ckpt_path)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                     os.path.dirname(args.ckpt_path))
    else:
        if args.ckpt_path is not None:
            model = model.load_from_checkpoint(args.ckpt_path)
        trainer.fit(model)
