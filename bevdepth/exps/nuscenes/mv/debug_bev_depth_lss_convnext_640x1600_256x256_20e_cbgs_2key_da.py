"""
Try to reproduce the leaderboard submission
[BEVDepth-pure: mAP 0.520, NDS 0.609]

DONE:
- higher input resolution (640, 1600)
    - final_dim
- larger BEV (256, 256)
    - x_bound
    - y_bound
    - out_size_factor (in bbox_coder, train_cfg and test_cfg)
- ConvNeXt-base backbone (img_backbone)
- DCN centerhead

TODO:
- use both train and val set for training
- total batch size 64 (can't fit 8 * 8 at this res, currently using 4 * 8)
- data augmentation by randomly sampling time intervals in previous frames
"""
import os

import torch
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_lss_convnext_640x1600_256x256_20e_cbgs_2key_da import \
    BEVDepthLightningModel  # noqa


class BEVDepthLightningModel(BEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.train_info_paths = [
            os.path.join(self.data_root, 'nuscenes_infos_mini_train.pkl'),
        ]

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-3)
        scheduler = MultiStepLR(optimizer, [1])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(
        BEVDepthLightningModel,
        'debug_bev_depth_lss_convnext_640x1600_256x256_20e_cbgs_2key_da',
        extra_trainer_config_args={
            'epochs': 1,
            'limit_train_batches': 10
        },
    )
