import os 
import torch 
from argparse import ArgumentParser
from collections import OrderedDict
import pickle

from bevdepth.exps.base_exp import head_conf, backbone_conf
from bevdepth.exps.mv.bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da import BEVDepthLightningModel
from visualize_nusc import demo


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('bevdepth_root', type=str, help='BEVDepth directory root path')
    parser.add_argument('result_json_path', help='Path of the result json file.')
    parser.add_argument('result_vis_path', help='Target path to save the visualization result.')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    bevdepth_root = args.bevdepth_root
    nusc_root = os.path.join(bevdepth_root, 'data/nuScenes')
    assert os.path.exists(nusc_root)
    lightning_module = BEVDepthLightningModel(data_root=nusc_root, batch_size_per_device=1)

    # load model and weight  
    ckpt = torch.load(os.path.join(bevdepth_root, 'bevdepth/exps/ckpt/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da.pth'))
    # remove the "model." prefix in statedict so that we can directly load the model 
    new_state_dict = OrderedDict()
    for k in ckpt['state_dict']:
        new_state_dict[k[6:]] = ckpt['state_dict'][k]
    model = lightning_module.model
    model.load_state_dict(new_state_dict)
    lightning_module.model = model.cuda().eval()

    with open(os.path.join(nusc_root, 'nuscenes_infos_val.pkl'), 'rb') as f:
        nusc_val_infos = pickle.load(f)

    nusc_val_dataloader = lightning_module.val_dataloader()
    evaluator = lightning_module.evaluator


    for i, batch in enumerate(iter(nusc_val_dataloader)):
        result = lightning_module.eval_step(batch, batch_idx=i, prefix='val')
        pred_result = result[0][:3]
        img_meta = result[0][3]

        result_path = evaluator._format_bbox([pred_result], [img_meta], jsonfile_prefix=args.result_json_path, jsonfile_name=f'results_nusc_val_{i}.json')

        demo(idx=i, nusc_results_file=result_path, dump_file=os.path.join(args.result_vis_path, f"{i}.jpg"), nusc_root=nusc_root, infos=nusc_val_infos, threshold=0.3)
    