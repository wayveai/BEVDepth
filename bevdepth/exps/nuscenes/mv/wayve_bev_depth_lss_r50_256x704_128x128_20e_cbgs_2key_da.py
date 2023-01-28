import os

from bevdepth.evaluators.wayve_det_evaluators import WayveDetNuscEvaluator
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.base_exp import ida_aug_conf, wayve_data_root
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa

WAYVE_DATASET_VERSION = 'v0.2'

H = 1280
W = 2048

# TODO: make backbone_conf match better with 3cam
ida_aug_conf.update({
    'H': H,
    'W': W,
})


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, ida_aug_conf=ida_aug_conf, **kwargs):
        self.key_idxes = [-1]
        super().__init__(**kwargs)

        self.ida_aug_conf = ida_aug_conf

        self.data_root = wayve_data_root
        assert os.path.exists(self.data_root)
        self.train_info_paths = os.path.join(
            self.data_root, f'infos_{WAYVE_DATASET_VERSION}-train.pkl')
        self.val_info_paths = os.path.join(
            self.data_root, f'infos_{WAYVE_DATASET_VERSION}-test.pkl')

        self.evaluator = WayveDetNuscEvaluator(
            output_dir=self.default_root_dir,
            data_root=self.data_root,
            version=f'{WAYVE_DATASET_VERSION}-train')


if __name__ == '__main__':
    run_cli(
        BEVDepthLightningModel,
        'wayve_bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da',
        extra_trainer_config_args={'epochs': 20},
    )
