import os

from bevdepth.evaluators.wayve_det_evaluators import WayveDetNuscEvaluator
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.base_exp import wayve_data_root
from bevdepth.exps.nuscenes.mv.wayve_bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_root = wayve_data_root
        assert os.path.exists(self.data_root)
        self.train_info_paths = os.path.join(self.data_root,
                                             'infos_v0.1-mini.pkl')
        self.val_info_paths = os.path.join(self.data_root,
                                           'infos_v0.1-mini.pkl')

        self.evaluator = WayveDetNuscEvaluator(
            output_dir=self.default_root_dir,
            data_root=self.data_root,
            version='v0.1-mini')


if __name__ == '__main__':
    run_cli(
        BEVDepthLightningModel,
        'debug_wayve_bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da',
        extra_trainer_config_args={
            'epochs': 20,
        },
    )
