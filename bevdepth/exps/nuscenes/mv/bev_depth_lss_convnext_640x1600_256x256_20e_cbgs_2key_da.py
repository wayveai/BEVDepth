"""
Try to reproduce the leaderboard submission [BEVDepth-pure: mAP 0.520, NDS 0.609]

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
- total batch size 64 
- data augmentation by randomly sampling time intervals in previous frames    
"""
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel as BaseBEVDepthLightningModel,
)  # noqa
from bevdepth.models.base_bev_depth import BaseBEVDepth as BaseBEVDepth

from bevdepth.exps.nuscenes.base_exp import backbone_conf, ida_aug_conf, head_conf

H = 900
W = 1600
final_dim = (640, 1600)

convnext_arch_to_params = {
    "small": {
        "checkpoint": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128-noema_in1k_20221208-4a618995.pth",
        "feature_channels": [96, 192, 384, 768],
    },
    "base": {
        "checkpoint": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128-noema_in1k_20221208-f8182678.pth",
        "feature_channels": [128, 256, 512, 1024],
    },
}
convnext_arch = "base"


# https://mmclassification.readthedocs.io/en/dev-1.x/papers/convnext.html
backbone_conf = {
    "x_bound": [-51.2, 51.2, 0.4],
    "y_bound": [-51.2, 51.2, 0.4],
    "z_bound": [-5, 3, 8],
    "d_bound": [2.0, 58.0, 0.5],
    "final_dim": final_dim,
    "output_channels": 80,
    "downsample_factor": 16,
    "img_backbone_conf": dict(
        type="mmcls.ConvNeXt",
        arch=convnext_arch,
        out_indices=[0, 1, 2, 3],  # downsample input by 4, 8, 16, 32
        drop_path_rate=0.5,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=convnext_arch_to_params[convnext_arch]["checkpoint"],
        ),
    ),
    "img_neck_conf": dict(
        type="SECONDFPN",
        in_channels=convnext_arch_to_params[convnext_arch]["feature_channels"],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    "depth_net_conf": dict(in_channels=512, mid_channels=512),
}

ida_aug_conf = {
    "resize_lim": (0.94, 1.25),
    "final_dim": final_dim,
    "rot_lim": (-5.4, 5.4),
    "H": H,
    "W": W,
    "rand_flip": True,
    "bot_pct_lim": (0.0, 0.0),
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
}

bbox_coder = dict(
    type="CenterPointBBoxCoder",
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=2,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=2,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=2,
    voxel_size=[0.2, 0.2, 8],
    nms_type="circle",
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf.update(
    dict(
        separate_head=dict(
            type="DCNSeparateHead",
            dcn_config=dict(
                type="DCN",
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4,
            ),
            init_bias=-2.19,
            final_kernel=3,
        ),
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        bbox_coder=bbox_coder,
    )
)


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(
        self,
        backbone_conf=backbone_conf,
        ida_aug_conf=ida_aug_conf,
        head_conf=head_conf,
        **kwargs
    ):
        super().__init__(**kwargs)
        # overwrite the default
        self.backbone_conf = backbone_conf
        self.ida_aug_conf = ida_aug_conf
        self.head_conf = head_conf
        self.dbound = self.backbone_conf["d_bound"]
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])

        self.backbone_conf["use_da"] = True
        self.data_use_cbgs = True
        self.model = BaseBEVDepth(
            self.backbone_conf, self.head_conf, is_train_depth=True
        )
        self.train_info_paths = [
            os.path.join(self.data_root, "nuscenes_infos_train.pkl"),
            # os.path.join(self.data_root, "nuscenes_infos_val.pkl"),
        ]

        # self.num_sweeps = 2
        # self.sweep_idxes = [4]

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = MultiStepLR(optimizer, [16, 19])
        return [[optimizer], [scheduler]]


if __name__ == "__main__":
    run_cli(
        BEVDepthLightningModel,
        "bev_depth_lss_convnext_640x1600_256x256_20e_cbgs_2key_da",
        extra_trainer_config_args={"epochs": 20},
    )
