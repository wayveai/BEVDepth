import os

import mmengine
from gen_info import generate_info
from nuscenes.nuscenes import NuScenes


def _generate_and_save_info_pkl(dataset_version: str, dataroot: str):
    dataset = NuScenes(version=dataset_version,
                       dataroot=dataroot,
                       verbose=True)
    scenes = [scene['name'] for scene in dataset.scene]

    infos = generate_info(dataset, scenes)

    mmengine.dump(infos, os.path.join(dataroot,
                                      f'infos_{dataset_version}.pkl'))


def main():
    wayve_scenes_root = './data/wayve-scenes'

    for dataset_version in ['v0.1-train', 'v0.1-test', 'v0.1-mini']:
        _generate_and_save_info_pkl(
            dataset_version=dataset_version,
            dataroot=wayve_scenes_root,
        )


if __name__ == '__main__':
    main()
