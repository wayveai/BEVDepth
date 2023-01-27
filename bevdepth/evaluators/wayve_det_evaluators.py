import os
import os.path as osp

import mmengine
import tqdm as tqdm
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (add_center_dist, filter_eval_boxes,
                                          load_prediction)
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox

from bevdepth.evaluators.det_evaluators import DetNuscEvaluator


# modified from 'load_gt' in
# nuscenes-devkit/python-sdk/nuscenes/eval/common/loaders.py
def wayve_load_gt(nusc: NuScenes,
                  eval_split: str,
                  box_cls,
                  verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.
              format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens = [s['token'] for s in nusc.sample]
    assert len(sample_tokens) > 0, 'Error: Database has no samples!'

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation',
                                         sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(
                    sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: \
                        GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(
                            sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have score.
                        attribute_name=attribute_name))
            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors
                # when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import \
                    category_to_tracking_name
                tracking_name = category_to_tracking_name(
                    sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(
                            sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    ))
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' %
                                          box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print('Loaded ground truth annotations for {} samples.'.format(
            len(all_annotations.sample_tokens)))

    return all_annotations


# identical to 'DetectionEval' in
# nuscenes-devkit/python-sdk/nuscenes/eval/detection/evaluate.py
# in order to use 'wayve_load_gt' above
class WayveScenesEval(DetectionEval):

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on,
            e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(
            result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing wayve detection evaluation')
        self.pred_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            DetectionBox,
            verbose=verbose)
        self.gt_boxes = wayve_load_gt(self.nusc,
                                      self.eval_set,
                                      DetectionBox,
                                      verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == \
            set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc,
                                            self.pred_boxes,
                                            self.cfg.class_range,
                                            verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc,
                                          self.gt_boxes,
                                          self.cfg.class_range,
                                          verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens


# modified from bevdepth/evaluators/det_evaluators.py
# in order to add wayve dataset logic
class WayveDetNuscEvaluator(DetNuscEvaluator):

    def __init__(
        self,
        data_root,
        version,
        # class_names=[
        #     'car',
        #     'truck',
        #     'bus',
        #     'motorcycle',
        #     'bicycle',
        #     'pedestrian',
        # ],
        class_names=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ],
        eval_version='detection_cvpr_2019',
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        ),
        output_dir=None,
        eval_set_map={
            'v0.1-train': 'v0.1-test',
            'v0.1-mini': 'v0.1-mini',
            'v0.2-train': 'v0.2-test',
            'v0.2-mini': 'v0.2-mini',
        }
    ) -> None:
        self.eval_version = eval_version
        self.data_root = data_root
        if self.eval_version is not None:
            from nuscenes.eval.detection.config import config_factory

            self.eval_detection_configs = config_factory(self.eval_version)
        self.version = version
        self.class_names = class_names
        self.modality = modality
        self.output_dir = output_dir
        self.eval_set_map = eval_set_map

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.eval_set_map[self.version],
                        dataroot=self.data_root,
                        verbose=False)
        nusc_eval = WayveScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=self.eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
        )
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for class_name in self.class_names:
            for k, v in metrics['label_aps'][class_name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, class_name,
                                                 k)] = val
            for k, v in metrics['label_tp_errors'][class_name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, class_name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
