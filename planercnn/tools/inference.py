import argparse
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import defaultdict
import cv2
from tqdm import tqdm
import pickle
import imageio
import random
import math
from glob import glob

import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import pairwise_iou, pairwise_ioa
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from detectron2.config import get_cfg

import planercnn.modeling  # noqa
from planercnn.data import PlaneRCNNMapper
from planercnn.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis
from planercnn.visualization.unit_vector_plot import get_normal_figure
from planercnn.evaluation import ArtiEvaluator
from planercnn.config import get_planercnn_cfg_defaults
from planercnn.utils.opt_utils import track_planes, optimize_planes
from planercnn.utils.arti_vis import create_instances, PlaneRCNN_Branch, draw_pred, draw_gt, get_normal_map


def main():
    random.seed(2020)
    np.random.seed(2020)

    # command line arguments
    parser = argparse.ArgumentParser(
        description="A script that generates results of articulation prediction."
    )
    parser.add_argument("--config", required=True, help="config/config.yaml")
    parser.add_argument("--input", required=True, help="input video file")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument('--save-obj', action='store_true')
    parser.add_argument('--webvis', action='store_true')
    parser.add_argument("--conf-threshold", default=0.7, type=float, help="confidence threshold")
    args = parser.parse_args()

    # setup logger
    logger = setup_logger()

    # load model
    cfg = get_cfg()
    get_planercnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config)
    model = PlaneRCNN_Branch(cfg)
    shortened_class_names = {'arti_rot':'R', 'arti_tran':'T'}
    metadata = MetadataCatalog.get('arti_train')
    cls_name_map = [shortened_class_names[cls] for cls in metadata.thing_classes]
    
    # read video and run per-frame inference
    video_path = args.input
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    frames = []
    preds = []
    org_vis_list = []
    for i, im in enumerate(tqdm(reader)):
        im = cv2.resize(im, (640, 480))
        frames.append(im)
        im = im[:, :, ::-1]
        pred = model.inference(im)
        #import pdb; pdb.set_trace()
        pred_dict = model.process(pred)
        p_instance = create_instances(
            pred_dict['instances'], im.shape[:2], 
            pred_planes=pred_dict['pred_plane'].numpy(), 
            pred_rot_axis=pred_dict['pred_rot_axis'],
            pred_tran_axis=pred_dict['pred_tran_axis'],
        )
        preds.append(p_instance)

        # visualization without optmization
        if args.output is not None:
            vis = Visualizer(im[:, :, ::-1])
            seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map)

            # surface normal
            if len(p_instance.pred_boxes) == 0:
                normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
            else:
                normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

            # combine visualization and generate output
            combined_vis = np.concatenate((seg_pred, normal_vis), axis=1)
            org_vis_list.append(combined_vis)

    reader.close()

    # temporal optimization
    planes = track_planes(preds)
    opt_preds = optimize_planes(preds, planes, '3dc', frames=frames)

    # video visualization in 2D
    writer = imageio.get_writer(os.path.join(args.output, '{}.mp4'.format('output')), fps=fps)
    for i, im in (enumerate(frames)):
        p_instance = opt_preds[i]
        org_vis = org_vis_list[i]

        vis = Visualizer(im)

        seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map)

        # surface normal
        if len(p_instance.pred_boxes) == 0:
            normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
        else:
            normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

        # combine visualization and generate output
        combined_vis = np.concatenate((seg_pred, normal_vis, org_vis), axis=1)
        writer.append_data(combined_vis)


if __name__ == "__main__":
    main()
