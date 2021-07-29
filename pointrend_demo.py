# Input an RGB image, output the predicted human & object masks


# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend


im = cv2.imread("/home/siyich/Articulation3D/siyich/cmr_art/random3701_output/mask_frame_1.png")
# print(len(im),len(im[0]), len(im[0][0])) # H,W,3

# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# mask_rcnn_predictor = DefaultPredictor(cfg)
# mask_rcnn_outputs = mask_rcnn_predictor(im)

cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("/home/siyich/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
# print(outputs)
instances = outputs["instances"]

is_person = instances.pred_classes == 0
bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()
masks_person = instances[is_person].pred_masks
masks_person = masks_person[0]

is_object = instances.pred_classes != 0
bboxes_object = instances[is_object].pred_boxes.tensor.cpu().numpy()
masks_object = instances[is_object].pred_masks
masks_object = masks_object[0]

masks_combine = masks_person | masks_object
masks_combine = ~ masks_combine

# create target masks where 1 represent mask of the interested obj/per,
# 0 represent nothing, -1 represent the other per/obj
# then target > 0 represent the ref_mask, which approximates gt where the interested obj/per
# is in the front; target >= 0 represent the "keep_mask"
target_person = masks_combine.clone()
target_object = masks_combine.clone()
target_person[masks_person] = masks_person[masks_person]
target_object[masks_object] = masks_object[masks_object]
target_person = target_person.float()
target_object = target_object.float()

# print(target_person.size()) # 1, H, W
# print(target_object.size()) # 1, H, W

# print(masks_person.size()) # 1, H, W
# print(masks_object.size()) # 1, H, W


"""
mask_outs = instances.get_fields()["pred_masks"] # represented by bool
label_outs = instances.get_fields()["pred_classes"] # person with label index 0
box_outs = instances.get_fields()["pred_boxes"]
score_outs = instances.get_fields()["scores"]

sample_mask0 = mask_outs[0].float() 
sample_mask1 = mask_outs[1].float() 
comb = sample_mask0.clone()
comb[sample_mask1 == 0] = 0
"""
# convert to float so that can be compared with the predicted mask of the render
# print(torch.max(sample_mask))


# Show and compare two predictions: 
# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
# mask_rcnn_result = v.draw_instance_predictions(mask_rcnn_outputs["instances"].to("cpu")).get_image()
v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
point_rend_result = v.draw_instance_predictions(instances.to("cpu")).get_image()
cv2.imwrite("mask_test.png", point_rend_result[:, :, ::-1])
cv2.imwrite("person_test.png", masks_person.cpu().numpy()*255)
cv2.imwrite("object_test.png", masks_object.cpu().numpy()*255)
# cv2.imwrite("mask_test.png", np.concatenate((point_rend_result, mask_rcnn_result), axis=0)[:, :, ::-1])