# Input an video, for each frame, output the predicted 3D smpl mesh and
# 3D articulation plane mesh as seperate .obj files, 2d masks of the plane 
# and person as seperate.txt files in the size of the original video, and
# the original frame

from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
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
from detectron2.utils.visualizer import Visualizer, GenericMask, ColorMode 
from detectron2.config import get_cfg
# import some common detectron2 utilities for pointrend
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.projects import point_rend

import pytorch3d
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.utils import ico_sphere

import planercnn.modeling  # noqa
from planercnn.data import PlaneRCNNMapper
from planercnn.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis
from planercnn.visualization.unit_vector_plot import get_normal_figure
from planercnn.evaluation import ArtiEvaluator
from planercnn.config import get_planercnn_cfg_defaults
from planercnn.utils.vis import get_pcd, project2D, random_colors, get_single_image_mesh_arti
from planercnn.utils.mesh_utils import save_obj, get_camera_meshes, transform_meshes, rotate_mesh_for_webview
from planercnn.utils.opt_utils import track_planes, optimize_planes
from planercnn.utils.arti_vis import create_instances, PlaneRCNN_Branch, draw_pred, draw_gt, get_normal_map

from utils import Mesh
from models import CMR, SMPL
from utils.imutils import preprocess_generic
from utils.renderer_p3d import Pytorch3dRenderer
from models.geometric_layers import orthographic_projection
import config


"""
python generate_smpl.py --config config/config.yaml --input /z/syqian/articulation_data/step2_filtered_clips/CxTFIEpSgew_34_360.mp4 --output /data/siyich/cmr_art/mask_mesh_3336_output --webvis
python generate_smpl.py --config config/config.yaml --webvis
"""


if torch.cuda.is_available():
    DEVICE = torch.device("cuda") 
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

colors = {
    # colorbline/print/copy safe:
    'light_gray':  [0.9, 0.9, 0.9],
    'light_purple':  [0.8, 0.53, 0.53],
    'light_green': [166/255.0, 178/255.0, 30/255.0],
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
}

SMPL_RES=224


def process_image(img_file, input_res=224):
    """
    Read image, do preprocessing
    """
    normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)
    #rgb_img_in = cv2.imread(img_file)[:,:,::-1].copy().astype(np.float32)
    rgb_img_in = img_file[:,:,::-1].copy().astype(np.float32)
    rgb_img = preprocess_generic(rgb_img_in, input_res)
    disp_img = preprocess_generic(rgb_img_in, input_res, display=True)
    # disp_img = rgb_img_in
    img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
    disp_img = np.transpose(disp_img.astype('float32'),(2,0,1))/255.0
    img = torch.from_numpy(img).float()
    disp_img = torch.from_numpy(disp_img).float()
    norm_img = normalize_img(img.clone())[None]

    return disp_img, norm_img


def process_resize(img_file, input_res=224):
    """
    Read image, do preprocessing
    """
    rgb_img_in = img_file[:,:,::-1].copy().astype(np.uint8)
    resize_img = preprocess_generic(rgb_img_in, input_res, display=True)
    resize_img = resize_img.astype(np.uint8)

    return resize_img


def save_obj_model(args, preds, frames, frame_id, axis_dir='l'):
    # for the most confident box of the first frame, visualize future frames
    p_instance = preds[frame_id]
    if p_instance.scores.shape[0] == 0:
        print("no prediction!")
        return

    box_id = p_instance.scores.argmax()
    vis = Visualizer(frames[frame_id])
    im = frames[frame_id]
    
    # computing the rotation axis
    pred_mask = p_instance.pred_masks[box_id]
    pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
    pred_plane[:, [1,2]] = pred_plane[:, [2, 1]]
    pred_plane[:, 1] = - pred_plane[:, 1]
    pred_box_centers = p_instance.pred_boxes.get_centers()

    pts = angle_offset_to_axis(p_instance.pred_rot_axis, pred_box_centers)
    verts = pred_mask.nonzero().flip(1)
    normal = F.normalize(pred_plane, p=2)[0]
    offset = torch.norm(pred_plane, p=2)
    verts_axis = pts[box_id].reshape(-1, 2)
    verts_axis_3d = get_pcd(verts_axis, normal, offset)
    if args.webvis:
        # 3d transformation for model-viewer
        verts_axis_3d = torch.tensor((np.array([[-1,0,0], [0,1,0], [0,0,-1]])@np.array([[-1,0,0],[0,-1,0],[0,0,1]])@verts_axis_3d.numpy().T).T)
    dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    # create visualization for rot axis
    axis_scale = pytorch3d.transforms.Scale(0.1).cuda()
    axis_pt1_t = pytorch3d.transforms.Transform3d().translate(verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
    axis_pt1_t = axis_pt1_t.cuda()
    axis_pt1 = ico_sphere(0).cuda()
    axis_pt1.verts_list()[0] = axis_scale.transform_points(axis_pt1.verts_list()[0])
    axis_pt1.verts_list()[0] = axis_pt1_t.transform_points(axis_pt1.verts_list()[0])
    axis_pt2_t = pytorch3d.transforms.Transform3d().translate(verts_axis_3d[1][0], verts_axis_3d[1][1], verts_axis_3d[1][2])
    axis_pt2_t = axis_pt2_t.cuda()
    axis_pt2 = ico_sphere(0).cuda()
    axis_pt2.verts_list()[0] = axis_scale.transform_points(axis_pt2.verts_list()[0])
    axis_pt2.verts_list()[0] = axis_pt2_t.transform_points(axis_pt2.verts_list()[0])
    axis_verts_rgb = torch.ones_like(verts)[None].cuda()  # (1, V, 3)
    axis_textures = Textures(verts_uvs=axis_verts_rgb, faces_uvs=axis_pt1.faces_list(), maps=torch.zeros((1,5,5,3)).cuda())
    axis_pt1.textures = axis_textures
    axis_pt2.textures = axis_textures
    
    # computing pcd
    plane_params = p_instance.pred_planes[box_id:(box_id + 1)]
    segmentations = p_instance.pred_masks[box_id:(box_id + 1)]
    reduce_size = False
    height = im.shape[0]
    width = im.shape[1]

    # bkgd mesh
    """
    mesh_bkgd, uv_maps_bkgd = get_single_image_mesh_arti(plane_params, 1 - segmentations, img=im, height=height, width=width, webvis=args.webvis, reduce_size=reduce_size)
    """

    # obj mesh
    mesh, uv_maps = get_single_image_mesh_arti(plane_params, segmentations, img=im, height=height, width=width, webvis=args.webvis, reduce_size=reduce_size)
    mesh = mesh.cuda()
    mesh_pcd = mesh.verts_list()[0].clone()
    
    # bitmask pcd
    pcd = get_pcd(verts, normal, offset)#, focal_length)
    pcd = pcd.float().cuda()

    # transforms
    t1 = pytorch3d.transforms.Transform3d().translate(verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
    t1 = t1.cuda()
    if axis_dir == 'l':
        # angles = torch.FloatTensor(np.arange(-1.8, 0.1, 1.8/4)[:, np.newaxis])
        angles = torch.FloatTensor(np.arange(0.0, 0.0001, 0.00009)[:, np.newaxis])
        # print(angles)
    elif axis_dir == 'r':
        # angles = torch.FloatTensor(np.arange(0.0, 1.8, 1.8/4)[:, np.newaxis])
        angles = torch.FloatTensor(np.arange(0.0, 0.0001, 0.00009)[:, np.newaxis])
        # print(angles)
    else:
        raise NotImplementedError
    axis_angles = angles * dir_vec
    rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
    t2 = pytorch3d.transforms.Rotate(rot_mats)
    t2 = t2.cuda()
    t3 = t1.inverse()
    pcd_trans = t3.transform_points(pcd)
    pcd_trans = t2.transform_points(pcd_trans)
    pcd_trans = t1.transform_points(pcd_trans)

    mesh_pcd_trans = t3.transform_points(mesh_pcd)
    # print(mesh_pcd_trans.shape[0])
    mesh_pcd_trans = t2.transform_points(mesh_pcd_trans)
    # print(mesh_pcd_trans.shape[0])
    mesh_pcd_trans = t1.transform_points(mesh_pcd_trans)
    # print(mesh_pcd_trans.shape[0])

    meshes = [mesh]
    uv_maps_list = [uv_maps[0]]
    for i in range(mesh_pcd_trans.shape[0]):
        faces_list = mesh.faces_list()
        new_faces_list = [f.clone() for f in faces_list]
        imesh = Meshes(verts=[mesh_pcd_trans[i]], faces=new_faces_list, textures=mesh.textures.clone())
        meshes.append(imesh)
        uv_maps_list.append(uv_maps[0])

    # add rot_axis
    """
    meshes.append(axis_pt1)
    meshes.append(axis_pt2)
    uv_maps_list.append(uv_maps[0])
    uv_maps_list.append(uv_maps[0])
    """

    meshes = pytorch3d.structures.join_meshes_as_batch(meshes)
    meshes = meshes.cpu()

    output_dir = os.path.join(args.output, 'frame_{:0>4}'.format(frame_id))

    basename = 'arti'
    save_obj(output_dir, basename+'_pred', meshes, decimal_places=10, uv_maps=uv_maps_list)


def main():
    # random.seed(2020)
    # np.random.seed(2020)
    # random_state = np.random.RandomState(seed=26)

    # command line arguments
    parser = argparse.ArgumentParser(
        description="A script that generates results of articulation prediction."
    )
    parser.add_argument("--config", required=True, help="config/config.yaml")
    parser.add_argument("--input", default=None, help="input video file")
    parser.add_argument("--output", default="output", help="output directory")
    parser.add_argument('--webvis', action='store_true')
    parser.add_argument("--conf-threshold", default=0.7, type=float, help="confidence threshold")
    parser.add_argument('--checkpoint', default="data/models/ours/2020_02_29-18_30_01.pt", help='Path to pretrained checkpoint')
    args = parser.parse_args()

    # setup logger
    logger = setup_logger()

    # load articulation model
    cfg = get_cfg()
    get_planercnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config)
    model = PlaneRCNN_Branch(cfg)
    shortened_class_names = {'arti_rot':'R', 'arti_tran':'T'}
    metadata = MetadataCatalog.get('arti_train')
    cls_name_map = [shortened_class_names[cls] for cls in metadata.thing_classes]

    # load smpl model
    mesh = Mesh(device=DEVICE)
    cmr = CMR(mesh, 5, 256, pretrained_checkpoint=args.checkpoint)
    smpl = SMPL()
    cmr = cmr.to(DEVICE)
    cmr.eval()
    smpl = smpl.to(DEVICE)

    # load pointrend model
    coco_metadata = MetadataCatalog.get("coco_2017_val")
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file("/home/siyich/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    pointrend = DefaultPredictor(cfg)

    # create render for smpl mesh
    renderer = Pytorch3dRenderer(
                    img_size=SMPL_RES, 
                    mesh_color=colors['light_purple'],
                    focal_length=config.FOCAL_LENGTH,
                    camera_t=np.array((0,0,0)).reshape(1,3),
                    device=DEVICE)
    
    # read video and run per-frame inference
    if args.input==None:
        data_path = "/z/syqian/articulation_data/step2_filtered_clips/"
        img_list = glob(data_path+"*.mp4")
        index = np.random.randint(len(img_list))
        args.input = img_list[index]
        print("Randomly select video", args.input)
        args.output = "/data/siyich/cmr_art/mask_mesh_" + str(index) + "_output"

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    video_path = args.input
    reader = imageio.get_reader(video_path)
    height, width = 480, 640
    # for saving results of articulation
    frames = []
    preds = []
    # for saving results of smpl
    pred_pose_array = []
    pred_shape_array = []
    # for saving results of pointrend
    pred_pointrend_person = []


    for i, ori_im in enumerate(tqdm(reader)):
        output_dir = os.path.join(args.output, 'frame_{:0>4}'.format(i))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        ori_height, ori_width, _ = ori_im.shape
        ori_size = (ori_width, ori_height)
        origin_name = os.path.join(output_dir, 'origin_frame.png')
        cv2.imwrite(origin_name, cv2.cvtColor(ori_im, cv2.COLOR_RGB2BGR))

        # preprocess for articulation
        im = cv2.resize(ori_im, (width, height))
        frames.append(im)
        im = im[:, :, ::-1]
        # articulation
        pred = model.inference(im)
        pred_dict = model.process(pred)
        p_instance = create_instances(
            pred_dict['instances'], im.shape[:2], 
            pred_planes=pred_dict['pred_plane'].numpy(), 
            pred_rot_axis=pred_dict['pred_rot_axis'],
            pred_tran_axis=pred_dict['pred_tran_axis'],
        )
        preds.append(p_instance)

        # preprocess for pose estimation
        img, norm_img = process_image(ori_im, input_res=SMPL_RES)
        norm_img = norm_img.to(DEVICE)
        # pose estimation
        with torch.no_grad():
            _, _, pred_camera, pred_pose, pred_shape = cmr(norm_img)
            pred_pose_array.append(pred_pose)
            pred_shape_array.append(pred_shape)
        # Calculate camera parameters for rendering
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*config.FOCAL_LENGTH/(config.INPUT_RES * pred_camera[:,0] +1e-9)],dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()
        pose_person_outfile = os.path.join(output_dir, 'person_pose.txt')
        shape_person_outfile = os.path.join(output_dir, 'person_shape.txt')
        camtrans_person_outfile = os.path.join(output_dir, 'person_trans.txt')
        file = open(pose_person_outfile, "wb")
        np.save(file, pred_pose.cpu().numpy())
        file.close
        file = open(shape_person_outfile, "wb")
        np.save(file, pred_shape.cpu().numpy())
        file.close
        np.savetxt(camtrans_person_outfile, camera_translation)
        print("Finish frame", str(i))

        """
        # Change the world coordinates to be same as the camera
        Rt = np.array(((-1,0,0,0),
                        (0,-1,0,0),
                        (0,0,1,0),
                        (0,0,0,1)))
        Tt = np.array(((1,0,0,camera_translation[0]),
                        (0,1,0,camera_translation[1]),
                        (0,0,1,camera_translation[2]),
                        (0,0,0,1)))
        Ct = Rt @ Tt
        pred_vertices = pred_vertices[0].cpu().numpy()
        pred_vertices_smpl = pred_vertices_smpl[0].cpu().numpy()
        ones = np.ones(pred_vertices.shape[0]).reshape(-1,1)
        pred_vertices = np.hstack((pred_vertices,ones))
        pred_vertices_smpl = np.hstack((pred_vertices_smpl,ones))
        pred_vertices = pred_vertices @ (Ct.T)
        pred_vertices_smpl = pred_vertices_smpl @ (Ct.T)
        pred_vertices = pred_vertices / (pred_vertices[:,3].reshape(-1,1))
        pred_vertices_smpl = pred_vertices_smpl / (pred_vertices_smpl[:,3].reshape(-1,1))
        pred_vertices = pred_vertices[:,0:3]
        pred_vertices_smpl = pred_vertices_smpl[:,0:3]
        camera_translation = np.array((0,0,0)).reshape(1,3)
        camera_r=np.array([[1,0,0],[0,1,0],[0,0,1]]).reshape(1,3,3)
        # save 3D smpl mesh
        smpl_obj_name = os.path.join(output_dir, 'smpl_pred.obj')
        only_smpl = renderer.render(pred_vertices_smpl, mesh.faces.cpu().numpy(), 
            img.permute(1,2,0).cpu().numpy(), False, True, smpl_obj_name)
        print("The 3D smpl mesh is saved in", smpl_obj_name)

        # pointrend
        pointrend_outputs = pointrend(ori_im)
        instances = pointrend_outputs["instances"]
        is_person = instances.pred_classes == 0
        bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()
        masks_person = instances[is_person].pred_masks
        try:
            mask_person_pred = masks_person[0] # only keep the first person
        except:
            print("No detected person mask")
        # append results
        pred_pointrend_person.append(mask_person_pred)
        # save 2D person mask
        mask_person_outfile = os.path.join(output_dir, 'person_mask.txt')
        np.savetxt(mask_person_outfile, mask_person_pred.cpu().numpy())
        print("The person mask is saved in", mask_person_outfile)
        """
    reader.close()

    """
    # temporal optimization for articulation
    planes = track_planes(preds)
    opt_preds = optimize_planes(preds, planes, '3dc', frames=frames)

    for i, im in (enumerate(frames)):
        output_dir = os.path.join(args.output, 'frame_{:0>4}'.format(i))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        p_instance = opt_preds[i]

        # save plane masks
        mask_plane_pred = p_instance.pred_masks.cpu()
        mask_plane_pred = (mask_plane_pred > 0.5).float()
        mask_plane_pred = np.asarray(mask_plane_pred)
        mask_plane_pred = np.moveaxis(mask_plane_pred, 0, -1)
        try:
            mask_plane_pred = cv2.resize(mask_plane_pred, ori_size)
            mask_plane_pred = mask_plane_pred[:,:,0]
        except:
            print("No detected plane mask")
        mask_plane_outfile = os.path.join(output_dir, 'plane_mask.txt')
        np.savetxt(mask_plane_outfile, mask_plane_pred)
        print("The plane mask is saved in", mask_plane_outfile)

        # save 3D plane mesh
        save_obj_model(args, opt_preds, frames, i)
        """

if __name__ == "__main__":
    main()
