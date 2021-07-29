# Input an video, output the predicted smpl mesh of each frame & articulation axis as well as 
# planes of frame 0,30,60,90 as .obj files 

from __future__ import division
from __future__ import print_function

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


from torchvision.transforms import Normalize
from utils import Mesh
from models import CMR, SMPL
from utils.imutils import preprocess_generic
from utils.renderer_p3d import Pytorch3dRenderer
from models.geometric_layers import orthographic_projection
import config
import os.path as osp

"""
python combined_demo.py --config config/config.yaml --input /z/syqian/articulation_data/step2_filtered_clips/TfqqWBlMj6Q_10_2880.mp4 --output example_opt329_output --save-obj --webvis
python combined_demo.py --config config/config.yaml --save-obj --webvis
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
    # height = 480
    # width = 640
    height = im.shape[0]
    width = im.shape[1]
    

    # bkgd mesh
    mesh_bkgd, uv_maps_bkgd = get_single_image_mesh_arti(plane_params, 1 - segmentations, img=im, height=height, width=width, webvis=args.webvis, reduce_size=reduce_size)
    #basename = 'bkgd'
    #save_obj(output_dir, basename+'_pred', mesh_bkgd, decimal_places=10, uv_maps=uv_maps_bkgd)

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
    # blend color for uv_maps
    # for i in range(len(uv_maps_list)):
        # color = np.array([[[252/255, 116/255, 81/255]]]) * (i/10 + 1/2)
        # uv_maps_list[i] = (uv_maps_list[i] / 255.0 + color) / 2
        # uv_maps_list[i] = (uv_maps_list[i] * 255.0).astype(np.uint8) 

    # uv_maps_list[-1] = (uv_maps_list[-1] / 255.0 + np.array([[[56/255, 207/255, 252/255]]])) / 2
    # uv_maps_list[-1] = (uv_maps_list[-1] * 255.0).astype(np.uint8) 
    # uv_maps_list[-2] = (uv_maps_list[-2] / 255.0 + np.array([[[56/255, 207/255, 252/255]]])) / 2
    # uv_maps_list[-2] = (uv_maps_list[-2] * 255.0).astype(np.uint8) 

    # meshes = pytorch3d.structures.join_meshes_as_batch([meshes, mesh_bkgd])
    # uv_maps_list = uv_maps_list + uv_maps_bkgd

    output_dir = os.path.join(args.output, 'frame_{:0>4}'.format(frame_id))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
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
    parser.add_argument('--save-obj', action='store_true')
    parser.add_argument('--webvis', action='store_true')
    parser.add_argument("--conf-threshold", default=0.7, type=float, help="confidence threshold")
    parser.add_argument('--checkpoint', default="data/models/ours/2020_02_29-18_30_01.pt", help='Path to pretrained checkpoint')
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

    # load smpl model
    mesh = Mesh(device=DEVICE)
    cmr = CMR(mesh, 5, 256, pretrained_checkpoint=args.checkpoint)
    smpl = SMPL()
    cmr = cmr.to(DEVICE)
    cmr.eval()
    smpl = smpl.to(DEVICE)
    
    # read video and run per-frame inference
    if args.input==None:
        data_path = "/z/syqian/articulation_data/step2_filtered_clips/"
        img_list = glob(data_path+"*.mp4")
        index = np.random.randint(len(img_list))
        args.input = img_list[index]
        print("Randomly select video", args.input)
        args.output = "example" + str(index) + "_output"

    video_path = args.input
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    frames = []
    preds = []
    # org_vis_list = []
    height, width = 480, 640

    # smpl_vis_array = []
    pred_pose_array = []
    pred_shape_array = []
    camera_translation_array = []
    pred_vertices_smpl_array = []


    for i, im in enumerate(tqdm(reader)):
        ori_height, ori_width, _ = im.shape
        ori_size = (ori_width, ori_height)
        # print(ori_size) #(1280, 720)

        # preprocess for pose estimation
        img, norm_img = process_image(im, input_res=SMPL_RES)
        norm_img = norm_img.to(DEVICE)

        #articulation
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
        # if args.output is not None:
        #     vis = Visualizer(im[:, :, ::-1])
        #     seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map)

            # surface normal
            # if len(p_instance.pred_boxes) == 0:
            #     # normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
            #     normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, height, width))
            # else:
            #     normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

            # combine visualization and generate output
            # combined_vis = np.concatenate((seg_pred, normal_vis), axis=1)
            # org_vis_list.append(combined_vis)


        #pose estimation

        with torch.no_grad():
            pred_vertices, pred_vertices_smpl, pred_camera, pred_pose, pred_shape = cmr(norm_img)
            pred_pose_array.append(pred_pose)
            pred_shape_array.append(pred_shape)
            # pred_keypoints_3d_smpl = smpl.get_joints(pred_vertices_smpl)
            # pred_keypoints_2d_smpl = orthographic_projection(pred_keypoints_3d_smpl, pred_camera.detach())[:, :, :2].cpu().data.numpy()
        
        # Calculate camera parameters for rendering
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*config.FOCAL_LENGTH/(config.INPUT_RES * pred_camera[:,0] +1e-9)],dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()
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
        # Ct = inv(Tt) @ inv(Rt)
        
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
        
        # camera_translation_array.append(camera_translation)
        pred_vertices_smpl_array.append(pred_vertices_smpl)
        # img = img.permute(1,2,0).cpu().numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # smpl_vis_array.append(img)

    reader.close()

    # temporal optimization
    planes = track_planes(preds)
    opt_preds = optimize_planes(preds, planes, '3dc', frames=frames)

    # video visualization in 2D
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    writer = imageio.get_writer(os.path.join(args.output, '{}.mp4'.format('output')), fps=fps)
    for i, im in (enumerate(frames)):
        p_instance = opt_preds[i]
        # org_vis = org_vis_list[i]
        # smpl_ori = smpl_vis_array[i]

        vis = Visualizer(im)

        seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map)

        # surface normal
        if len(p_instance.pred_boxes) == 0:
            # normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
            normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, height, width))
        else:
            normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

        # combine visualization and generate output

        
        seg_pred = cv2.resize(seg_pred, ori_size)
        normal_vis = cv2.resize(normal_vis, ori_size)

        # save plane masks
        mask_plane_pred = p_instance.pred_masks.cpu()
        mask_plane_pred = (mask_plane_pred > 0.5).float()
        mask_plane_pred = np.asarray(mask_plane_pred)
        mask_plane_pred = np.moveaxis(mask_plane_pred, 0, -1)
        print(mask_plane_pred.shape)
        try:
            mask_plane_pred = cv2.resize(mask_plane_pred, ori_size)
            mask_plane_pred = mask_plane_pred[:,:,0]
        except:
            print("No detected plane mask")
        print(mask_plane_pred.shape)
        mask_plane_outfile = args.output+'/frame'+str(i)+'planemask_preds'+'.txt'
        np.savetxt(mask_plane_outfile, mask_plane_pred)
        print("The .txt should be saved in", mask_plane_outfile)
        

        seg_pred = process_resize(seg_pred, SMPL_RES)
        normal_vis = process_resize(normal_vis, SMPL_RES)
        seg_pred = cv2.cvtColor(seg_pred, cv2.COLOR_RGB2BGR)
        # normal_vis = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
        

        renderer = Pytorch3dRenderer(
                    img_size=SMPL_RES, 
                    mesh_color=colors['light_purple'],
                    focal_length=config.FOCAL_LENGTH,
                    camera_t=np.array((0,0,0)).reshape(1,3),
                    device=DEVICE)
        
        smpl_obj_name = args.output+'/frame'+str(i)+'_preds'+'.obj'
        img_smpl = renderer.render(pred_vertices_smpl_array[i], mesh.faces.cpu().numpy(), seg_pred/255, True)
        only_smpl = renderer.render(pred_vertices_smpl_array[i], mesh.faces.cpu().numpy(), seg_pred/255, False, True, smpl_obj_name)
        print("The .obj should be saved in", smpl_obj_name)


        combined = (np.concatenate((img_smpl, only_smpl), axis=1)*255).astype(np.uint8)
        

        combined_vis = np.concatenate((combined, normal_vis), axis=1)
        writer.append_data(combined_vis)

    if args.save_obj:
        # select frame_ids you want to visualize
        frame_ids = [0, 30, 60, 89]
        for frame_id in frame_ids:
            save_obj_model(args, opt_preds, frames, frame_id)


if __name__ == "__main__":
    main()
