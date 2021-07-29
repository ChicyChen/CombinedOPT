
import cv2
import numpy as np
import imageio
import os
import torch
import torch.nn.functional as F
from glob import glob
import random
from scipy.stats import linregress
import pycocotools.mask as mask_util
import pdb

import pytorch3d
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures,
    TexturesUV,
    TexturesVertex
)

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures.boxes import pairwise_iou, pairwise_ioa
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from detectron2.structures import Boxes, BoxMode, Instances

from .vis import get_pcd, project2D
from planercnn.visualization import get_gt_labeled_seg, get_labeled_seg
from planercnn.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis


def optimize_planes_average(preds, planes):
    for plane in planes:
        std_axes = []
        for idx in plane['ids']:
            box_id = plane['ids'][idx]

            # transform pred_axis to std_axis
            p_instance = preds[idx]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            img_centers = torch.FloatTensor(np.array([[320, 240]]))
            std_axis = axis_to_angle_offset(pts.numpy().tolist(), img_centers)
            std_axis = std_axis[:, :3]
            std_axis = std_axis[box_id:(box_id+1)]
            std_axes.append(std_axis)

        std_axes = torch.cat(std_axes)
        std_axis = std_axes.mean(axis=0)
        plane['std_axis'] = std_axis

    opt_preds = []
    for idx, p_instance in enumerate(preds):
        pred_boxes = p_instance.pred_boxes
        chosen = [False for _ in range(pred_boxes.tensor.shape[0])]
        for plane in planes:
            if idx not in plane['ids']:
                continue
            box_id = plane['ids'][idx]
            chosen[box_id] = True
            p_instance.pred_rot_axis[box_id] = plane['std_axis']

        opt_preds.append(p_instance)

    return opt_preds


def optimize_planes_3d(preds, planes):
    for plane in planes:
        id_list = list(plane['ids'].keys())
        clusters = []
        for _ in range(5):
            # select a random frame
            select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = preds[select_idx]

            # fetch rotation axis and pcd
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)[0]
            offset = torch.norm(pred_plane, p=2)
            verts_axis = pts[box_id].reshape(-1, 2)
            verts_axis_3d = get_pcd(verts_axis, normal, offset)
            dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            pcd = get_pcd(verts, normal, offset)  # , focal_length)
            pcd = pcd.float().cuda()

            # assign transformations
            t1 = pytorch3d.transforms.Transform3d().translate(
                verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
            t1 = t1.cuda()
            angles = torch.FloatTensor(
                np.arange(-np.pi/2, 0.1, np.pi/30)[:, np.newaxis])
            axis_angles = angles * dir_vec
            rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
            t2 = pytorch3d.transforms.Rotate(rot_mats)
            t2 = t2.cuda()
            t3 = t1.inverse()
            pcd_trans = t3.transform_points(pcd)
            pcd_trans = t2.transform_points(pcd_trans)
            pcd_trans = t1.transform_points(pcd_trans)

            # project pcd to 2d space
            proj_masks = []
            for i in range(pcd_trans.shape[0]):
                this_pcd = pcd_trans[i]
                proj_verts = project2D(this_pcd)
                proj_verts = proj_verts.long()
                proj_verts = proj_verts.flip(1)
                proj_mask = torch.zeros_like(pred_mask).cuda()
                proj_verts[:, 0][proj_verts[:, 0] >=
                                 proj_mask.shape[0]] = proj_mask.shape[0] - 1
                proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
                proj_verts[:, 1][proj_verts[:, 1] >=
                                 proj_mask.shape[1]] = proj_mask.shape[1] - 1
                proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
                full_mask = torch.ones_like(pred_mask).cuda()
                proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                          ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
                proj_mask = proj_mask.unsqueeze(0)
                proj_masks.append(proj_mask)

            proj_masks = torch.cat(proj_masks)

            # all ious
            #ious = []
            cluster_inliners = []
            cluster_angles = []
            cluster_ious = []
            for idx in id_list:
                box_id = plane['ids'][idx]
                p_instance = preds[idx]
                pred_mask = p_instance.pred_masks[box_id]
                pred_mask = pred_mask.unsqueeze(0)
                pred_mask = pred_mask.cuda()

                intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
                intersec = intersec.sum(2).sum(1)
                un = (pred_mask > 0.5) | (proj_masks > 0.5)
                un = un.sum(2).sum(1)
                ious = intersec / un

                angle_id = ious.argmax()
                try:
                    angle = angles[angle_id][0]
                except:
                    pdb.set_trace()
                    pass

                if ious.max() > 0.5:
                    cluster_inliners.append(idx)
                    #id_list.remove(idx)
                    cluster_angles.append(angle)
                    cluster_ious.append(ious.max().cpu().item())

            cluster_angles = torch.FloatTensor(cluster_angles)
            cluster = {
                'center_id': select_idx,
                'inliners': cluster_inliners,
                'angles': cluster_angles,
                'ious': cluster_ious
            }
            # print(cluster)

            clusters.append(cluster)

        # now we have all clusters
        # determine the dominant cluster
        rsqs = []
        for cluster in clusters:
            if len(cluster['inliners']) < 5:
                rsqs.append(0.0)
                continue
            reg_results = linregress(
                range(cluster['angles'].shape[0]), cluster['angles'])

            #if reg_results.slope < 0.01:
            #    rsq = 0.0
            #else:
            rsq = reg_results.rvalue ** 2
            rsqs.append(rsq)

        #cluster_cnts = np.array([len(cluster['inliners']) for cluster in clusters])
        #cluster_id = cluster_cnts.argmax()
        #final_cluster = clusters[cluster_id]

        # from the cluster, infer the articulation model
        #reg_results = linregress(range(final_cluster['angles'].shape[0]), final_cluster['angles'])
        rsqs = np.array(rsqs)

        # pdb.set_trace()

        #if rsqs.max() < 0:  # impossible
        if rsqs.max() < 0.3:
            plane['has_rot'] = False
            continue
        else:
            plane['has_rot'] = True

        # then determine the regularized mask and rot axis
        try:
            final_cluster = clusters[rsqs.argmax()]
        except:
            pdb.set_trace()
            pass
        select_idx = final_cluster['center_id']
        box_id = plane['ids'][select_idx]
        p_instance = preds[select_idx]
        std_axis = p_instance.pred_rot_axis[box_id]

        # fetch rotation axis and pcd
        pred_mask = p_instance.pred_masks[box_id]
        pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
        pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
        pred_plane[:, 1] = - pred_plane[:, 1]
        pred_box_centers = p_instance.pred_boxes.get_centers()
        pts = angle_offset_to_axis(p_instance.pred_rot_axis, pred_box_centers)
        verts = pred_mask.nonzero().flip(1)
        normal = F.normalize(pred_plane, p=2)[0]
        offset = torch.norm(pred_plane, p=2)
        verts_axis = pts[box_id].reshape(-1, 2)
        verts_axis_3d = get_pcd(verts_axis, normal, offset)
        dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        pcd = get_pcd(verts, normal, offset)  # , focal_length)
        pcd = pcd.float().cuda()

        # assign transformations
        t1 = pytorch3d.transforms.Transform3d().translate(
            verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
        t1 = t1.cuda()
        angles = torch.FloatTensor(
            np.arange(-np.pi/2, 0.1, np.pi/30)[:, np.newaxis])
        axis_angles = angles * dir_vec
        rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
        t2 = pytorch3d.transforms.Rotate(rot_mats)
        t2 = t2.cuda()
        t3 = t1.inverse()
        pcd_trans = t3.transform_points(pcd)
        pcd_trans = t2.transform_points(pcd_trans)
        pcd_trans = t1.transform_points(pcd_trans)

        # project pcd to 2d space
        proj_masks = []
        for i in range(pcd_trans.shape[0]):
            this_pcd = pcd_trans[i]
            proj_verts = project2D(this_pcd)
            proj_verts = proj_verts.long()
            proj_verts = proj_verts.flip(1)
            proj_mask = torch.zeros_like(pred_mask).cuda()
            proj_verts[:, 0][proj_verts[:, 0] >=
                             proj_mask.shape[0]] = proj_mask.shape[0] - 1
            proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
            proj_verts[:, 1][proj_verts[:, 1] >=
                             proj_mask.shape[1]] = proj_mask.shape[1] - 1
            proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
            full_mask = torch.ones_like(pred_mask).cuda()
            proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                      ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
            proj_mask = proj_mask.unsqueeze(0)
            proj_masks.append(proj_mask)

        proj_masks = torch.cat(proj_masks)

        plane['reg_masks'] = {}
        for idx in plane['ids']:
            box_id = plane['ids'][idx]
            p_instance = preds[idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_mask = pred_mask.unsqueeze(0)
            pred_mask = pred_mask.cuda()

            intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
            intersec = intersec.sum(2).sum(1)
            un = (pred_mask > 0.5) | (proj_masks > 0.5)
            un = un.sum(2).sum(1)
            ious = intersec / un
            angle_id = ious.argmax()

            plane['reg_masks'][idx] = proj_masks[angle_id].cpu()

        plane['std_axis'] = std_axis

    opt_preds = []
    for idx, p_instance in enumerate(preds):
        pred_boxes = p_instance.pred_boxes
        chosen = [False for _ in range(pred_boxes.tensor.shape[0])]
        for plane in planes:
            if idx not in plane['ids']:
                continue
            box_id = plane['ids'][idx]
            if not plane['has_rot']:
                chosen[box_id] = False
                continue
            chosen[box_id] = True
            p_instance.pred_rot_axis[box_id] = plane['std_axis']
            continue
            if plane['reg_masks'][idx] is not None:
                p_instance.pred_masks[box_id] = plane['reg_masks'][idx]
                # bbox
                #mask = GenericMask(plane['reg_masks'][idx].numpy(), 480, 640)
                #box_tensor = p_instance.pred_boxes.tensor
                #box_tensor[box_id] = torch.FloatTensor(mask.bbox())
                #p_instance.pred_boxes = Boxes(box_tensor)

        chosen = np.array(chosen, dtype=bool)
        no_chosen = np.logical_not(chosen)

        # soft filter
        new_instance = Instances(p_instance.image_size)
        scores = np.copy(p_instance.scores)
        scores[no_chosen] = scores[no_chosen] * 0.6
        new_instance.scores = scores
        new_instance.pred_boxes = p_instance.pred_boxes
        new_instance.pred_planes = p_instance.pred_planes
        new_instance.pred_rot_axis = p_instance.pred_rot_axis
        new_instance.pred_tran_axis = p_instance.pred_tran_axis
        new_instance.pred_masks = p_instance.pred_masks
        new_instance.pred_classes = p_instance.pred_classes

        opt_preds.append(new_instance)

    return opt_preds


def optimize_planes_3dc(preds, planes, frames=None):
    """
    optimizatino w/ 3d clustering
    """
    for plane in planes:
        #best_idx = -1
        #min_mean_loss = 100000

        id_list = list(plane['ids'].keys())
        clusters = []
        for _ in range(5):
            if len(id_list) == 0:
                break

            # select a random frame
            select_idx = random.choice(id_list)
            box_id = plane['ids'][select_idx]
            p_instance = preds[select_idx]

            # fetch rotation axis and pcd
            pred_mask = p_instance.pred_masks[box_id]
            pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
            pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
            pred_plane[:, 1] = - pred_plane[:, 1]
            pred_box_centers = p_instance.pred_boxes.get_centers()
            pts = angle_offset_to_axis(
                p_instance.pred_rot_axis, pred_box_centers)
            verts = pred_mask.nonzero().flip(1)
            normal = F.normalize(pred_plane, p=2)[0]
            offset = torch.norm(pred_plane, p=2)
            verts_axis = pts[box_id].reshape(-1, 2)
            verts_axis_3d = get_pcd(verts_axis, normal, offset)
            dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            pcd = get_pcd(verts, normal, offset)  # , focal_length)
            pcd = pcd.float().cuda()

            # assign transformations
            t1 = pytorch3d.transforms.Transform3d().translate(
                verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
            t1 = t1.cuda()
            angles = torch.FloatTensor(
                np.arange(-np.pi/2, 0.1, np.pi/30)[:, np.newaxis])
            axis_angles = angles * dir_vec
            rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
            t2 = pytorch3d.transforms.Rotate(rot_mats)
            t2 = t2.cuda()
            t3 = t1.inverse()
            pcd_trans = t3.transform_points(pcd)
            pcd_trans = t2.transform_points(pcd_trans)
            pcd_trans = t1.transform_points(pcd_trans)

            # project pcd to 2d space
            proj_masks = []
            for i in range(pcd_trans.shape[0]):
                this_pcd = pcd_trans[i]
                proj_verts = project2D(this_pcd)
                proj_verts = proj_verts.long()
                proj_verts = proj_verts.flip(1)
                proj_mask = torch.zeros_like(pred_mask).cuda()
                proj_verts[:, 0][proj_verts[:, 0] >=
                                 proj_mask.shape[0]] = proj_mask.shape[0] - 1
                proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
                proj_verts[:, 1][proj_verts[:, 1] >=
                                 proj_mask.shape[1]] = proj_mask.shape[1] - 1
                proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
                full_mask = torch.ones_like(pred_mask).cuda()
                proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                          ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
                proj_mask = proj_mask.unsqueeze(0)
                proj_masks.append(proj_mask)

            proj_masks = torch.cat(proj_masks)

            # all ious
            #ious = []
            cluster_inliners = []
            cluster_angles = []
            cluster_ious = []
            for idx in id_list:
                box_id = plane['ids'][idx]
                p_instance = preds[idx]
                pred_mask = p_instance.pred_masks[box_id]
                pred_mask = pred_mask.unsqueeze(0)
                pred_mask = pred_mask.cuda()

                intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
                intersec = intersec.sum(2).sum(1)
                un = (pred_mask > 0.5) | (proj_masks > 0.5)
                un = un.sum(2).sum(1)
                ious = intersec / un

                angle_id = ious.argmax()
                try:
                    angle = angles[angle_id][0]
                except:
                    pdb.set_trace()
                    pass

                if ious.max() > 0.5:
                    cluster_inliners.append(idx)
                    id_list.remove(idx)
                    cluster_angles.append(angle)
                    cluster_ious.append(ious.max().cpu().item())

            cluster_angles = torch.FloatTensor(cluster_angles)
            cluster = {
                'center_id': select_idx,
                'inliners': cluster_inliners,
                'angles': cluster_angles,
                'ious': cluster_ious
            }
            # print(cluster)

            clusters.append(cluster)

        # now we have all clusters
        # determine the dominant cluster
        rsqs = []
        for cluster in clusters:
            if len(cluster['inliners']) < 5:
                rsqs.append(0.0)
                continue
            reg_results = linregress(
                range(cluster['angles'].shape[0]), cluster['angles'])
            
            rsq = reg_results.rvalue ** 2
            #if reg_results.slope < 0.01:
            #    rsq = 0.0
            #else:
            #    rsq = reg_results.rvalue ** 2
            rsqs.append(rsq)

        #cluster_cnts = np.array([len(cluster['inliners']) for cluster in clusters])
        #cluster_id = cluster_cnts.argmax()
        #final_cluster = clusters[cluster_id]

        # from the cluster, infer the articulation model
        #reg_results = linregress(range(final_cluster['angles'].shape[0]), final_cluster['angles'])
        rsqs = np.array(rsqs)

        # pdb.set_trace()

        #if rsqs.max() < 0:  # impossible
        if rsqs.max() < 0.3:
            plane['has_rot'] = False
            continue
        else:
            plane['has_rot'] = True

        # then determine the regularized mask and rot axis
        try:
            final_cluster = clusters[rsqs.argmax()]
        except:
            pdb.set_trace()
            pass
        select_idx = final_cluster['center_id']
        box_id = plane['ids'][select_idx]
        p_instance = preds[select_idx]
        std_axis = p_instance.pred_rot_axis[box_id]

        # fetch rotation axis and pcd
        pred_mask = p_instance.pred_masks[box_id]
        pred_plane = p_instance.pred_planes[box_id:(box_id + 1)].clone()
        pred_plane[:, [1, 2]] = pred_plane[:, [2, 1]]
        pred_plane[:, 1] = - pred_plane[:, 1]
        pred_box_centers = p_instance.pred_boxes.get_centers()
        pts = angle_offset_to_axis(p_instance.pred_rot_axis, pred_box_centers)
        verts = pred_mask.nonzero().flip(1)
        normal = F.normalize(pred_plane, p=2)[0]
        offset = torch.norm(pred_plane, p=2)
        verts_axis = pts[box_id].reshape(-1, 2)
        verts_axis_3d = get_pcd(verts_axis, normal, offset)
        dir_vec = verts_axis_3d[1] - verts_axis_3d[0]
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        pcd = get_pcd(verts, normal, offset)  # , focal_length)
        pcd = pcd.float().cuda()

        # assign transformations
        t1 = pytorch3d.transforms.Transform3d().translate(
            verts_axis_3d[0][0], verts_axis_3d[0][1], verts_axis_3d[0][2])
        t1 = t1.cuda()
        angles = torch.FloatTensor(
            np.arange(-np.pi/2, 0.1, np.pi/30)[:, np.newaxis])
        axis_angles = angles * dir_vec
        rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
        t2 = pytorch3d.transforms.Rotate(rot_mats)
        t2 = t2.cuda()
        t3 = t1.inverse()
        pcd_trans = t3.transform_points(pcd)
        pcd_trans = t2.transform_points(pcd_trans)
        pcd_trans = t1.transform_points(pcd_trans)

        # project pcd to 2d space
        proj_masks = []
        for i in range(pcd_trans.shape[0]):
            this_pcd = pcd_trans[i]
            proj_verts = project2D(this_pcd)
            proj_verts = proj_verts.long()
            proj_verts = proj_verts.flip(1)
            proj_mask = torch.zeros_like(pred_mask).cuda()
            proj_verts[:, 0][proj_verts[:, 0] >=
                             proj_mask.shape[0]] = proj_mask.shape[0] - 1
            proj_verts[:, 0][proj_verts[:, 0] < 0] = 0
            proj_verts[:, 1][proj_verts[:, 1] >=
                             proj_mask.shape[1]] = proj_mask.shape[1] - 1
            proj_verts[:, 1][proj_verts[:, 1] < 0] = 0
            full_mask = torch.ones_like(pred_mask).cuda()
            proj_mask[proj_verts[:, 0], proj_verts[:, 1]
                      ] = full_mask[proj_verts[:, 0], proj_verts[:, 1]]
            proj_mask = proj_mask.unsqueeze(0)
            proj_masks.append(proj_mask)

        proj_masks = torch.cat(proj_masks)

        plane['reg_masks'] = {}
        for idx in plane['ids']:
            box_id = plane['ids'][idx]
            p_instance = preds[idx]
            pred_mask = p_instance.pred_masks[box_id]
            pred_mask = pred_mask.unsqueeze(0)
            pred_mask = pred_mask.cuda()

            intersec = (pred_mask > 0.5) & (proj_masks > 0.5)
            intersec = intersec.sum(2).sum(1)
            un = (pred_mask > 0.5) | (proj_masks > 0.5)
            un = un.sum(2).sum(1)
            ious = intersec / un
            angle_id = ious.argmax()

            plane['reg_masks'][idx] = proj_masks[angle_id].cpu()

        plane['std_axis'] = std_axis

    opt_preds = []
    for idx, p_instance in enumerate(preds):
        pred_boxes = p_instance.pred_boxes
        
        chosen = [False for _ in range(pred_boxes.tensor.shape[0])]

        # do not filter out translation
        pred_classes = p_instance.pred_classes
        for i in range(pred_classes.size):
            if pred_classes[i] == 1:
                chosen[i] = True

        for plane in planes:
            if idx not in plane['ids']:
                continue
            box_id = plane['ids'][idx]
            if not plane['has_rot']:
                chosen[box_id] = False
                continue
            chosen[box_id] = True
            p_instance.pred_rot_axis[box_id] = plane['std_axis']
            continue
            if plane['reg_masks'][idx] is not None:
                p_instance.pred_masks[box_id] = plane['reg_masks'][idx]
                # bbox
                #mask = GenericMask(plane['reg_masks'][idx].numpy(), 480, 640)
                #box_tensor = p_instance.pred_boxes.tensor
                #box_tensor[box_id] = torch.FloatTensor(mask.bbox())
                #p_instance.pred_boxes = Boxes(box_tensor)

        chosen = np.array(chosen, dtype=bool)
        no_chosen = np.logical_not(chosen)

        # soft filter
        new_instance = Instances(p_instance.image_size)
        scores = np.copy(p_instance.scores)
        scores[no_chosen] = scores[no_chosen] * 0.6
        new_instance.scores = scores
        new_instance.pred_boxes = p_instance.pred_boxes
        new_instance.pred_planes = p_instance.pred_planes
        new_instance.pred_rot_axis = p_instance.pred_rot_axis
        new_instance.pred_tran_axis = p_instance.pred_tran_axis
        new_instance.pred_masks = p_instance.pred_masks
        new_instance.pred_classes = p_instance.pred_classes

        opt_preds.append(new_instance)

    return opt_preds


def optimize_planes(preds, planes, method, frames=None):
    if method == 'average':
        return optimize_planes_average(preds, planes)
    elif method == '3d':
        return optimize_planes_3d(preds, planes)
    elif method == '3dc':
        return optimize_planes_3dc(preds, planes, frames=frames)
    else:
        raise NotImplementedError


def track_planes(preds):
    # tracking
    planes = []
    for idx, p_instance in enumerate(preds):
        plane = {}
        pred_classes = p_instance.pred_classes
        pred_boxes = p_instance.pred_boxes
        for box_id in range(p_instance.pred_boxes.tensor.shape[0]):
            current_box = pred_boxes[box_id]
            if pred_classes[box_id] == 1:
                continue

            has_overlap = False
            for plane in planes:
                if idx - plane['latest_frame'] > 5:
                    continue
                plane_box = plane['bbox']
                iou = pairwise_iou(current_box, plane_box)
                if iou.item() > 0.5:
                    # record it
                    has_overlap = True
                    plane['ids'][idx] = box_id
                    plane['bbox'] = current_box
                    plane['latest_frame'] = idx
                    break

            if not has_overlap:  # create new box
                plane = {
                    'bbox': current_box,
                    'ids': {
                        idx: box_id,
                    },
                    'latest_frame': idx,
                }
                planes.append(plane)

    # filter short sequence
    filter_planes = []
    for plane in planes:
        if len(plane['ids']) < 10:
            continue
        filter_planes.append(plane)
    planes = filter_planes

    return planes
