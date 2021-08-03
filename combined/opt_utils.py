# Structure definition and help functions definition of the opt system

import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import cv2
from tqdm.auto import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene,join_meshes_as_batch
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.transforms import RotateAxisAngle, Scale
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    BlendParams,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments,
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex
)

# Util function for optimization
from pytorch3d.loss import chamfer_distance

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend

from utils.imutils import preprocess_generic
from models import SMPL


# Note: need several files existed in advance.
"""
python opt_single_img.py
"""


##################################################################
# Opt System Structure Definition
##################################################################

def compute_transformation_persp(
    meshes, translations, rotations=None, intrinsic_scales=None
):
    """
    Computes the 3D transformation.
    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        translations (B x 1 x 3).
        rotations (B x 3 x 3).
        intrinsic_scales (B).
    Returns:
        vertices (B x V x 3).
    """

    # print("meshes size", meshes.ndimension()) # 3
    # print("translations", translations, translations.shape[0]) # B = 1

    B = translations.shape[0]
    device = meshes.device
    translations = translations.to(device)
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotations = rotations.to(device)
    if intrinsic_scales is None:
        intrinsic_scales = torch.ones(B).to(device)

    # verts_rot = torch.matmul(meshes.detach().clone(), rotations)
    # verts_trans = verts_rot + translations
    # verts_final = intrinsic_scales.view(-1, 1, 1) * verts_trans

    # verts_trans = meshes.detach().clone() + translations
    verts_trans = meshes + translations

    verts_rot = torch.matmul(verts_trans, rotations)
    verts_final = intrinsic_scales.view(-1, 1, 1) * verts_rot
    

    return verts_final


class Losses(object):
    def __init__(
        self,
        gene_mask,
        renderer,
        renderer_silhouette,
        device,
        object_textures,
        smpl_textures,
        textures,
        masks_person,
        masks_object,
        target_person,
        target_object,
        # K_rois,
        # class_name,
        # labels_person,
        # labels_object,
        # interaction_map_parts,
    ):
        self.gene_mask = gene_mask
        self.renderer = renderer
        self.renderer_silhouette = renderer_silhouette
        self.device = device
        self.object_textures = object_textures,
        self.smpl_textures = smpl_textures,
        self.textures = textures
        self.masks_person = masks_person
        self.masks_object = masks_object
        self.target_person = target_person
        self.target_object = target_object
        self.mse = nn.MSELoss()

    def compute_ordinal_depth_loss(self, smpl_pred_mask, obj_pred_mask, smpl_depth, obj_depth):
        device = self.device
        loss_ordinal_depth = torch.tensor(0.0).float().cuda()
        
        gt_person = self.masks_person
        gt_object = self.masks_object

        front_person_gt = gt_person > 0
        front_object_gt = gt_object > 0

        front_person_pred = smpl_pred_mask[0] > 0
        front_object_pred = obj_pred_mask[0] > 0

        both_pred = front_person_pred & front_object_pred

        false_person = front_person_gt & front_object_pred & both_pred
        false_object = front_object_gt & front_person_pred & both_pred

        false_person_dists = torch.clamp(smpl_depth - obj_depth, min=0.0, max=0.5) * false_person.float()
        false_object_dists = torch.clamp(obj_depth - smpl_depth, min=0.0, max=0.5) * false_object.float()

        # l_p = torch.sum(torch.log(1 + torch.exp(false_person_dists)))
        # l_o = torch.sum(torch.log(1 + torch.exp(false_object_dists)))
        # quite fast to get overflow

        l_p = torch.sum(false_person_dists / 1e3)
        l_o = torch.sum(false_object_dists / 1e3)

        loss_ordinal_depth = loss_ordinal_depth + l_p + l_o
        loss_ordinal_depth /= 2

        print(l_p, l_o, loss_ordinal_depth)
        
        return loss_ordinal_depth

    def compute_sil_loss(self, smpl_pred_mask):
        device = self.device
        loss_sil = torch.tensor(0.0).float().cuda()

        # person_image = self.target_person * smpl_pred_mask
        # object_image = self.target_object * obj_pred_mask

        person_image = smpl_pred_mask * self.gene_mask
        # person_image = smpl_pred_mask
        # object_image = obj_pred_mask

        l_p = torch.sum((person_image - self.masks_person) ** 2 / 1e3) 
        # l_o = torch.sum((object_image - self.masks_object) ** 2 / 1e3) 

        # loss_sil = loss_sil + l_p +l_o
        # loss_sil /= 2

        # return loss_sil
        print("sil_loss", l_p)

        return l_p
        

    def compute_inter_loss(self, verts_object, verts_person):
        device = self.device
        loss_inter = torch.tensor(0.0).float().cuda()

        loss_inter, loss_normals = chamfer_distance(verts_object.to(device),
                                                    verts_person.to(device))
        print("inter_loss", loss_inter)

        return loss_inter

    def compute_centroid_loss(self, verts_object, verts_person):
        device = self.device
        loss_centroid = torch.tensor(0.0).float().cuda()

        loss_centroid = self.mse(verts_person[0].mean(0), verts_object[0].mean(0))
        # print(loss_centroid)

        return loss_centroid

class simplePHOSA(nn.Module):
    def __init__(
        self,
        in_w,
        gene_mask,
        translations_person,
        rotations_person,
        translations_object,
        rotations_object,
        masks_person,
        masks_object,
        target_person,
        target_object,
        verts_object,
        faces_object,
        verts_person,
        faces_person,
        # masks_object,
        # masks_person,
        # K_rois,
        # target_masks,
        # labels_person,
        # labels_object,
        # interaction_map_parts,
        # class_name,
        object_textures,
        smpl_textures,
        textures,
        int_scale_init=1.0,
        device=torch.device("cuda"),
    ):
        super(simplePHOSA, self).__init__()
        self.device = device
        self.object_textures = object_textures
        self.smpl_textures = smpl_textures
        self.textures = textures
        translation_person = translations_person.detach().clone()
        self.translations_person = nn.Parameter(translation_person, requires_grad=True)
        rotations_person = rotations_person.detach().clone()
        self.rotations_person = nn.Parameter(rotations_person, requires_grad=False)
        translation_object = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_object, requires_grad=False)
        rotations_object = rotations_object.detach().clone()
        self.rotations_object = nn.Parameter(rotations_object, requires_grad=False)

        self.register_buffer("gene_mask", gene_mask)

        self.register_buffer("masks_object", masks_object)
        self.register_buffer("masks_person", masks_person)
        self.register_buffer("target_object", target_object)
        self.register_buffer("target_person", target_person)

        self.register_buffer("verts_object", verts_object)
        self.register_buffer("verts_person", verts_person)
        self.register_buffer("faces_object", faces_object)
        self.register_buffer("faces_person", faces_person)
        # print("initial", faces_person.unsqueeze(0).size())
        # self.register_buffer("cams_person", cams_person)

        self.int_scales_object = nn.Parameter(
            int_scale_init * torch.ones(1).float(), requires_grad=False
        )
        self.int_scale_object_mean = nn.Parameter(
            torch.tensor(int_scale_init).float(), requires_grad=False
        )
        self.int_scales_person = nn.Parameter(
            int_scale_init * torch.ones(1).float(), requires_grad=False
        )
        self.int_scale_person_mean = nn.Parameter(
            torch.tensor(1.0).float().cuda(), requires_grad=False
        )

        self.cuda()

        # Set up the differentiable render
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        cameras = FoVPerspectiveCameras(device=device)           
        raster_settings = RasterizationSettings(
            image_size = (in_w, in_w),
            blur_radius = 0,
            faces_per_pixel = 1,
        )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (0,0,0))
        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )

        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size = (in_w, in_w),
            blur_radius = np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
        )
        # Silhouette renderer 
        renderer_silhouette = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        self.renderer_silhouette = renderer_silhouette

        self.losses = Losses(
            gene_mask = gene_mask,
            renderer = self.renderer,
            renderer_silhouette = self.renderer_silhouette,
            device = self.device,
            object_textures = self.object_textures,
            smpl_textures = self.smpl_textures,
            textures = self.textures,
            masks_person = self.masks_person,
            masks_object = self.masks_object,
            target_person = self.target_person,
            target_object = self.target_object,
            # ref_mask=self.ref_mask,
            # keep_mask=self.keep_mask,
            # K_rois=self.K_rois,
            # interaction_map_parts=interaction_map_parts,
            # labels_person=labels_person,
            # labels_object=labels_object,
            # class_name=class_name,
        )
    
    def get_verts_object(self):
        return compute_transformation_persp(
            meshes=self.verts_object,
            translations=self.translations_object,
            rotations=self.rotations_object,
            intrinsic_scales=self.int_scales_object,
        )

    def get_verts_person(self):
        return compute_transformation_persp(
            meshes=self.verts_person,
            translations=self.translations_person,
            rotations=self.rotations_person,
            intrinsic_scales=self.int_scales_person,
        )

    def compute_sil_loss(self, verts_object, verts_person):
        # generte smpl mask
        smpl_mesh = Meshes(verts=self.get_verts_person().to(self.device), 
                    faces=self.faces_person.to(self.device), 
                    textures=self.smpl_textures)
        smpl_pred, _ = self.renderer_silhouette(smpl_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]]))
        smpl_pred_silhouette = smpl_pred[..., 3]
        """
        # generte obj mask
        obj_mesh = Meshes(verts=self.get_verts_object().to(self.device), 
                    faces=self.faces_object.to(self.device), 
                    textures=self.object_textures)
        obj_pred, _ = self.renderer_silhouette(obj_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]]))
        obj_pred_silhouette = obj_pred[..., 3]

        return self.losses.compute_sil_loss(
                                smpl_pred_mask = smpl_pred_silhouette,
                                obj_pred_mask = obj_pred_silhouette
                            )
        """

        return self.losses.compute_sil_loss(
                                smpl_pred_mask = smpl_pred_silhouette
                            )


    def compute_ordinal_depth_loss(self, verts_object, verts_person):
        # generte smpl mask
        smpl_mesh = Meshes(verts=verts_person.to(self.device), 
                    faces=self.faces_person.to(self.device), 
                    textures=self.smpl_textures)
        smpl_pred, smpl_frag = self.renderer_silhouette(smpl_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]]))
        smpl_pred_silhouette = smpl_pred[..., 3]
        smpl_depth = smpl_frag.zbuf[0, :, :, 0]
        # print(smpl_depth.size())
        # generte obj mask
        obj_mesh = Meshes(verts=verts_object.to(self.device), 
                    faces=self.faces_object.to(self.device), 
                    textures=self.object_textures)
        obj_pred, obj_frag = self.renderer_silhouette(obj_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]]))
        obj_pred_silhouette = obj_pred[..., 3]
        obj_depth = obj_frag.zbuf[0, :, :, 0]
        # print(obj_depth.size())

        return self.losses.compute_ordinal_depth_loss(
                    smpl_pred_mask = smpl_pred_silhouette,
                    obj_pred_mask = obj_pred_silhouette,
                    smpl_depth = smpl_depth,
                    obj_depth = obj_depth
                )


    def forward(self, loss_weights=None):
        """
        If a loss weight is zero, that loss isn't computed (to avoid unnecessary
        compute).
        """
        loss = torch.tensor(0.0).float().cuda()

        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()
        print(torch.isnan(verts_person).any())

        if loss_weights["lw_ordinal_depth"] > 0:
            loss_ordinal_depth = self.compute_ordinal_depth_loss(
                                    verts_object = verts_object,
                                    verts_person = verts_person
                                ) * loss_weights["lw_ordinal_depth"]
            loss += loss_ordinal_depth
            
        if loss_weights["lw_sil"] > 0:
            loss_sil = self.compute_sil_loss(
                            verts_object = verts_object,
                            verts_person = verts_person
                        ) * loss_weights["lw_sil"]
            loss += loss_sil
        
        if loss_weights["lw_inter"] > 0:
            loss_inter = self.losses.compute_inter_loss(
                            verts_object = verts_object, 
                            verts_person = verts_person
                        ) * loss_weights["lw_inter"]
            loss += loss_inter

        if loss_weights["lw_centroid"] > 0:
            loss_centroid = self.losses.compute_centroid_loss(
                            verts_object = verts_object, 
                            verts_person = verts_person
                        ) * loss_weights["lw_centroid"]
            loss += loss_centroid

        return loss

    
    def get_parameters(self):
        """
        Computes a json-serializable dictionary of optimized parameters.
        Returns:
            parameters (dict): Dictionary mapping parameter name to list.
        """
        parameters = {
            "scales_object": self.int_scales_object,
            "scales_person": self.int_scales_person,
            "rotations_object": self.rotations_object,
            "rotations_person": self.rotations_person,
            "translations_object": self.translations_object,
            "translations_person": self.translations_person,
        }
        for k, v in parameters.items():
            parameters[k] = v.detach().cpu().numpy().tolist()
        return parameters


class smplPHOSA(nn.Module):
    def __init__(
        self,
        in_w,
        gene_mask,
        translations_person,
        rotations_person,
        translations_object,
        rotations_object,
        masks_person,
        masks_object,
        target_person,
        target_object,
        verts_object,
        faces_object,
        faces_person,
        person_pose,
        person_shape,
        # masks_object,
        # masks_person,
        # K_rois,
        # target_masks,
        # labels_person,
        # labels_object,
        # interaction_map_parts,
        # class_name,
        object_textures,
        smpl_textures,
        textures,
        int_scale_init=1.0,
        int_person_zscale=7/20,
        device=torch.device("cuda"),
    ):
        super(smplPHOSA, self).__init__()
        self.device = device
        self.object_textures = object_textures
        self.smpl_textures = smpl_textures
        self.textures = textures
        translation_person = translations_person.detach().clone()
        self.translations_person = nn.Parameter(translation_person, requires_grad=True)
        rotations_person = rotations_person.detach().clone()
        self.rotations_person = nn.Parameter(rotations_person, requires_grad=False)
        translation_object = translations_object.detach().clone()
        self.person_pose = nn.Parameter(person_pose, requires_grad=False)
        person_pose = person_pose.detach().clone()
        self.person_shape = nn.Parameter(person_shape, requires_grad=True)
        person_shape = person_shape.detach().clone()
        self.translations_object = nn.Parameter(translation_object, requires_grad=False)
        rotations_object = rotations_object.detach().clone()
        self.rotations_object = nn.Parameter(rotations_object, requires_grad=False)

        self.register_buffer("gene_mask", gene_mask)

        self.register_buffer("masks_object", masks_object)
        self.register_buffer("masks_person", masks_person)
        self.register_buffer("target_object", target_object)
        self.register_buffer("target_person", target_person)

        self.register_buffer("verts_object", verts_object)
        # self.register_buffer("verts_person", verts_person)
        self.register_buffer("faces_object", faces_object)
        self.register_buffer("faces_person", faces_person)

        # print("initial", faces_person.unsqueeze(0).size())
        # self.register_buffer("cams_person", cams_person)

        """
        self.int_scales_object = nn.Parameter(
            int_scale_init * torch.ones(1).float(), requires_grad=False
        )
        self.int_scale_object_mean = nn.Parameter(
            torch.tensor(int_scale_init).float(), requires_grad=False
        )
        self.int_scales_person = nn.Parameter(
            int_scale_init * torch.ones(1).float(), requires_grad=False
        )
        self.int_scale_person_mean = nn.Parameter(
            torch.tensor(1.0).float().cuda(), requires_grad=False
        )
        self.person_zscale = nn.Parameter(
            torch.tensor(1.0).float().cuda() * int_person_zscale, requires_grad=False
        )
        """
        self.register_buffer("int_scales_object", int_scale_init * torch.ones(1).float())
        self.register_buffer("int_scale_object_mean", torch.tensor(int_scale_init).float())
        self.register_buffer("int_scales_person", int_scale_init * torch.ones(1).float())
        self.register_buffer("int_scale_person_mean", torch.tensor(1.0).float().cuda())
        # self.register_buffer("person_zscale", torch.tensor(1.0).float().cuda() * int_person_zscale)
        self.person_zscale = nn.Parameter(
            torch.tensor(1.0).float().cuda() * int_person_zscale, requires_grad=False
        )

        self.cuda()

        # Set up the differentiable render
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        cameras = FoVPerspectiveCameras(device=device)           
        raster_settings = RasterizationSettings(
            image_size = (in_w, in_w),
            blur_radius = 0,
            faces_per_pixel = 1,
        )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (0,0,0))
        self.renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )

        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size = (in_w, in_w),
            blur_radius = np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
        )
        # Silhouette renderer 
        renderer_silhouette = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        self.renderer_silhouette = renderer_silhouette

        self.losses = Losses(
            gene_mask = gene_mask,
            renderer = self.renderer,
            renderer_silhouette = self.renderer_silhouette,
            device = self.device,
            object_textures = self.object_textures,
            smpl_textures = self.smpl_textures,
            textures = self.textures,
            masks_person = self.masks_person,
            masks_object = self.masks_object,
            target_person = self.target_person,
            target_object = self.target_object,
            # ref_mask=self.ref_mask,
            # keep_mask=self.keep_mask,
            # K_rois=self.K_rois,
            # interaction_map_parts=interaction_map_parts,
            # labels_person=labels_person,
            # labels_object=labels_object,
            # class_name=class_name,
        )

        smpl = SMPL()
        self.smpl = smpl.to(self.device)  
    
    def get_verts_object(self):
        return compute_transformation_persp(
            meshes=self.verts_object,
            translations=self.translations_object,
            rotations=self.rotations_object,
            intrinsic_scales=self.int_scales_object,
        )

    def get_verts_person(self):
        device = self.device
        pred_vertices_smpl = self.smpl(self.person_pose, self.person_shape)
        print(pred_vertices_smpl)
        verts_person =  compute_transformation_persp(
            meshes=pred_vertices_smpl,
            translations=self.translations_person,
            rotations=self.rotations_person,
            intrinsic_scales=self.int_scales_person,
        )
        sz = self.person_zscale.cpu()
        scaleAZ = Scale(x=1.0, y=1.0, z=sz).to(device)
        verts_person = scaleAZ.transform_points(verts_person.to(device))
        return verts_person

    def compute_sil_loss(self, verts_object, verts_person):
        # generte smpl mask
        smpl_mesh = Meshes(verts=self.get_verts_person().to(self.device), 
                    faces=self.faces_person.to(self.device), 
                    textures=self.smpl_textures)
        smpl_pred, _ = self.renderer_silhouette(smpl_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]]))
        smpl_pred_silhouette = smpl_pred[..., 3]
        """
        # generte obj mask
        obj_mesh = Meshes(verts=self.get_verts_object().to(self.device), 
                    faces=self.faces_object.to(self.device), 
                    textures=self.object_textures)
        obj_pred, _ = self.renderer_silhouette(obj_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]]))
        obj_pred_silhouette = obj_pred[..., 3]

        return self.losses.compute_sil_loss(
                                smpl_pred_mask = smpl_pred_silhouette,
                                obj_pred_mask = obj_pred_silhouette
                            )
        """
        return self.losses.compute_sil_loss(
                                smpl_pred_mask = smpl_pred_silhouette
                            )


    def compute_ordinal_depth_loss(self, verts_object, verts_person):
        # generte smpl mask
        smpl_mesh = Meshes(verts=verts_person.to(self.device), 
                    faces=self.faces_person.to(self.device), 
                    textures=self.smpl_textures)
        smpl_pred, smpl_frag = self.renderer_silhouette(smpl_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]]))
        smpl_pred_silhouette = smpl_pred[..., 3]
        smpl_depth = smpl_frag.zbuf[0, :, :, 0]
        # print(smpl_depth.size())
        # generte obj mask
        obj_mesh = Meshes(verts=verts_object.to(self.device), 
                    faces=self.faces_object.to(self.device), 
                    textures=self.object_textures)
        obj_pred, obj_frag = self.renderer_silhouette(obj_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=self.device, location=[[0.0, 0.0, -3.0]]))
        obj_pred_silhouette = obj_pred[..., 3]
        obj_depth = obj_frag.zbuf[0, :, :, 0]
        # print(obj_depth.size())

        return self.losses.compute_ordinal_depth_loss(
                    smpl_pred_mask = smpl_pred_silhouette,
                    obj_pred_mask = obj_pred_silhouette,
                    smpl_depth = smpl_depth,
                    obj_depth = obj_depth
                )


    def forward(self, loss_weights=None):
        """
        If a loss weight is zero, that loss isn't computed (to avoid unnecessary
        compute).
        """
        loss = torch.tensor(0.0).float().cuda()

        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()

        if loss_weights["lw_ordinal_depth"] > 0:
            loss_ordinal_depth = self.compute_ordinal_depth_loss(
                                    verts_object = verts_object,
                                    verts_person = verts_person
                                ) * loss_weights["lw_ordinal_depth"]
            loss += loss_ordinal_depth
            
        if loss_weights["lw_sil"] > 0:
            loss_sil = self.compute_sil_loss(
                            verts_object = verts_object,
                            verts_person = verts_person
                        ) * loss_weights["lw_sil"]
            loss += loss_sil
        
        if loss_weights["lw_inter"] > 0:
            loss_inter = self.losses.compute_inter_loss(
                            verts_object = verts_object, 
                            verts_person = verts_person
                        ) * loss_weights["lw_inter"]
            loss += loss_inter

        if loss_weights["lw_centroid"] > 0:
            loss_centroid = self.losses.compute_centroid_loss(
                            verts_object = verts_object, 
                            verts_person = verts_person
                        ) * loss_weights["lw_centroid"]
            loss += loss_centroid

        return loss

    def get_parameters(self):
        """
        Computes a json-serializable dictionary of optimized parameters.
        Returns:
            parameters (dict): Dictionary mapping parameter name to list.
        """
        parameters = {
            "scales_object": self.int_scales_object,
            "scales_person": self.int_scales_person,
            "rotations_object": self.rotations_object,
            "rotations_person": self.rotations_person,
            "translations_object": self.translations_object,
            "translations_person": self.translations_person,
            "person_zscale": self.person_zscale,
            "person_pose": self.person_pose,
            "person_shape": self.person_shape,
        }
        for k, v in parameters.items():
            parameters[k] = v.detach().cpu().numpy().tolist()
        return parameters



##################################################################
# Help Functions Definition
##################################################################

def create_mesh_texture(mesh,mesh_color,device):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    # mesh_color = colors['light_purple']
    mesh_color = np.array(mesh_color)[::-1]
    mesh_color = torch.from_numpy(mesh_color.copy()).view(1, 1, 3).float().to(device)
    mesh_color = mesh_color.repeat(1, verts.shape[0], 1)
    return mesh_color


def get_smpl_mesh(obj_filename,move=True,dist=2.5,texture=True,scale=True,device=torch.device("cuda")):
    verts, faces_idx, _ = load_obj(obj_filename)
    faces = faces_idx.verts_idx
    verts = verts.float().unsqueeze(0)
    faces = faces.long().unsqueeze(0)

    if move:
        Ct = np.array(((1,0,0,0),
                        (0,1,0,0),
                        (0,0,1,-dist),
                        (0,0,0,1)))
        pred_vertices = verts[0].cpu().numpy()
        ones = np.ones(pred_vertices.shape[0]).reshape(-1,1)
        pred_vertices = np.hstack((pred_vertices,ones))
        pred_vertices = pred_vertices @ (Ct.T)
        pred_vertices = pred_vertices / (pred_vertices[:,3].reshape(-1,1))
        pred_vertices = pred_vertices[:,0:3]
        pred_vertices = torch.from_numpy(pred_vertices).float()
        verts = pred_vertices.float().unsqueeze(0)

    if scale:
        # actually this is not very suitable, because the human size estimated by
        # smpl model is about 1.5m high, and this scale will make it too short
        sz = 7 / 20
        scaleAZ = Scale(x=1.0, y=1.0, z=sz).to(device)
        verts = scaleAZ.transform_points(verts.to(device))


    mesh2 = Meshes(
    verts=verts.to(device),   
    faces=faces.to(device))

    [verts] = mesh2.verts_list()
    [faces] = mesh2.faces_list()

    if texture:
        nocolor = torch.zeros((100, 100), device=device)
        color_gradient = torch.linspace(0, 1, steps=100, device=device)
        color_gradient1 = color_gradient[None].expand_as(nocolor)
        color_gradient2 = color_gradient[:, None].expand_as(nocolor)
        colors1 = torch.stack([nocolor, color_gradient1, color_gradient2], dim=2)
        verts_uvs1 = torch.rand(size=(verts.shape[0], 2), device=device)

        textures1 = TexturesUV(
            maps=[colors1], faces_uvs=[faces], verts_uvs=[verts_uvs1]
        )

        mesh2 = Meshes(verts=[verts], faces=[faces], textures=textures1)
    else:
        mesh2 = Meshes(verts=[verts], faces=[faces])

    return mesh2


def save_out(mesh, renderer, outpath, device, use_bg=False, bg_img=None):
    img = renderer(mesh.to(device))
    img = img[0].cpu().numpy()
    img = img[:, :, :3]/np.amax(img[:, :, :3])
    img = img*255
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if use_bg:
        mask = np.sum(img, axis=2, keepdims=True) == 0
        img = img + mask*bg_img
    cv2.imwrite(outpath, img)
    return img


def process_resize(img_file, input_res=1280):
    """
    Read image, do preprocessing
    """
    rgb_img_in = img_file[:,:,::-1].copy().astype(np.uint8)
    resize_img = preprocess_generic(rgb_img_in, input_res, display=True)
    resize_img = resize_img.astype(np.uint8)

    return resize_img


def gen_video_out(outVideo_fileName, the_fps, img_array):

    print(f">> Generating video in {outVideo_fileName}")
    oneim = img_array[0]
    height, width, layers = oneim.shape
    size = (width,height)
    
    out = cv2.VideoWriter(outVideo_fileName,cv2.VideoWriter_fourcc(*'mp4v'), the_fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()