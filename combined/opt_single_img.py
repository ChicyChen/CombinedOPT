# After obtaining 2D mask of the plane, 3D mesh of the plane, 3D mesh of the person,
# use PointRend2 to get 2d mask of the person,
# and then optimize rotation & translation (& scale) parameters of the plane & the person 
# using the render of pytorch3d.

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

    verts_rot = torch.matmul(meshes.detach().clone(), rotations)
    verts_trans = verts_rot + translations

    verts_final = intrinsic_scales.view(-1, 1, 1) * verts_trans

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

        # print(gt_person.shape)
        # print(gt_object.shape)

        front_person_gt = gt_person
        front_person_gt[gt_object > 0] = 0
        front_object_gt = gt_object
        front_object_gt[gt_person > 0] = 0

        # print(front_person_gt.shape)
        # print(front_object_gt.shape)

        front_person_pred = smpl_pred_mask[0]
        front_object_pred = obj_pred_mask[0]

        # print(front_person_pred.shape)
        # print(front_object_pred.shape)

        both_pred = front_person_pred * front_object_pred

        false_person = front_person_gt * front_object_pred * both_pred
        false_object = front_object_gt * front_person_pred * both_pred

        false_person_dists = torch.clamp(smpl_depth - obj_depth, min=0.0, max=2.0) * false_person.float()
        false_object_dists = torch.clamp(obj_depth - smpl_depth, min=0.0, max=2.0) * false_object.float()

        l_p = torch.sum(torch.log(1 + torch.exp(false_person_dists)))
        l_o = torch.sum(torch.log(1 + torch.exp(false_object_dists)))
        loss_ordinal_depth = loss_ordinal_depth + l_p + l_o
        loss_ordinal_depth /= 2
        loss_ordinal_depth /= 1e6

        # print(l_p, l_o, loss_ordinal_depth)
        
        return {"loss_ordinal_depth": loss_ordinal_depth}

    def compute_sil_loss(self, smpl_pred_mask, obj_pred_mask):
        device = self.device
        loss_sil = torch.tensor(0.0).float().cuda()

        person_image = self.target_person * smpl_pred_mask
        object_image = self.target_object * obj_pred_mask

        l_p = torch.sum((person_image - self.masks_person) ** 2) / self.target_person.sum()
        l_o = torch.sum((object_image - self.masks_object) ** 2) / self.target_object.sum()

        loss_sil = loss_sil + l_p +l_o

        return {"loss_sil": loss_sil}
        

    def compute_inter_loss(self, verts_object, verts_person):
        device = self.device
        loss_inter = torch.tensor(0.0).float().cuda()

        loss_inter, loss_normals = chamfer_distance(verts_object.to(device),
                                                    verts_person.to(device))
        # print(loss_inter)

        return {"loss_inter": loss_inter}

    def compute_centroid_loss(self, verts_object, verts_person):
        device = self.device
        loss_centroid = torch.tensor(0.0).float().cuda()

        loss_centroid = self.mse(verts_person[0].mean(0), verts_object[0].mean(0))
        # print(loss_centroid)

        return {"loss_centroid": loss_centroid}

class simplePHOSA(nn.Module):
    def __init__(
        self,
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
        self.rotations_person = nn.Parameter(rotations_person, requires_grad=True)
        translation_object = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_object, requires_grad=True)
        rotations_object = rotations_object.detach().clone()
        self.rotations_object = nn.Parameter(rotations_object, requires_grad=True)

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
            int_scale_init * torch.ones(1).float(), requires_grad=True,
        )
        self.int_scale_object_mean = nn.Parameter(
            torch.tensor(int_scale_init).float(), requires_grad=False
        )
        self.int_scales_person = nn.Parameter(
            int_scale_init * torch.ones(1).float(), requires_grad=True
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
                            lights=PointLights(device=device, location=[[0.0, 0.0, -3.0]]))
        smpl_pred_silhouette = smpl_pred[..., 3]
        # generte obj mask
        obj_mesh = Meshes(verts=self.get_verts_object().to(self.device), 
                    faces=self.faces_object.to(self.device), 
                    textures=self.object_textures)
        obj_pred, _ = self.renderer_silhouette(obj_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=device, location=[[0.0, 0.0, -3.0]]))
        obj_pred_silhouette = obj_pred[..., 3]

        return self.losses.compute_sil_loss(
                                smpl_pred_mask = smpl_pred_silhouette,
                                obj_pred_mask = obj_pred_silhouette
                            )


    def compute_ordinal_depth_loss(self, verts_object, verts_person):
        # generte smpl mask
        smpl_mesh = Meshes(verts=verts_person.to(self.device), 
                    faces=self.faces_person.to(self.device), 
                    textures=self.smpl_textures)
        smpl_pred, smpl_frag = self.renderer_silhouette(smpl_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=device, location=[[0.0, 0.0, -3.0]]))
        smpl_pred_silhouette = smpl_pred[..., 3]
        smpl_depth = smpl_frag.zbuf[0, :, :, 0]
        # print(smpl_depth.size())
        # generte obj mask
        obj_mesh = Meshes(verts=verts_object.to(self.device), 
                    faces=self.faces_object.to(self.device), 
                    textures=self.object_textures)
        obj_pred, obj_frag = self.renderer_silhouette(obj_mesh, 
                            cameras=FoVPerspectiveCameras(device=self.device), 
                            lights=PointLights(device=device, location=[[0.0, 0.0, -3.0]]))
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
        loss_dict = {}
        verts_object = self.get_verts_object()
        verts_person = self.get_verts_person()

        if loss_weights is None or loss_weights["lw_ordinal_depth"] > 0:
            loss_dict.update(
                self.compute_ordinal_depth_loss(
                    verts_object = verts_object,
                    verts_person = verts_person
                )
            )


        if loss_weights is None or loss_weights["lw_sil"] > 0:
            loss_dict.update(
                self.compute_sil_loss(
                    verts_object = verts_object,
                    verts_person = verts_person
                )
            )
            

        if loss_weights is None or loss_weights["lw_inter"] > 0:
            loss_dict.update(
                self.losses.compute_inter_loss(
                    verts_object = verts_object, 
                    verts_person = verts_person
                )
            )


        if loss_weights is None or loss_weights["lw_centroid"] > 0:
            loss_dict.update(
                self.losses.compute_centroid_loss(
                    verts_object = verts_object, 
                    verts_person = verts_person
                )
            )

        return loss_dict

    
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


def get_smpl_mesh(obj_filename,move=True,dist=2.5,texture=True,scale=True):
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


def save_out(mesh, renderer, outpath):
    img = renderer(mesh.to(device))
    img = img[0].cpu().numpy()
    img = img[:, :, :3]/np.amax(img[:, :, :3])
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outpath, img*255)


##################################################################
# Global Values Definition
##################################################################

# Set up
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Set paths
obj_filename1 = "example_opt329_output/frame_0060/arti_pred.obj"
obj_filename2 = "example_opt329_output/frame60_preds.obj"
mask_plane_path = "example_opt329_output/frame60planemask_preds.txt"
only_art = "visualize_opt329_obj_art.png"
only_smpl = "visualize_opt329_obj_smpl.png"
combine_initial = "visualize_opt329_obj_combine0.png"
combine_opt = "visualize_opt329_obj_optimize0.png"
target_image_name = "/home/siyich/Articulation3D/siyich/cmr_art/random329_output/origin_frame_60.png"

# Read input image
im = cv2.imread(target_image_name)
in_h = len(im)
in_w = len(im[0])

colors = {
    # colorbline/print/copy safe:
    'light_gray':  [0.9, 0.9, 0.9],
    'light_purple':  [0.8, 0.53, 0.53],
    'light_green': [166/255.0, 178/255.0, 30/255.0],
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_red': [240/255, 128/255, 128/255]
}


##################################################################
# Predict 2d mask of the person from RGB input
##################################################################

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

pads = int((in_w - in_h)/2)
m = nn.ZeroPad2d((0, 0, pads, in_w - in_h - pads))

gene_mask = torch.ones([in_h, in_w], dtype=torch.float64, device=device)
gene_mask = m(gene_mask)

is_person = instances.pred_classes == 0
bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()
masks_person = instances[is_person].pred_masks
masks_person = masks_person[0]
masks_person = m(masks_person)

draw_masks_person = masks_person.float().cpu().numpy()
draw_masks_person = draw_masks_person.reshape((in_w,in_w,1))
cv2.imwrite("draw_masks_person.png", draw_masks_person*255)


##################################################################
# Load 2d mask of the plane from input
##################################################################
masks_object = np.loadtxt(mask_plane_path)
masks_object = torch.tensor(masks_object).to(device)
masks_object = m(masks_object)
masks_object = masks_object > 0

# draw_masks_object = masks_object.float().cpu().numpy()
# draw_masks_object = draw_masks_object.reshape((in_w,in_w,1))
# cv2.imwrite("draw_masks_object.png", draw_masks_object*255)

masks_combine = masks_person | masks_object
masks_combine = ~ masks_combine

target_person = masks_combine.clone()
target_object = masks_combine.clone()
target_person[masks_person] = masks_person[masks_person]
target_object[masks_object] = masks_object[masks_object]
target_person = target_person.float()
target_object = target_object.float()


##################################################################
# Load mesh of the plane from input obj file
##################################################################

mesh1 = load_objs_as_meshes([obj_filename1], device=device)
conversion = RotateAxisAngle(axis="y", angle=180).to(device)
mesh1.verts_list()[0] = conversion.transform_points(mesh1.verts_list()[0])

# Scale back to original aspect ratio
scaleAR = Scale(x=1.0, y=0.75, z=1.0).to(device)
mesh1.verts_list()[0] = scaleAR.transform_points(mesh1.verts_list()[0])


##################################################################
# Load mesh of the smpl from input obj file
##################################################################

mesh2 = get_smpl_mesh(obj_filename2,move=False,scale=True)


##################################################################
# create the combined mesh
##################################################################

mesh = join_meshes_as_batch([mesh1, mesh2])

verts = mesh.verts_packed()
faces = mesh.faces_packed()

verts1 = mesh1.verts_packed()
faces1 = mesh1.faces_packed()
mesh_color1 = colors['light_gray']
mesh_color1 = np.array(mesh_color1)[::-1]
mesh_color1 = torch.from_numpy(mesh_color1.copy()).view(1, 1, 3).float().to(device)
mesh_color1 = mesh_color1.repeat(1, verts1.shape[0], 1)

verts2 = mesh2.verts_packed()
faces2 = mesh2.faces_packed()
mesh_color2 = create_mesh_texture(mesh2,colors['light_purple'],device)
mesh_color = torch.cat((mesh_color1, mesh_color2), dim=1)

object_textures = Textures(verts_rgb = mesh_color1)
smpl_textures = Textures(verts_rgb = mesh_color2)
textures = Textures(verts_rgb = mesh_color)

mesh = Meshes(verts=verts.unsqueeze(0), 
        faces=faces.unsqueeze(0),
        textures=textures
        )


##################################################################
# Optimize the combinition of the two
##################################################################

loss_weights = {
                "lw_ordinal_depth": torch.tensor(0.0).float().to(device),
                "lw_sil": torch.tensor(0.0).float().to(device),
                "lw_inter": torch.tensor(1.0).float().to(device),
                "lw_centroid": torch.tensor(1.0).float().to(device),
                }
num_iterations = 10
lr = 1e-4

model = simplePHOSA(
    gene_mask = gene_mask,
    translations_person = torch.FloatTensor([[0, 0, 0]]),
    rotations_person = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    translations_object = torch.FloatTensor([[0, 0, 0]]),
    rotations_object = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    masks_person = masks_person.float(), # float
    masks_object = masks_object.float(), # float
    target_person = target_person,
    target_object = target_object,
    verts_object = mesh1.verts_packed().unsqueeze(0),
    faces_object = mesh1.faces_packed().unsqueeze(0),
    verts_person = mesh2.verts_packed().unsqueeze(0),
    faces_person = mesh2.faces_packed().unsqueeze(0),
    object_textures = object_textures,
    smpl_textures = smpl_textures,
    textures = textures,
    int_scale_init=1.0,
    device = device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loop = tqdm(range(num_iterations))
for _ in loop:
    optimizer.zero_grad()
    loss_dict = model(loss_weights=loss_weights)
    # loss_dict_weighted = {
    #     k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict
    # }
    loss_dict_weighted = loss_dict
    loss = sum(loss_dict_weighted.values())
    print("loss:",loss)
    loop.set_description(f"Loss {loss.data:.4f}")
    loss.backward()
    optimizer.step()


##################################################################
# Output the optimized parameters
##################################################################

parameters = model.get_parameters()
print("scales_object", parameters["scales_object"])
print("scales_person", parameters["scales_person"])
print("rotations_object", parameters["rotations_object"])
print("rotations_person", parameters["rotations_person"])
print("translations_object", parameters["translations_object"])
print("translations_person", parameters["translations_person"])


##################################################################
# Get the new rendered mesh
##################################################################

new_verts1 = model.get_verts_object()
new_verts2 = model.get_verts_person()

new_mesh1 = Meshes(
    verts=new_verts1.to(device),   
    faces=faces1.unsqueeze(0))
new_mesh2 = Meshes(
    verts=new_verts2.to(device),   
    faces=faces2.unsqueeze(0))

new_mesh = join_meshes_as_batch([new_mesh1,new_mesh2])
new_verts = new_mesh.verts_packed()
new_faces = new_mesh.faces_packed()

new_mesh = Meshes(verts=new_verts.unsqueeze(0), 
        faces=new_faces.unsqueeze(0),
        textures=textures
        )


##################################################################
# Render and save results
##################################################################
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
cameras = FoVPerspectiveCameras(device=device)
            
raster_settings = RasterizationSettings(
    image_size = (in_w, in_w),
    blur_radius = 0,
    faces_per_pixel = 1,
)
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (0,0,0))

renderer = MeshRenderer(
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

save_out(mesh1, renderer, only_art)
save_out(mesh2, renderer, only_smpl)
save_out(mesh, renderer, combine_initial)

with torch.no_grad():
    save_out(new_mesh, renderer, combine_opt)
