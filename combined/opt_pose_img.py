# After obtaining 2D mask of the plane, 3D mesh of the plane, 2d mask of the person,
# and pose & camera translation
# and then optimize rotation & translation (& scale) parameters of the plane & the person 
# using the render of pytorch3d.

import argparse
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
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj

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


from opt_utils import *
from models import SMPL


# Note: need several files existed in advance.
"""
python opt_pose_img.py --input /data/siyich/cmr_art/mask_mesh_758_output --frame 60 --output /data/siyich/cmr_art/opt_758_output
"""

parser = argparse.ArgumentParser(
    description="A script that generates results of articulation prediction."
)
parser.add_argument("--input", default="/data/siyich/cmr_art/mask_mesh_3336_output", help="input directory")
parser.add_argument("--output", default="/data/siyich/cmr_art/opt_3336_output", help="output directory")
parser.add_argument("--frame", default=60, type=int, help="target frame")
args = parser.parse_args()



##################################################################
# Global Values Definition
##################################################################

# Set up
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Set paths
input_dir = os.path.join(args.input, 'frame_{:0>4}'.format(args.frame))
obj_object = os.path.join(input_dir, "arti_pred.obj")
obj_person = os.path.join(input_dir, "smpl_pred.obj")
pose_person = os.path.join(input_dir, "person_pose.txt")
shape_person = os.path.join(input_dir, "person_shape.txt")
trans_person = os.path.join(input_dir, "person_trans.txt")
mask_plane_path = os.path.join(input_dir, "plane_mask.txt")
mask_person_path = os.path.join(input_dir, "person_mask.txt")
target_image_name = os.path.join(input_dir, "origin_frame.png")

if not os.path.isdir(args.output):
    os.mkdir(args.output)
output_dir = os.path.join(args.output, 'frame_{:0>4}'.format(args.frame))
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
only_arti = os.path.join(output_dir, "arti_opt.obj")
only_smpl = os.path.join(output_dir, "smpl_opt.obj")
combine_initial = os.path.join(output_dir, "combine_initial.png")
combine_opt = os.path.join(output_dir, "combine_optimize.png")


# Read input image
im = cv2.imread(target_image_name)
in_h = len(im)
in_w = len(im[0])
square_im = process_resize(img_file=im, input_res=in_w)
square_im = cv2.cvtColor(square_im, cv2.COLOR_RGB2BGR)

colors = {
    # colorbline/print/copy safe:
    'light_gray':  [0.9, 0.9, 0.9],
    'light_purple':  [0.8, 0.53, 0.53],
    'light_green': [166/255.0, 178/255.0, 30/255.0],
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_red': [240/255, 128/255, 128/255]
}


##################################################################
# Load 2d masks from input
##################################################################

pads = int((in_w - in_h)/2)
m = nn.ZeroPad2d((0, 0, pads, in_w - in_h - pads))

gene_mask = torch.ones([in_h, in_w], dtype=torch.float64, device=device)
gene_mask = m(gene_mask)

masks_person = np.loadtxt(mask_person_path)
masks_person = torch.tensor(masks_person).to(device)
masks_person = m(masks_person)
masks_person = masks_person > 0

# draw_masks_person = masks_person.float().cpu().numpy()
# draw_masks_person = draw_masks_person.reshape((in_w,in_w,1))
# cv2.imwrite("draw_masks_person.png", draw_masks_person*255)

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

mesh1 = load_objs_as_meshes([obj_object], device=device)
conversion = RotateAxisAngle(axis="y", angle=180).to(device)
mesh1.verts_list()[0] = conversion.transform_points(mesh1.verts_list()[0])

# Scale back to original aspect ratio
scaleAR = Scale(x=1.0, y=0.75, z=1.0).to(device)
mesh1.verts_list()[0] = scaleAR.transform_points(mesh1.verts_list()[0])


##################################################################
# Load mesh of the smpl from input obj file
##################################################################

mesh2 = get_smpl_mesh(obj_person,move=False,scale=True,device=device)

##################################################################
# Load parameters of the smpl from input
##################################################################

file = open(pose_person, "rb")
person_pose = np.load(file)
file.close

file = open(shape_person, "rb")
person_shape = np.load(file)
file.close

person_trans = np.loadtxt(trans_person)

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
                "lw_sil": torch.tensor(1.0).float().to(device),
                "lw_inter": torch.tensor(0.0).float().to(device),
                "lw_centroid": torch.tensor(0.0).float().to(device),
                }
num_iterations = 5
lr = 1

model = smplPHOSA(
    gene_mask = gene_mask,
    in_w = in_w,
    translations_person = torch.FloatTensor([[person_trans[0], person_trans[1], person_trans[2]]]),
    rotations_person = torch.FloatTensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    translations_object = torch.FloatTensor([[0, 0, 0]]),
    rotations_object = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    masks_person = masks_person.float(), # float
    masks_object = masks_object.float(), # float
    target_person = target_person,
    target_object = target_object,
    verts_object = mesh1.verts_packed().unsqueeze(0),
    faces_object = mesh1.faces_packed().unsqueeze(0),
    # verts_person = mesh2.verts_packed().unsqueeze(0),
    faces_person = mesh2.faces_packed().unsqueeze(0),
    person_pose = torch.tensor(person_pose),
    person_shape =  torch.tensor(person_shape),
    object_textures = object_textures,
    smpl_textures = smpl_textures,
    textures = textures,
    int_scale_init=1.0,
    device = device)

# model.get_verts_object()
# model.get_verts_person()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loop = tqdm(range(num_iterations))
for _ in loop:
    optimizer.zero_grad()
    loss = model(loss_weights=loss_weights)
    print("loss:",loss)
    loop.set_description(f"Loss {loss.data:.4f}")
    loss.backward()
    optimizer.step()

    parameters = model.get_parameters()
    # print("scales_object", parameters["scales_object"])
    # print("scales_person", parameters["scales_person"])
    # print("rotations_object", parameters["rotations_object"])
    # print("rotations_person", parameters["rotations_person"])
    # print("translations_object", parameters["translations_object"])
    print("translations_person", parameters["translations_person"])
    # print("person_zscale", parameters["person_zscale"])
    # print("person_pose", parameters["person_pose"])
    print("person_shape", parameters["person_shape"])


##################################################################
# Output the optimized parameters
##################################################################

parameters = model.get_parameters()
# print("scales_object", parameters["scales_object"])
# print("scales_person", parameters["scales_person"])
# print("rotations_object", parameters["rotations_object"])
# print("rotations_person", parameters["rotations_person"])
# print("translations_object", parameters["translations_object"])
print("translations_person", parameters["translations_person"])
# print("person_zscale", parameters["person_zscale"])
# print("person_pose", parameters["person_pose"])
print("person_shape", parameters["person_shape"])


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

# save_out(mesh1, renderer, only_arti, device)
# save_out(mesh2, renderer, only_smpl, device)
save_obj(only_arti, verts=new_verts1[0].to(device), faces=faces1)
save_obj(only_smpl, verts=new_verts2[0].to(device), faces=faces2)
save_out(mesh, renderer, combine_initial, device, True, square_im)
with torch.no_grad():
    save_out(new_mesh, renderer, combine_opt, device, True, square_im)
