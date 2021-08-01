import os
import sys
import torch
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import cv2
import argparse
from glob import glob

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
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

import config

"""
python visualize_group.py --arti_input=example_output/frame_0089/arti_pred.obj --obj_smpl_dict=example_output/ --out_smpl_dict=example_output/render/ --arti_output=example_output/articulation.png --smpl_input=example_output/frame89_preds.obj
python visualize_group.py --img_index=4259
"""


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# command line arguments
parser = argparse.ArgumentParser(
    description="A script that combine results of articulation prediction and smpl prediction."
)
parser.add_argument("--img_index", default=None, help="input index of the folder, will override the others")
parser.add_argument("--arti_input", default="example_output/frame_0000/arti_pred.obj", help="input articulation result")
parser.add_argument("--arti_output", default="example_output/articulation.png", help="output articulation result")
parser.add_argument("--smpl_input", default="example_output/frame0_preds.obj", help="input smpl result")
parser.add_argument("--smpl_output", default="example_output/smpl.png", help="output smpl result")
parser.add_argument("--combine_output", default="example_output/combine.png", help="output combine result")
parser.add_argument("--obj_smpl_dict", default="example_output/", help="input smpl directory")
parser.add_argument("--out_smpl_dict", default="example_output/render/", help="output smpl directory")
args = parser.parse_args()

if args.img_index != None:
    args.arti_input = "example" + str(args.img_index) + "_output/frame_0000/arti_pred.obj"
    args.arti_output = "example" + str(args.img_index) + "_output/articulation.png"
    args.smpl_input = "example" + str(args.img_index) + "_output/frame0_preds.obj"
    args.smpl_output = "example" + str(args.img_index) +"_output/smpl.png"
    args.combine_output = "example" + str(args.img_index) +"_output/combine.png"
    args.obj_smpl_dict = "example" + str(args.img_index) + "_output/"
    args.out_smpl_dict = "example" + str(args.img_index) + "_output/render/"
    


# Set paths
obj_filename1 = args.arti_input
mesh_name1 = args.arti_output

obj_smpl_dict = args.obj_smpl_dict
out_smpl_dict = args.out_smpl_dict

colors = {
    # colorbline/print/copy safe:
    'light_gray':  [0.9, 0.9, 0.9],
    'light_purple':  [0.8, 0.53, 0.53],
    'light_green': [166/255.0, 178/255.0, 30/255.0],
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
}


# Load obj file of the articulation
##################################################################
# Shengyi: right now mesh1 is under habitat corrdinate system
# we want to convert it to pytorch3d corrdinate system
#
# Transforms the transforms from habitat world to pytorch3d world.
# Habitat world: +X right, +Y up, +Z from screen to us.
# Pytorch3d world: +X left, +Y up, +Z from us to screen.
# Compose the rotation by adding 180 rotation about the Y axis.

mesh1 = load_objs_as_meshes([obj_filename1], device=device)
conversion = RotateAxisAngle(axis="y", angle=180).to(device)
mesh1.verts_list()[0] = conversion.transform_points(mesh1.verts_list()[0])

# Scale back to original aspect ratio
scaleAR = Scale(x=1.0, y=0.75, z=1.0).to(device)
mesh1.verts_list()[0] = scaleAR.transform_points(mesh1.verts_list()[0])

##################################################################
# Shengyi: R should be identity, T should be 0
# so that we're using the same camera

lights1 = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
# cameras1 = FoVPerspectiveCameras(device=device)
cameras1 = PerspectiveCameras(device=device, focal_length=config.FOCAL_LENGTH,
            image_size=((960, 1280),), 
            principal_point=((480, 640),))
####################################################################

lights2 = PointLights(
            ambient_color = [[1.0, 1.0, 1.0],],
            diffuse_color = [[1.0, 1.0, 1.0],],
            device=device, location=[[1.0, 1.0, -30]])
cameras2 = PerspectiveCameras(device=device, 
            focal_length=config.FOCAL_LENGTH,
            image_size=((448, 448),), 
            principal_point=((224, 224),))

cameras = PerspectiveCameras(device=device, 
            focal_length=config.FOCAL_LENGTH,
            image_size=((1280, 1280),), 
            principal_point=((640, 640),))


raster_settings1 = RasterizationSettings(
    image_size = (480, 480),
    blur_radius = 0,
    faces_per_pixel = 1,
)

            
raster_settings2 = RasterizationSettings(
    image_size = (480, 480),
    blur_radius = 0,
    faces_per_pixel = 1,
)
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (0,0,0))


renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings1
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights1,
        blend_params=blend_params
    )
)

renderer1 = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras1, 
        raster_settings=raster_settings1
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras1,
        lights=lights1,
        blend_params=blend_params
    )
)


renderer2 = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras2, 
        raster_settings=raster_settings2
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras2,
        lights=lights2,
        blend_params=blend_params
    )
)

images1 = renderer(mesh1.to(device))
images1 = images1[0].cpu().numpy()
images1 = images1[:, :, :3]/np.amax(images1[:, :, :3])
# images1 = cv2.cvtColor(images1, cv2.COLOR_RGB2BGR)
cv2.imwrite(mesh_name1, images1*255)

####################################################################

img_list = glob(obj_smpl_dict+"*.obj")

for i in range(len(img_list)-1):
    if not os.path.isdir(out_smpl_dict):
        os.mkdir(out_smpl_dict)
    obj_filename2 = obj_smpl_dict + "frame" + str(i+1) + "_preds.obj"
    mesh_name2 = out_smpl_dict + "frame" + str(i+1) + "_render.png"

    # Load obj file of the smpl
    verts, faces_idx, _ = load_obj(obj_filename2)
    faces = faces_idx.verts_idx
    verts = verts.float().unsqueeze(0)
    faces = faces.long().unsqueeze(0)

    # Scale the z value of the image according to resolution
    sz = 7 / 20
    scaleAZ = Scale(x=1.0, y=1.0, z=sz).to(device)
    verts = scaleAZ.transform_points(verts.to(device))

    mesh2 = Meshes(
        verts=verts.to(device),   
        faces=faces.to(device)
    )

    [verts] = mesh2.verts_list()
    [faces] = mesh2.faces_list()
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


    # create the combined mesh
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
    mesh_color2 = colors['light_purple']
    mesh_color2 = np.array(mesh_color2)[::-1]
    mesh_color2 = torch.from_numpy(mesh_color2.copy()).view(1, 1, 3).float().to(device)
    mesh_color2 = mesh_color2.repeat(1, verts2.shape[0], 1)

    mesh_color = torch.cat((mesh_color1, mesh_color2), dim=1)

    textures = Textures(verts_rgb = mesh_color)
    mesh = Meshes(verts=verts.unsqueeze(0), 
            faces=faces.unsqueeze(0),
            textures=textures
            )

    images = renderer(mesh.to(device))
    images = images[0].cpu().numpy()
    images = images[:, :, :3]/np.amax(images[:, :, :3])
    # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mesh_name2, images*255)
    print("Image saved as", mesh_name2)

obj_filename2 = args.smpl_input
# Load obj file of the smpl
verts, faces_idx, _ = load_obj(obj_filename2)
faces = faces_idx.verts_idx
verts = verts.float().unsqueeze(0)
faces = faces.long().unsqueeze(0)

sz = 7 / 20
scaleAZ = Scale(x=1.0, y=1.0, z=sz).to(device)
verts = scaleAZ.transform_points(verts.to(device))

mesh2 = Meshes(
    verts=verts.to(device),   
    faces=faces.to(device)
)
[verts] = mesh2.verts_list()
[faces] = mesh2.faces_list()
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

images2 = renderer(mesh2.to(device))
images2 = images2[0].cpu().numpy()
images2 = images2[:, :, :3]/np.amax(images2[:, :, :3])
# images2 = cv2.cvtColor(images2, cv2.COLOR_RGB2BGR)
cv2.imwrite(args.smpl_output, images2*255)


#############################################################
# Used only when aspect ratio & scale ratio not adjusted
#############################################################
"""
images2 = cv2.resize(images2, (853,853))
delta_h, delta_w = 853-480, 853-480
top, bottom = delta_h//2, delta_h-(delta_h//2)
left, right = delta_w//2, delta_w-(delta_w//2)
color = [0, 0, 0]
images1 = cv2.copyMakeBorder(images1, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)

alpha = images1.copy()
beta = images2.copy()
alpha[alpha>0] = 0.5
alpha[beta==0] = 1.0
new_img = alpha * images1+ (1.0 - alpha) * images2
cv2.imwrite(args.combine_output, new_img*255)
"""
#############################################################

