import os
import sys
import torch
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import cv2
import config

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

"""
python visualize_corrected.py
"""


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Set paths
obj_filename1 = "example_opt329_output/frame_0060/arti_pred.obj"
obj_filename2 = "example_opt329_output/frame60_preds.obj"
# obj_filename2 = "example1456_output/frame10_preds.obj"
# obj_filename3 = "example1456_output/frame20_preds.obj"
# obj_filename4 = "example1456_output/frame30_preds.obj"
# obj_filename5 = "example1456_output/frame40_preds.obj"
# obj_filename6 = "example1456_output/frame50_preds.obj"
only_art = "visualize329_obj_art.png"
only_smpl = "visualize329_obj_smpl.png"
combine_initial = "visualize329_obj_combine0.png"
combine_angle1 = "visualize329_obj_combine1.png"
combine_angle2 = "visualize329_obj_combine2.png"


colors = {
    # colorbline/print/copy safe:
    'light_gray':  [0.9, 0.9, 0.9],
    'light_purple':  [0.8, 0.53, 0.53],
    'light_green': [166/255.0, 178/255.0, 30/255.0],
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_red': [240/255, 128/255, 128/255]
}


# Load obj file of the articulation
##################################################################
# Shengyi: right now mesh1 is under habitat coordinate system
# we want to convert it to pytorch3d coordinate system
#
# Transforms the transforms from habitat world to pytorch3d world.
# Habitat world: +X right, +Y up, +Z from screen to us.
# Pytorch3d world: +X left, +Y up, +Z from us to screen.
# Compose the rotation by adding 180 rotation about the Y axis.

mesh1 = load_objs_as_meshes([obj_filename1], device=device)
conversion = RotateAxisAngle(axis="y", angle=180).to(device)
mesh1.verts_list()[0] = conversion.transform_points(mesh1.verts_list()[0])
# mesh1.verts_list() is a list containing B tensors, each tensor
# is of size (Vi,3)

# Scale back to original aspect ratio
# scaleAR = Scale(x=1.0, y=0.75, z=1.0).to(device)
# mesh1.verts_list()[0] = scaleAR.transform_points(mesh1.verts_list()[0])


####################################################################

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

    """
    if scale:
        # actually this is not very suitable, because the human size estimated by
        # smpl model is about 1.5m high, and this scale will make it too short
        sz = 7 / 20
        scaleAZ = Scale(x=1.0, y=1.0, z=sz).to(device)
        verts = scaleAZ.transform_points(verts.to(device))
    """


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


# Load obj file of the smpl
smpl_mesh_list = []
# name_list = [obj_filename2, obj_filename3, obj_filename4, obj_filename5, obj_filename6]
name_list = [obj_filename2]
for obj_filename in name_list:
    smpl_mesh = get_smpl_mesh(obj_filename,move=False,scale=False)
    smpl_mesh_move = get_smpl_mesh(obj_filename,dist=1,move=True,scale=False)
    smpl_mesh_scale = get_smpl_mesh(obj_filename,move=False,scale=True)
    smpl_mesh_list.append(smpl_mesh)
    smpl_mesh_list.append(smpl_mesh_move)
    smpl_mesh_list.append(smpl_mesh_scale)


# create the combined mesh
mesh2 = join_meshes_as_batch(smpl_mesh_list)
mesh = join_meshes_as_batch([mesh1, smpl_mesh_scale])

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
# mesh_color2 = colors['light_purple']
# mesh_color2 = np.array(mesh_color2)[::-1]
# mesh_color2 = torch.from_numpy(mesh_color2.copy()).view(1, 1, 3).float().to(device)
# mesh_color2 = mesh_color2.repeat(1, verts2.shape[0], 1)

mesh_color2 = create_mesh_texture(smpl_mesh,colors['light_purple'],device)
mesh_color3 = create_mesh_texture(smpl_mesh_move,colors['light_green'],device)
mesh_color4 = create_mesh_texture(smpl_mesh_scale,colors['light_red'],device)

mesh_color = torch.cat((mesh_color1, mesh_color4), dim=1)
mesh_color_smpl = torch.cat((mesh_color2, mesh_color3, mesh_color4), dim=1)

textures = Textures(verts_rgb = mesh_color)
mesh = Meshes(verts=verts.unsqueeze(0), 
        faces=faces.unsqueeze(0),
        textures=textures
        )

textures_smpl = Textures(verts_rgb = mesh_color_smpl)
mesh2 = Meshes(verts=verts2.unsqueeze(0), 
        faces=faces2.unsqueeze(0),
        textures=textures_smpl
        )


##################################################################
# Shengyi: R should be identity, T should be 0
# so that we're using the same camera

lights1 = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

cameras0 = FoVPerspectiveCameras(device=device)
# cameras0 = PerspectiveCameras(device=device, focal_length=config.FOCAL_LENGTH,
#             image_size=((1280, 1280),), 
#             principal_point=((640, 640),))


R, T = look_at_view_transform(eye=np.array((-0.5,0,-1)).reshape(1,3))
cameras1 = FoVPerspectiveCameras(device=device, R=R, T=T)
R, T = look_at_view_transform(eye=np.array((0,0.5,-1)).reshape(1,3))
cameras2 = FoVPerspectiveCameras(device=device, R=R, T=T)

####################################################################

            
raster_settings = RasterizationSettings(
    image_size = (1280, 1280),
    blur_radius = 0,
    faces_per_pixel = 1,
)
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (0,0,0))

renderer0 = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras0, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras0,
        lights=lights1,
        blend_params=blend_params
    )
)

renderer1 = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras1, 
        raster_settings=raster_settings
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
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras2,
        lights=lights1,
        blend_params=blend_params
    )
)


save_out(mesh1, renderer0, only_art)
save_out(mesh2, renderer0, only_smpl)
save_out(mesh, renderer0, combine_initial)
save_out(mesh, renderer1, combine_angle1)
save_out(mesh, renderer2, combine_angle2)
