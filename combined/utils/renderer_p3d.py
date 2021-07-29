# Copyright (c) Facebook, Inc. and its affiliates.

# Part of code is modified from https://github.com/facebookresearch/pytorch3d

import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from pytorch3d.io import save_obj
from planercnn.utils.mesh_utils import save_obj as p_save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    BlendParams,
    MeshRasterizer,  
    SoftPhongShader,
)

from iopath.common.file_io import PathManager
from pytorch3d.io.utils import _open_file

class Pytorch3dRenderer(object):

    def __init__(self, img_size, mesh_color, focal_length, camera_t, device, camera_r=np.array([[1,0,0],[0,1,0],[0,0,1]]).reshape(1,3,3)):
        # self.device = torch.device("cuda:0")
        self.device = device

        self.img_size = img_size

        # mesh color
        mesh_color = np.array(mesh_color)[::-1]
        self.mesh_color = torch.from_numpy(
            mesh_color.copy()).view(1, 1, 3).float().to(self.device)

        # renderer for large objects, such as whole body.
        #self.render_size = img_size
        self.render_size = img_size
        lights = PointLights(
            ambient_color = [[1.0, 1.0, 1.0],],
            diffuse_color = [[1.0, 1.0, 1.0],],
            device=self.device, location=[[1.0, 1.0, -30]])
        self.renderer = self.__get_renderer(self.render_size, lights, focal_length, camera_t, camera_r)



    def __get_renderer(self, render_size, lights, focal_length, camera_t, camera_r):

        # cameras = FoVOrthographicCameras(
        #     device = self.device,
        #     znear=0.1,
        #     zfar=10.0,
        #     max_y=1.0,
        #     min_y=-1.0,
        #     max_x=1.0,
        #     min_x=-1.0,
        #     scale_xyz=((1.0, 1.0, 1.0),),  # (1, 3)
        # )
        #cameras = FoVOrthographicCameras(device=self.device, T=camera_t.reshape(1,3), R=np.array([[-1,0,0],[0,-1,0],[0,0,1]]).reshape(1,3,3))
        # cameras = PerspectiveCameras(device=self.device, focal_length=5000.0, image_size=((render_size, render_size),), principal_point=((render_size*0.5, render_size*0.5),),T=camera_t.reshape(1,3), R=np.array([[-1,0,0],[0,-1,0],[0,0,1]]).reshape(1,3,3))
        cameras = PerspectiveCameras(device=self.device, 
                    focal_length=focal_length, 
                    image_size=((render_size, render_size),), 
                    principal_point=((render_size*0.5, render_size*0.5),),
                    T=camera_t.reshape(1,3), 
                    R=camera_r)
        raster_settings = RasterizationSettings(
            image_size = render_size,
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
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )

        return renderer
    

    def render(self, verts, faces, bg_img, use_bg, save_object=False, obj_name=None):
        verts = verts.copy()
        faces = faces.copy()

        #print(verts.shape)

        renderer = self.renderer
        render_size = self.render_size

        verts_tensor = torch.from_numpy(verts).float().unsqueeze(0)
        faces_tensor = torch.from_numpy(faces.copy()).long().unsqueeze(0)

        # set color
        mesh_color = self.mesh_color.repeat(1, verts.shape[0], 1)
        textures = Textures(verts_rgb = mesh_color)

        verts_ten = torch.from_numpy(verts).float()
        faces_ten = torch.from_numpy(faces.copy()).long()

        # print(verts_ten.size(), faces_ten.size())

        color = np.ones((224,224,3)) * np.array((0.8,0.53,0.53)).reshape(1,1,3)
        color = torch.from_numpy(color).float()

        if save_object:
            save_obj(obj_name, 
                    verts=verts_ten, 
                    faces=faces_ten)

        #     image_path = "test_out.png"
        #     mtl_path = "test_out.mtl"
        #     output_path = Path("test_out.obj")
        #     path_manager = PathManager()
        #     # Save texture map to output folder
        #     # pyre-fixme[16] # undefined attribute cpu
        #     texture_map = color.detach().cpu() * 255.0
        #     image = Image.fromarray(texture_map.numpy().astype(np.uint8))
        #     with _open_file(image_path, path_manager, "wb") as im_f:
        #         # pyre-fixme[6] # incompatible parameter type
        #         image.save(im_f)
        #     # Create .mtl file with the material name and texture map filename
        #     # TODO: enable material properties to also be saved.
        #     with _open_file(mtl_path, path_manager, "w") as f_mtl:
        #         lines = f"newmtl mesh\n" f"map_Kd {output_path.stem}.png\n"
        #         f_mtl.write(lines)


        # rendering
        mesh = Meshes(verts=verts_tensor, faces=faces_tensor, textures=textures)

        # if save_object:
        #     output_dir = "test_out"
        #     if not os.path.isdir(output_dir):
        #         os.mkdir(output_dir)
        #     p_save_obj(output_dir, 
        #             'pred',
        #             mesh,
        #             decimal_places=10, 
        #             uv_maps=color)

        # blending rendered mesh with background image
        rend_img = renderer(mesh.to(self.device))
        rend_img = rend_img[0].cpu().numpy()
        rend_img_resize = cv2.resize(rend_img, (self.img_size, self.img_size))

        alpha = rend_img_resize[:, :, 3:4]
        alpha[alpha>0] = 1.0
        #print(alpha.shape, rend_img_resize.shape, bg_img.shape)
        if use_bg:
            res_img = alpha * rend_img_resize[:, :, :3]/np.amax(rend_img_resize[:, :, :3])
            # print(rend_img_resize.shape, alpha.shape, res_img.shape)
            res_img = res_img + (1.0 - alpha) * bg_img
        else:
            res_img = rend_img_resize[:, :, :3]/np.amax(rend_img_resize[:, :, :3]) + (1.0 - alpha) * np.ones(bg_img.shape)

        return res_img 
        #return rend_img_resize[:, :, :3]


    



        


    



        