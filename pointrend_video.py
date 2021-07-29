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
import argparse
import json
from glob import glob
import os
import os.path as osp

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend



parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default=None, help='Path to input image')
parser.add_argument('--out', type=str, default=None, help='Path to output image')


if torch.cuda.is_available():
    DEVICE = torch.device("cuda") 
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")



def __get_input_type(args):
    input_type =None
    image_exts = ('jpg', 'png', 'jpeg', 'bmp')
    video_exts = ('mp4', 'avi', 'mov')
    extension = osp.splitext(args.img)[1][1:]

    if extension.lower() in video_exts:
        input_type ='video'
    elif extension.lower() in image_exts:
        input_type = 'image'
        print(extension.lower())
    elif osp.isdir(args.img):
        file_list = os.listdir(args.img)
        assert len(file_list) >0, f"{args.img} is a blank folder"
        extension = osp.splitext(file_list[0])[1][1:]
        assert extension.lower() in image_exts
        input_type ='image_dir'
    else:
        assert False, "Unknown input path. It should be an image, or an image folder, or a video file"
    return input_type


def setup_input(args):
    """
    Input type can be 
        an image file
        a video file
        a folder with image files
    """
    image_exts = ('jpg', 'png', 'jpeg', 'bmp')
    video_exts = ('mp4', 'avi', 'mov')

    # get type of input 
    input_type = __get_input_type(args)

    if input_type =='video':
        cap = cv2.VideoCapture(args.img)
        assert cap.isOpened(), f"Failed in opening video: {args.img}"
        # __video_setup(args)
        return input_type, cap

    elif input_type =='image':
        return input_type, args.img

    elif input_type =='image_dir':
        image_list = gnu.get_all_files(args.img, image_exts, "relative") 
        image_list = [ osp.join(args.img, image_name) for image_name in image_list ]
        # __img_seq_setup(args)
        return input_type, image_list

    else:
        assert False, "Unknown input type"



def gen_video_out(args, the_fps, img_array):
    outVideo_fileName = args.out+'/combined_video.mp4'
    print(f">> Generating video in {outVideo_fileName}")
    
    oneim = img_array[0]
    height, width, layers = oneim.shape
    size = (width,height)
    
    
    out = cv2.VideoWriter(outVideo_fileName,cv2.VideoWriter_fourcc(*'mp4v'), the_fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



if __name__ == '__main__':
    args = parser.parse_args()

    if args.img == None:
        data_path = "/z/syqian/articulation_data/step2_filtered_clips/"
        img_list = glob(data_path+"*.mp4")
        index = np.random.randint(len(img_list))
        args.img = img_list[index]
        if args.out == None:
            args.out = "/data/siyich/cmr_art/random" + str(index) + "_output"
        print("A mp4 video", args.img, "is randomly selected as", str(index))
    else:
        if args.out == None:
            args.out = "/data/siyich/cmr_art"+args.img[:-4]+'_output'


    if not os.path.isdir(args.out):
        os.mkdir(args.out)
        

    input_type, input_data = setup_input(args)
    img_array = []

    cur_frame = 0
    #while cur_frame < 3:
    while True:
        if input_type == 'image':
            img_original_bgr  = cv2.imread(input_data)

        elif input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
        
        else:
            assert False, "Unknown input_type"

        cur_frame += 1
        if img_original_bgr is None:
            break


        im = img_original_bgr

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

        # Show and compare two predictions: 
        # v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # mask_rcnn_result = v.draw_instance_predictions(mask_rcnn_outputs["instances"].to("cpu")).get_image()
        v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        # result_img = np.concatenate((point_rend_result, mask_rcnn_result), axis=0)[:, :, ::-1]
        result_img = point_rend_result[:, :, ::-1]
        cv2.imwrite(args.out+"/mask_" + "frame_" + str(cur_frame) + ".png", result_img)
        cv2.imwrite(args.out+"/origin_" + "frame_" + str(cur_frame) + ".png", img_original_bgr)
        img_array.append(result_img)

        if cur_frame == 1 and input_type == 'image':
            break
    
    if input_type == 'video':
        gen_video_out(args, input_data.get(cv2.CAP_PROP_FPS), img_array)