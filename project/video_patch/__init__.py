"""Image/Video Patch Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import redos
import todos

import cv2
import torchvision.transforms as T
from PIL import Image
import numpy as np

from . import patch

import pdb


NEIGHBOR_STRIDE = 5  # neighbor radius
REFERENCE_STRIDE = 10

MODEL_H_TILE_SIZE = 240
MODEL_W_TILE_SIZE = 432

def get_nb_list(index, length):
    nb_list = [i for i in range(max(0, index - NEIGHBOR_STRIDE), min(length, index + NEIGHBOR_STRIDE + 1))]
    # nb_list -- [0, 1, 2, 3, 4, 5]
    ref_list = []
    for i in range(0, length, REFERENCE_STRIDE): # ref_length -- step -- 10
        if not i in nb_list:
            ref_list.append(i) # make sure not repeat
    # ref_list -- [10, 20, 30, 40] for length == 50
    return nb_list, ref_list


def dilate(mask):
    for i in range(mask.size(0)):
        image = T.ToPILImage()(mask[i])
        m = np.array(image.convert("L"))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
        m = Image.fromarray(m * 255)
        tensor = T.ToTensor()(m)
        mask[i] = tensor

    return mask


def get_model():
    """Create model."""

    model_path = "models/video_patch.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = patch.VideoPatchModel()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/video_patch.torch"):
        model.save("output/video_patch.torch")

    return model, device


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    # video = redos.video.Sequence(input_file)
    print(video)

    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    image_list = []
    omask_list = []

    def reading_video_frames(no, data):
        data_tensor = todos.data.frame_totensor(data)
        image = data_tensor[:, 0:3, :, :]
        mask = (data_tensor[:, 3:4, :, :] > 0.90).float() # convert 0.0 or 1.0

        image_list.append(image)
        omask_list.append(mask) # orignal mask(labeled)

    video.forward(callback=reading_video_frames)
    image_tensor = torch.cat(image_list, dim = 0)
    omask_tensor = torch.cat(omask_list, dim = 0)

    output_list = [None] * video.n_frames

    print(f"  process {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)
    for index in range(0, video.n_frames,  NEIGHBOR_STRIDE):
        progress_bar.update(NEIGHBOR_STRIDE)
        nb_list, ref_list = get_nb_list(index, video.n_frames)

        selected_imgs = image_tensor[nb_list + ref_list, :, :, :] * 2.0 - 1.0 # from [0, 1.0 --> [-1.0, 1.0]
        selected_mask = omask_tensor[nb_list + ref_list, :, :, :]

        selected_mask = 1.0 - selected_mask
        selected_mask = dilate(selected_mask)
        selected_mask = 1.0 - selected_mask

        input_tensors = selected_imgs * selected_mask

        # pdb.set_trace()       

        output_tensors = todos.model.tile_forward(model, device, input_tensors, 
            h_tile_size=MODEL_H_TILE_SIZE, w_tile_size=MODEL_W_TILE_SIZE, overlap_size=20, scale=1)

        for i in range(len(nb_list)):
            k = nb_list[i]
            # output_tensor = image_tensor[k] * omask_tensor[k] +  output_tensors[i] * (1.0 - omask_tensor[k])

            output_list[k] = output_tensors[i] # output_tensor

    # save
    for index in range(video.n_frames):
        output_temp_file = "{}/{:06d}.png".format(output_dir, index + 1)
        todos.data.save_tensor(output_list[index], output_temp_file)        

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.patch(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_patch", video_service, host, port)


def video_predict(input_file, output_file):
    return video_service(input_file, output_file, None)
