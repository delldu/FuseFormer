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


PATCH_ZEROPAD_TIMES = 8
PATCH_NEIGHBOR_RADIUS = 5  # neighbor

MODEL_H_TILE_SIZE = 240
MODEL_W_TILE_SIZE = 432


def dialte(mask):
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
    print(video)

    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    frame_list = []

    def reading_video_frames(no, data):
        data_tensor = todos.data.frame_totensor(data)
        # data_tensor = todos.data.resize_tensor(data_tensor, 240, 432)

        frame_list.append(data_tensor)

    video.forward(callback=reading_video_frames)

    print(f"  process {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)
    for index in range(video.n_frames):
        progress_bar.update(1)

        start = max(index - PATCH_NEIGHBOR_RADIUS, 0)
        stop = min(index + PATCH_NEIGHBOR_RADIUS, video.n_frames - 1)
        sub_frame_list = [frame_list[j] for j in range(start, stop + 1)]

        input_tensor = torch.cat(sub_frame_list, dim=0)
        image_tensor = input_tensor[:, 0:3, :, :] * 2.0 - 1.0  # [0.0, 1.0] -> [-1.0, 1.0]
        mask_tensor = (input_tensor[:, 3:4, :, :] > 0.95).float()

        mask_tensor = 1.0 - mask_tensor
        mask_tensor = dialte(mask_tensor)
        mask_tensor = 1.0 - mask_tensor

        image_tensor = image_tensor * mask_tensor

        # image_tensor = todos.data.resize_tensor(image_tensor, 240, 432)
        output_tensor = todos.model.tile_forward(model, device, image_tensor, h_tile_size=MODEL_H_TILE_SIZE, w_tile_size=MODEL_W_TILE_SIZE, overlap_size=20, scale=1)

        output_index = index - start
        temp_output_tensor = output_tensor[output_index : output_index + 1, :, :, :]
        # temp_output_tensor = todos.data.resize_tensor(temp_output_tensor, video.height, video.width)

        image_tensor = frame_list[index][:, 0:3, :, :]
        mask_tensor = (frame_list[index][:, 3:4, :, :] > 0.95).float()
        output_tensor = image_tensor * mask_tensor + temp_output_tensor.cpu() * (1.0 - mask_tensor)

        output_temp_file = "{}/{:06d}.png".format(output_dir, index + 1)
        todos.data.save_tensor(output_tensor, output_temp_file)

    redos.video.encode(output_dir, output_file)

    # # delete temp files
    # for i in range(video.n_frames):
    #     temp_output_file = "{}/{:06d}.png".format(output_dir, i)
    #     os.remove(temp_output_file)

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
