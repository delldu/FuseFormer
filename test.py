# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms

from core.utils import Stack, ToTorchFormatTensor
import pdb
from tqdm import tqdm

parser = argparse.ArgumentParser(description="FuseFormer")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-m", "--mask",   type=str, required=True)
parser.add_argument("-c", "--ckpt",   type=str, required=True)
parser.add_argument("--model", type=str, default='fuseformer')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--outw", type=int, default=432)
parser.add_argument("--outh", type=int, default=240)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)
parser.add_argument("--use_mp4", action='store_true')
args = parser.parse_args()


W, H = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    # neighbor_ids -- [0, 1, 2, 3, 4, 5]
    ref_index = []
    if num_ref == -1: # True
        for i in range(0, length, ref_length): # ref_length -- step -- 10
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                #if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)

    # [10, 20, 30, 40] for length == 50
    return ref_index


# read frame-wise masks 
def read_mask(mpath):
    # mpath -- 'data/DAVIS/Annotations/blackswan'

    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((W, H), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    # masks[0] -- <PIL.Image.Image image mode=L size=864x480 at 0x7F9BDC097C10>
    return masks


#  read frames from video 
def read_frame_from_videos(args):
    vname = args.video # args.video -- 'data/DAVIS/JPEGImages/blackswan'

    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((W,H)))
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname+'/'+name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((W,H)))

    # len(frames) -- 50
    # frames[0] -- <PIL.Image.Image image mode=RGB size=864x480 at 0x7F04EC221460>
    # frames[0].getpixel((0,0)) --(61, 80, 35)
    return frames       


def main_worker():
    # set up models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)  # args.model -- 'fuseformer'
    model = net.InpaintGenerator().to(device)

    model_path = args.ckpt # args.ckpt -- 'checkpoints/fuseformer.pth'

    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print('loading from: {}'.format(args.ckpt))
    model.eval()

    # prepare datset, encode all frames into deep space 
    frames = read_frame_from_videos(args)

    video_length = len(frames)
    imgs = _to_tensors(frames).unsqueeze(0)*2.0 - 1.0
    # imgs.size() -- torch.Size([1, 50, 3, 240, 432]) # Element range [-1.0, 1.0]

    frames = [np.array(f).astype(np.uint8) for f in frames] # frames[0].shape -- (240, 432, 3), Element range [0, 255]


    masks = read_mask(args.mask)

    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    #  len(binary_masks), binary_masks[0].shape, binary_masks[0].max(), binary_masks[0].min()
    # (50, (240, 432, 1), 1, 0)

    masks = _to_tensors(masks).unsqueeze(0)
    # masks.size() -- torch.Size([1, 50, 1, 240, 432])


    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None]*video_length
    print('loading videos and masks from: {}'.format(args.video))

    # completing holes by spatial-temporal transformers

    # video_length, neighbor_stride -- (50, 5)
    progress_bar = tqdm(total=video_length)
    for f in range(0, video_length, neighbor_stride):
        progress_bar.update(neighbor_stride)

        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)

        # if f == 10:
        #   neighbor_ids -- [5, 6, 7, 8, 9, <10>, 11, 12, 13, 14, 15]
        #   ref_ids -- [0, 20, 30, 40]
        #   neighbor_ids+ref_ids -- [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 20, 30, 40]

        # print("f = ", f, "neighbor_ids = ", neighbor_ids, "ref_ids =", ref_ids)

        # len_temp = len(neighbor_ids) + len(ref_ids)
        selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]

        masked_imgs = selected_imgs*(1-selected_masks)            

        with torch.no_grad():
            pred_img = model(masked_imgs)
        # masked_imgs.size() -- [1, 7, 3, 240, 432]
        # pred_img.size() -- [7, 3, 240, 432]

        # if f == 0
        # masked_imgs.size() -- [1, 10, 3, 240, 432]
        # pred_img.size() -- [10, 3, 240, 432]

        # if f == 10:
        #     masked_imgs.size() -- [1, 15, 3, 240, 432]
        #     pred_img.size() -- [15, 3, 240, 432]

        pred_img = (pred_img + 1) / 2   # [-1, 1.0] -> [0.0, 1.0]
        pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255

        # neighbor_ids --[0, 1, 2, 3, 4, 5] ?
        for i in range(len(neighbor_ids)):
            idx = neighbor_ids[i]
            img = np.array(pred_img[i]).astype(
                np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
            # if comp_frames[idx] is None:
            #     comp_frames[idx] = img
            # else:
            #     comp_frames[idx] = comp_frames[idx].astype(
            #         np.float32)*0.5 + img.astype(np.float32)*0.5
            comp_frames[idx] = img

    name = args.video.strip().split('/')[-1]
    writer = cv2.VideoWriter(f"{name}_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        if W != args.outw:
            comp = cv2.resize(comp, (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(f"{name}_result.mp4"))


if __name__ == '__main__':
    main_worker()
