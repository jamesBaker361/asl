import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch
import random
import csv
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.sana.pipeline_sana_video import ASPECT_RATIO_480_BIN, ASPECT_RATIO_720_BIN, SanaVideoPipeline
from diffusers.video_processor import VideoProcessor
from diffusers import AutoencoderKL
from datasets import load_dataset
from typing import Union
import cv2

import numpy as np
import torch.nn.functional as F

def convert(image_cv):
    # 2. Convert BGR to RGB (PyTorch models typically expect RGB)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # 3. Define the ToTensor transform
    # This transform converts a PIL Image or NumPy array to a PyTorch Tensor.
    # It also scales the pixel values from [0, 255] to [0.0, 1.0].
    transform = transforms.ToTensor()

    # 4. Apply the transform to convert the NumPy array to a PyTorch Tensor
    tensor_image = transform(image_rgb)
    return tensor_image

class VideoData(Dataset):
    def __init__(self,ratio:Union[str,float],src_data):
        super().__init__()
        dataset=load_dataset(src_data,split="train")
        self.text_list=dataset["label"]
        aspect_ratio=ASPECT_RATIO_480_BIN[ratio]
        self.tensor_video_list=[]
        for cv2_image_list in dataset["video_cv2"]:
            tensor_list=[]
            for cv2_image in cv2_image_list:
                cv2_image=np.asarray(cv2_image).astype(np.float32)/255.0
                print(cv2_image.shape,cv2_image.size,cv2_image.max(),cv2_image.min(),cv2_image.dtype)
                tens=convert(cv2_image)
                tens=tens.resize(aspect_ratio)
                tensor_list.append(tens)
            self.tensor_video_list.append(torch.stack(tensor_list))
            
    def __len__(self):
        return len(self.tensor_video_list)
    
    def __getitem__(self, index):
        return {
            "video":self.tensor_video_list[index],
            "text":self.text_list[index]
        }
        
if __name__=="__main__":
    data=VideoData("0.5","jlbaker361/wlasl")
    for d in data:
        print(d["video"].size())