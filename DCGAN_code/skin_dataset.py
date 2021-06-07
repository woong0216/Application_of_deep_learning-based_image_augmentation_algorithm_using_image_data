# Skin cancer 데이터셋 이미지를 import 하기 위한 파일.
# 데이터 구성은 데이터셋 이름 폴더 > train or test 폴더 > label(benign or malignant) 폴더로 구성되어 있어야 한다.
# 마지막 파일이 속한 폴더(benign or malignant) 이름을 해당 파일의 label로 정하게 된다.
# 각 이미지는 3 x 64 x 64 크기의 이미지로 변환되어 사용된다.

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import glob

class SKIN_CANCER(Dataset):

    def __init__(self, data, data_type, label=None, img_size=64):
        
        self.img_list = glob.glob(data+'\\'+data_type+'\\'+ label +'\\*') if label is not None else glob.glob(data+'\\'+data_type+'\\'+ '*\\*')
        self.transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                             transforms.ToTensor()
                                            ])
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        single_img_path = self.img_list[idx]
        
        im = Image.open(single_img_path)
        img = self.transform(im)
        
        label = self.img_list[idx].split('\\')[-2]
        label = 1 if label == 'malignant' else 0

        return img, label
