import torch, torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision.transforms import ToTensor

import numpy as np

from PIL import Image
from typing import Tuple
import json


class VRD_DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_vrd_object_list_path:str,
                 train_vrd_annotations_json_path:str,
                 train_sg_annotations_json_path:str,
                 train_img_folder_path:str,
                 test_vrd_object_list_path:str,
                 test_vrd_annotations_json_path:str,
                 test_sg_annotations_json_path:str,
                 test_img_folder_path:str,
                 test_shuffle = True,
                 test_drop_last = True,
		 train_shuffle= True,
		 train_drop_last=True):
        super(VRD_DataModule, self).__init__()
        self.train_ds = VRD_Dataset(train_vrd_object_list_path,
                                        train_vrd_annotations_json_path,
                                        train_sg_annotations_json_path,
                                        train_img_folder_path)
        self.test_ds = VRD_Dataset(test_vrd_object_list_path,
                                        test_vrd_annotations_json_path,
                                        test_sg_annotations_json_path,
                                        test_img_folder_path)
        self.test_shuffle =test_shuffle
        self.train_shuffle =train_shuffle
        self.test_drop_last = test_drop_last
        self.train_drop_last = train_drop_last
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          shuffle=self.train_shuffle,
                          drop_last=self.train_drop_last)
    
    def val_dataloader(self):
        return DataLoader(self.test_ds, 
                          shuffle=self.test_shuffle,
                          drop_last=self.test_drop_last)


class VRD_Dataset(Dataset):
    def __init__(self,
                 vrd_object_list_path:str,
                 vrd_annotations_json_path:str,
                 sg_annotations_json_path:str,
                 img_folder_path:str):
        '''
        '''
        super(VRD_Dataset, self).__init__()
        try:
            with open(vrd_object_list_path) as f:
                self.object_list = json.load(f)    
        except:
            print('can\'t open the object label file')
        
        try:
            with open(vrd_annotations_json_path) as f:
                self.vrd_annotations = json.load(f)    
        except:
            print('can\'t open the vrd_annotations_json_path')
        
        try:
            with open(sg_annotations_json_path) as f:
                self.sg_annotations_json = json.load(f)    
        except:
            print('can\'t open the sg_annotations_json_path')    
        
        if img_folder_path[-1] != '/':
            img_folder_path = img_folder_path + '/'
        self.img_folder_path = img_folder_path
        img_name_list = list(self.vrd_annotations.keys())
    
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, idx) -> Tuple[Image.Image, float, float, torch.Tensor, torch.Tensor]:
        img_file_name = self.img_name_list[idx]
        
        # --- Load img
        image = Image.open(self.img_folder_path + img_file_name)
        
        # --- img infomation
        find_img_info = lambda value, f_list : filter(lambda x : x['filename']==value, f_list)
        img_info = list(find_img_info(img_file_name, self.sg_annotations_json))[0]
        img_w, img_h = img_info['width'], img_info['height']
        
        # --- class and bb
        # bb format YMIN, YMAX, XMIN, XMAX
        # it shuld be change at XMIN, YMIN, XMAX, YMAX
        img_anno = self.vrd_annotations[img_file_name]
        
        # --- without annotation img
        if len(img_anno) == 0:
            bbs = torch.tensor([]).reshape(-1,4)
            labels = torch.tensor([], dtype=torch.int64).reshape(-1,1)
        # --- with anntation img
        else:
            cls_bb_list = []
            for object_idx in img_anno:
                object_info = list(object_idx['object'].values())
                object_info = [object_info[0], object_info[1][2], object_info[1][0], object_info[1][3], object_info[1][1]]
                subject_info = list(object_idx['subject'].values())
                subject_info = [subject_info[0], subject_info[1][2], subject_info[1][0], subject_info[1][3], subject_info[1][1]]
                if object_info not in cls_bb_list:
                    cls_bb_list.append(object_info)
                
                if subject_info not in cls_bb_list:
                    cls_bb_list.append(subject_info)
            cls_bb_list = np.array(cls_bb_list)
            labels = torch.tensor(cls_bb_list[:,0], dtype=torch.int64)
            bbs = torch.tensor(cls_bb_list[:,1:], dtype=torch.float32)
        return (ToTensor()(image), 
                img_w, 
                img_h, 
                labels, 
                bbs)
                
        
    
