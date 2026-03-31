#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import numpy
import torch
from .BaseDataset import BaseDataset
import os
import cv2
import io
from config import config
from io import BytesIO
import pickle
import copy
import random
from PIL import Image
from torchvision import transforms
import pickle5 as pickle
import os.path as osp

class OccScanNet(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, s3client=None):
        super(OccScanNet, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._root = setting['root']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.s3client = s3client

        self._root = osp.join('/media/psdz/data/dataset/', 'occscannet/')

        self._gt_path = osp.join('/media/psdz/data/dataset/', 'occscannet/')
        self._train_source = osp.join(self._root, "train_subscenes.txt")
        self._eval_source = osp.join(self._root, "train_subscenes.txt")


        # OccScanNet specific parameters
        self.n_classes = 12

        # Color jitter for augmentation
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4) if split_name == 'train' else None

        # Normalize RGB
        self.normalize_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']

        # Read subscenes list
        subscenes_list = os.path.join(self._root, f'{split_name}_subscenes.txt')
        with open(subscenes_list, 'r') as f:
            used_subscenes = f.readlines()
            for i in range(len(used_subscenes)):
                used_subscenes[i] = os.path.join(self._root, used_subscenes[i].strip())
        #used_subscenes= used_subscenes[:2]

        return used_subscenes

    def __len__(self):
        if self._file_length is not None:
            return len(self._file_names)
        return len(self._file_names)
        
    def get_length(self):
        return self.__len__()
    def __getitem__(self, index):

        names = self._file_names[index]
        # Load data from pickle file
        with open(names, 'rb') as f:
            data = pickle.load(f)

        cam_intrin = data['intrinsic']

        # Load image
        img_path = data['img']
        # Replace path to match your dataset location
        if '/scannet' in img_path:
            index = img_path.find("/scannet")
            short_path = img_path[index:]
            short_path = short_path.replace("scannet/posed_images", "occscannet/gathered_data")
            img_path = '/media/psdz/data/dataset' + short_path

        img = cv2.imread(img_path)
        if img is None:
            # If image not found, try to get a random sample
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load depth
        depth_path = data['depth_gt']
        if '/scannet' in depth_path:
            index = depth_path.find("/scannet")
            short_path = depth_path[index:]
            short_path = short_path.replace("scannet/posed_images", "occscannet/gathered_data")
            depth_path = '/media/psdz/data/dataset' + short_path

        depth_img = Image.open(depth_path).convert('I;16')
        depth_img = np.array(depth_img) / 1000.0

        # Resize image and adjust camera intrinsics
        img_H, img_W = img.shape[0], img.shape[1]
        img = cv2.resize(img, (640, 480))
        W_factor = 640 / img_W
        H_factor = 480 / img_H
        img_H, img_W = img.shape[0], img.shape[1]

        cam_intrin[0, 0] *= W_factor
        cam_intrin[1, 1] *= H_factor
        cam_intrin[0, 2] *= W_factor
        cam_intrin[1, 2] *= H_factor

        # Get target voxel grid
        target = data["target_1_4"]

        #label_weight = ((target != 0) &(target!=255)).astype(np.float)
        label_weight = ((target != 255).astype(np.float))

        #target = np.where(target == 255, 0, target)
        if target.sum() == 0:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # Convert image to tensor
        img = Image.fromarray(img).convert('RGB')

        # Image augmentation
        # if self.color_jitter is not None and self._split_name == 'train':
        #     img = self.color_jitter(img)


        # Random horizontal flip (needs 3D–2D projection update if enabled)
        # if self._split_name == 'train' and np.random.rand() < 0.5:
        #     img = np.ascontiguousarray(np.fliplr(img))
        #     depth_img = np.ascontiguousarray(np.fliplr(depth_img))

        # Normalize image
        img = self.normalize_rgb(img)

        prefix = '/media/psdz/data/dataset/occscannet/'
        suffix = '.pkl'
        relative_path = names[len(prefix):]
        relative_path = relative_path[:-len(suffix)]
        depth_mapping_3d_path = '/media/psdz/data/dataset/3D_mapping/' + relative_path +'.npy.npz'
        depth_mapping_3d = np.load(depth_mapping_3d_path)
        depth_mapping_3d = depth_mapping_3d['arr_0']
        depth_mapping_3d = torch.from_numpy(depth_mapping_3d).long()
        depth_mapping_3d = depth_mapping_3d.view(-1)
        # Convert to tensors
        if self._split_name == 'train':
            
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            #img = img.permute(2, 0, 1)
            target = torch.from_numpy(np.ascontiguousarray(target)).long()
            depth_img = torch.from_numpy(np.ascontiguousarray(depth_img)).float()



        hha = torch.zeros(img.shape, dtype=torch.float32) # Dummy HHA data
        tsdf = torch.zeros((1, 60, 36, 60),dtype=torch.float32)  # Dummy TSDF data
        sketch_gt = torch.zeros(target.shape)  # Dummy sketch ground truth
        seg_2D = torch.zeros(480, 640)  # Dummy 2D segmentation
        gt_sc = torch.zeros(target.shape)  # Dummy scene completion ground truth

        output_dict = dict(
            data=img,
            label=target,
            label_weight=label_weight,
            depth_mapping_3d=depth_mapping_3d,
            tsdf=tsdf,
            sketch_gt=sketch_gt,
            seg=seg_2D,
            fn=str(names),
            n=len(self._file_names),
            label_sc=gt_sc,
            hha_img=hha,
            depth_gt=depth_img,
        )

        return output_dict


    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 12  # OccScanNet has 12 classes
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
