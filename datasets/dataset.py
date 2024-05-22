import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch.utils.data import Dataset

from pathlib import Path
import json

from misc.tum_tools import read_trajectory

class ImageDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)

        # Required data for predictions
        self.images = self.acquire_images()
        self.cam_calibration = self.acquire_cam_calibration()
        self.mask = self.acquire_mask()

        # Optional ground truth
        self.depths, self.depths_available = self.acquire_depths()
        self.poses, self.poses_available = self.acquire_poses()

        self.num_frames = len(self.images)

        #if self.depths_available:
        #    assert self.num_frames == len(self.depths_available)

        #if self.poses_available:
        #    assert self.num_frames == len(self.depths_available)

        self.transform = transform

    def acquire_images(self):
        image_path = self.data_root / "images"
        #breakpoint()
        assert image_path.exists()
        assert image_path.is_dir()

        image_path_list = sorted(list(image_path.glob('*.jpg')) + list(image_path.glob('*.png')))

        image_dict = dict()
        for path in image_path_list:
            image_dict[int(str(path)[-12:-4])] = path

        return image_dict
    
    def acquire_cam_calibration(self):
        cam_calibration_file = self.data_root / "calibration.json"
        assert cam_calibration_file.exists()
        assert cam_calibration_file.is_file()

        cam_calibration_data = None
        with open(cam_calibration_file) as file:
            cam_calibration_data = json.load(file)

        return cam_calibration_data
    
    def acquire_mask(self):
        mask_path = self.data_root / "mask.bmp"
        if not mask_path.exists() or not mask_path.is_file():
            return None
        
        mask = np.array(Image.open(mask_path))
        mask[mask < 255] = 0
        mask[mask == 255] = 255

        if len(mask.shape) > 2:
            mask = mask[..., 0]

        return mask[..., None]
    
    def acquire_depths(self):
        depth_path = self.data_root / "depths"
        if not depth_path.exists() or not depth_path.is_dir():
            return dict(), False

        depth_path_list = sorted(list(depth_path.glob('*.png')))

        depth_dict = dict()
        for path in depth_path_list:
            depth_dict[int(str(path)[-12:-4])] = path

        return depth_dict, True
        
    def acquire_poses(self):
        pose_path = self.data_root / "poses_gt.txt"
        if not pose_path.exists() or not pose_path.is_file():
            return dict(), False

        return read_trajectory(str(pose_path), matrix=True), True

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        assert idx in self.images.keys()

        # Load data in numpy format
        image = np.array(Image.open(self.images[idx]))
        depth = np.zeros_like(image)[..., 0][..., None]
        pose = np.identity(4)
        mask = 255*np.ones_like(image)[..., 0][..., None] if self.mask is None else self.mask


        if self.depths_available and idx in self.depths.keys():
            depth = np.asarray(cv.imread(str(self.depths[idx]), cv.IMREAD_UNCHANGED)).astype(np.float32)[..., None]
            depth /= 1000.
            depth[mask != 255] = 0

        if self.poses_available:
            pose = self.poses[idx]


        intrinisics = self.cam_calibration['intrinsics']
        intrinisics_arr = np.array([intrinisics['fx'], intrinisics['fy'], intrinisics['cx'], intrinisics['cy']])

        sample = {
            'frame_idx': idx,
            'image':image,
            'intrinsics':intrinisics_arr,
            'mask':mask,
            'pose':pose,
            'depth': depth
        }

        if self.transform:
            sample = self.transform(sample)

        return sample