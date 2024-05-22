import torch
import numpy as np

import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

class SampleToTensor(object):
    def __init__(self):
        self.np_to_torch_image = ToTensor()

    def __call__(self, sample):
        return {'frame_idx': sample['frame_idx'],
                'image': self.np_to_torch_image(sample['image']),
                'intrinsics': torch.from_numpy(sample['intrinsics']).float(),
                'mask': self.np_to_torch_image(sample['mask']).float(),
                'pose': torch.from_numpy(sample['pose']).float(),
                'depth': self.np_to_torch_image(sample['depth'])
                }

class MaskOutLuminosity(object):
    def __init__(self, threshold_low=0.05, threshold_high=0.95):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def mask_out_luminosity(self, rgb, mask):
        grey = (rgb[0, ...] + rgb[1, ...] + rgb[2, ...])/3.
        grey = grey[None]
        mask[grey < self.threshold_low] = 0
        mask[grey > self.threshold_high] = 0

        return mask

    def __call__(self, sample):
        return {'frame_idx': sample['frame_idx'],
                'image': sample['image'],
                'intrinsics': sample['intrinsics'],
                'mask': self.mask_out_luminosity(sample['image'], sample['mask']),
                'pose': sample['pose'],
                'depth': sample['depth']
                }
    


class SampleToDevice(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return {'frame_idx': sample['frame_idx'],
                'image': sample['image'].to(self.device),
                'intrinsics': sample['intrinsics'].to(self.device),
                'mask': sample['mask'].to(self.device),
                'pose': sample['pose'].to(self.device),
                'depth': sample['depth'].to(self.device),
            }

  
class RescaleImages(object):
    def __init__(self, target_size):
        self.target_size = target_size
    
    def adjust_intrinsics(self, intrinsics, org_shape):
        y_scale = self.target_size[0]/org_shape[1] if self.target_size[0] > 0 else 1.
        x_scale = self.target_size[1]/org_shape[2] if self.target_size[1] > 0 else 1.

        intrinsics[0] *= x_scale
        intrinsics[1] *= y_scale
        intrinsics[2] *= x_scale
        intrinsics[3] *= y_scale

        return intrinsics

    def __call__(self, sample):
        local_target_size = (self.target_size[0] if self.target_size[0] > 0 else sample['image'].shape[1],
                             self.target_size[1] if self.target_size[1] > 0 else sample['image'].shape[2],)
        
        mask_scaled = F.interpolate(sample['mask'][None, ...], size=local_target_size, mode='bilinear')[0, ...]
        mask_scaled[mask_scaled < 1] = 0

        return {'frame_idx': sample['frame_idx'],
                'image': F.interpolate(sample['image'][None, ...], size=local_target_size, mode='bilinear')[0, ...],
                'intrinsics': self.adjust_intrinsics(sample['intrinsics'], sample['image'].shape),
                'mask': mask_scaled,
                'pose': sample['pose'], 
                'depth': F.interpolate(sample['depth'][None, ...], size=local_target_size, mode='bilinear')[0, ...]
            }


class CropImagesToMask(object):
    def __init__(self, ref_mask):
        self.ref_mask = ref_mask
        self.crop_size = self.get_crop_size() # (x1, y1, x2, y2)

    def get_crop_size(self):
        H, W = self.ref_mask.shape[1:]
        bb_min = np.array([100000, 100000])
        bb_max = np.array([-1, -1])
        for x in range(W):
            for y in range(H):
                if self.ref_mask[0, y, x] == 0:
                    continue
                
                bb_min = np.minimum(bb_min, np.array([x, y]))
                bb_max = np.maximum(bb_max, np.array([x, y]))

        self.crop_size = (bb_min[0], bb_min[1], bb_max[0], bb_max[1])
        return self.crop_size
    
    def adjust_intrinsics(self, intrinsics):
        intrinsics[2] -= self.crop_size[0]
        intrinsics[3] -= self.crop_size[1]

        return intrinsics
    def __call__(self, sample):
        return {'frame_idx': sample['frame_idx'],
                    'image': sample['image'][:,  self.crop_size[1]:self.crop_size[3], self.crop_size[0]:self.crop_size[2]],
                    'intrinsics': self.adjust_intrinsics(sample['intrinsics']),
                    'mask': sample['mask'][:,  self.crop_size[1]:self.crop_size[3], self.crop_size[0]:self.crop_size[2]],
                    'pose': sample['pose'], 
                    'depth': sample['depth'][:,  self.crop_size[1]:self.crop_size[3], self.crop_size[0]:self.crop_size[2]],
                }
