# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from DBoW.r2d2.extract import extract_r2d2, load_configs_from_json

class R2D2():
    def __init__(self, json_file_path='./DBoW/r2d2_config.json'):
        self.configs_dict = load_configs_from_json(json_file_path)
        
    def r2d2_features(self, img_dict, topk):
        """
        Extract r2d2 features
        """
        print('Extracting r2d2 features ...') 
        descriptors = []
        for i in tqdm(img_dict.keys()):
            keypoints, descriptor = extract_r2d2(self.configs_dict, img_dict[i], topk)
            if descriptor is not None:
                descriptor = descriptor.tolist()
                descriptors.append(descriptor)
        return descriptors

    def update_image(self, img_name, topk):
        
        keypoints, descriptor = extract_r2d2(self.configs_dict, img_name, topk)
        if descriptor is not None:
            descriptor = descriptor.tolist()
            return keypoints, descriptor
        else:
            return None, None
