#Here we extract feature using multi-view depth map on the pcd

import numpy as np
import os
import torch
from tqdm import tqdm
import sys
from rendering_utils import *
sys.path.append(os.path.abspath('..'))
from data_utils.ModelNetDataLoader import ModelNetDataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batchSize", type=int, default=1)
parser.add_argument('--output_data_path', default='../data/20_depth_norm_test/', help='path of the output feature')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number')
parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
parser.add_argument("--workers", default=0)
parser.set_defaults(train=False)

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = '../data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10, drop_last=False)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10)

    with torch.no_grad():
        for j, (points, target, fn) in tqdm(enumerate(train_loader), total=len(train_loader)):
            Class = [s.replace(',', '') for s in fn[0]]
            print("Processing {} ...".format(Class[0]))
            points = torch.tensor(points[:,:,:3])
            points = points.cuda()
            feature_map = create_descriptor(points)
            
            out_feature_file = args.output_data_path + Class[0] + '/' + 'train'
            if not os.path.exists(out_feature_file):
                os.makedirs(out_feature_file)
            file_name = fn[1][0].split('/')[-1].split('.')[0]
            feature_save = np.save(f'{out_feature_file}/{file_name}.npz', feature_map.squeeze().cpu().numpy())

        for j, (points, target, fn) in tqdm(enumerate(val_loader), total=len(val_loader)):
            Class = [s.replace(',', '') for s in fn[0]]
            print("Processing {} ...".format(Class[0]))
            points = torch.tensor(points[:,:,:3])
            points = points.cuda()
            feature_map = create_descriptor(points)
            
            out_feature_file = args.output_data_path + Class[0] + '/' + 'test'
            if not os.path.exists(out_feature_file):
                os.makedirs(out_feature_file)
            file_name = fn[1][0].split('/')[-1].split('.')[0]
            feature_save = np.save(f'{out_feature_file}/{file_name}.npz', feature_map.squeeze().cpu().numpy())