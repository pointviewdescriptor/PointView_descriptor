#Here we extract feature using pre-trained pointnet++ model on the single-view pcd
#The extracted shape feature would be thee input of the GCN for shape classification

import numpy as np
import os
import torch
import sys
from tqdm import tqdm
from ModelNetDataLoader import ModelNetDataLoader
from projection_utils import *
from pointnet2_utils_view import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batchSize", type=int, default=1)
parser.add_argument("-num_class", type=int, default=40)
parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
parser.add_argument('--output_data_path', default='../data/12_depth_norm_001_1M/', help='path of the output feature')
parser.add_argument("--workers", default=0)
parser.set_defaults(train=False)


if __name__ == '__main__':
    args = parser.parse_args()
    data_path = '../data/modelnet40_normal_resampled/'
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10, drop_last=False)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10)

    SF = readIcosahedron('../data_utils/SF0.txt', 12)

    with torch.no_grad():
        for j, (points, target, fn) in tqdm(enumerate(train_loader), total=len(train_loader)):
            Class = [s.replace(',', '') for s in fn[0]]
            print("Processing {} ...".format(Class[0]))
            points = torch.tensor(points[:,:,:3])
            points = points.cuda()

            feature_map = create_view_feature(points, SF)
            
            out_feature_file = args.output_data_path + Class[0] + '/' + 'train'
            if not os.path.exists(out_feature_file):
                os.makedirs(out_feature_file)
            # 튜플에서 파일 이름만 추출
            file_name = fn[1][0].split('/')[-1].split('.')[0]
            feature_save = np.save(f'{out_feature_file}/{file_name}.npz', feature_map.squeeze().cpu().numpy())

        for j, (points, target, fn) in tqdm(enumerate(val_loader), total=len(val_loader)):
            Class = [s.replace(',', '') for s in fn[0]]
            print("Processing {} ...".format(Class[0]))
            points = torch.tensor(points[:,:,:3])
            points = points.cuda()

            feature_map = create_view_feature(points, SF)
            
            
            out_feature_file = args.output_data_path + Class[0] + '/' + 'test'
            if not os.path.exists(out_feature_file):
                os.makedirs(out_feature_file)
            # 튜플에서 파일 이름만 추출
            file_name = fn[1][0].split('/')[-1].split('.')[0]
            feature_save = np.save(f'{out_feature_file}/{file_name}.npz', feature_map.squeeze().cpu().numpy())