'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import torch
import open3d as o3d

from tqdm import tqdm
from torch.utils.data import Dataset
# np.random.seed(42)

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point, centroids


class view_loader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.num_category = args.num_category
        self.norm = args.norm
        self.point_root = self.root + 'modelnet40_normal_resampled'
        ###revise here  [20, SF1]###
        self.feature_root = self.root + '20_depth_norm_001_1M'
        #################

        if self.num_category == 10:
            self.catfile = os.path.join(self.point_root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.point_root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.point_root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.point_root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.point_root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.point_root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.point_datapath = [(shape_names[i], os.path.join(self.point_root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        self.feature_datapath = [(shape_names[i], os.path.join(self.feature_root, shape_names[i], split, shape_ids[split][i]) + '.npz.npy') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.point_datapath)))

        
    def __len__(self):
        return len(self.point_datapath)

    def _get_item(self, index):
        fn = self.feature_datapath[index]
        cls = self.classes[self.feature_datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        feature = np.load(fn[1]).astype(np.float32) #N,42

        fn2 = self.point_datapath[index]

        point_set = np.loadtxt(fn2[1], delimiter=',').astype(np.float32)

        point_set, farthest = farthest_point_sample(point_set, self.npoints)
        feature = feature[farthest.astype(np.int32)]

        if self.norm:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        else:
            point_set[:, 0:3] = point_set[:, 0:3]

        return point_set, feature, label[0]

    def __getitem__(self, index):
        return self._get_item(index)
