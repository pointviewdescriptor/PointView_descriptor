#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def Knn(nsample, xyz, new_xyz):
    dist = square_distance(xyz, new_xyz)
    id = torch.topk(dist,k=nsample,dim=1,largest=False)[1]
    id = torch.transpose(id, 1, 2)
    return id

class Featuration(nn.Module):
    def __init__(self):
        super(Featuration,self).__init__()
        self.R = nn.Sequential(
            nn.Linear(4,4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(4,4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(4,2),
        )

    def forward(self, x, feature):
        feature = torch.tensor(feature).cuda()
        #vertex coordinates of dodecahedron
        V = torch.tensor([[1.44337567, 1.44337567, 1.44337567], [1.44337567, 1.44337567, -1.44337567], [1.44337567, -1.44337567, 1.44337567], [1.44337567, -1.44337567, -1.44337567],
                    [-1.44337567, 1.44337567, 1.44337567], [-1.44337567, 1.44337567, -1.44337567], [-1.44337567, -1.44337567, 1.44337567], [-1.44337567, -1.44337567, -1.44337567],
                    [0, 0.89205522, 2.3354309], [0, 0.89205522, -2.3354309], [0, -0.89205522, 2.3354309], [0, -0.89205522, -2.3354309],
                    [2.3354309, 0, 0.89205522], [2.3354309, 0, -0.89205522], [-2.3354309, 0, 0.89205522], [-2.3354309, 0, -0.89205522],
                    [0.89205522, 2.3354309, 0], [-0.89205522, 2.3354309, 0], [0.89205522, -2.3354309, 0], [-0.89205522, -2.3354309, 0]]).cuda()
        V = V.unsqueeze(0).unsqueeze(0).repeat(feature.shape[0], x.shape[2], 1, 1)
        #visibility depth map normalization
        feature = (feature - torch.min(feature, dim=-1,keepdim=True)[0]) / (torch.max(feature, dim=-1,keepdim=True)[0] - torch.min(feature, dim=-1,keepdim=True)[0])
        feature[torch.isnan(feature)] = 0
        #creating descriptor using MLP
        feat = torch.concat([feature.unsqueeze(-1), V-x.permute(0,2,1).unsqueeze(2).repeat(1,1,20,1)], dim=-1)
        feature = self.R(feat).flatten(start_dim=2)
        
        return feature

def get_graph_feature_1(x, feature, k=20, idx=None):
    '''
    [input]
    x : B,C,N
    feature : B,N,M
    [output]
    new_points : B,N,C*2+M*2+1
    '''
    x = x.permute(0,2,1)    #B,N,3
    combined_feature = torch.concat([x, feature], dim=-1)
    idx = view_grouping(k, combined_feature, x)

    grouped_xyz = index_points(x, idx) # [B, npoint, nsample, C]
    grouped_feature = index_points(feature, idx)   # [B, npoint, nsample, M]

    x = x.unsqueeze(2).repeat(1,1,k,1)
    feature = feature.unsqueeze(2).repeat(1,1,k,1)

    grouped_xyz_norm = grouped_xyz - x
    grouped_depth_norm = grouped_feature - feature

    new_points = torch.cat([grouped_xyz_norm, x, torch.norm(grouped_xyz_norm, p=2, dim=-1, keepdim=True),
     grouped_depth_norm, feature], dim=-1).permute(0, 3, 1, 2).contiguous()

    return new_points

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature
