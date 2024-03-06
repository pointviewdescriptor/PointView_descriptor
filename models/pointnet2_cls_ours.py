import torch.nn as nn
import torch
import numpy as np
import open3d as o3d

import torch.nn.functional as F
from pointnet2_utils_depth import *


class get_model(nn.Module):
    def __init__(self,class_num,args,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.num_point = args.num_point
        self.sa1 = PointNetSetAbstraction(npoint=self.num_point//2, radius=0.2, nsample=self.num_point//32, in_channel=in_channel, mlp=[64*2, 64*2, 128], group_all=False, layer='1')
        self.sa2 = PointNetSetAbstraction(npoint=self.num_point//4, radius=0.4, nsample=self.num_point//16, in_channel=128, mlp=[128, 128, 256], group_all=False, layer='2')
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True, layer='3')
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, class_num)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
        #self.R = nn.Sequential(
        #    nn.Linear(4,4),
        #    nn.LeakyReLU(0.2,inplace=True),
        #    nn.Linear(4,4),
        #    nn.LeakyReLU(0.2,inplace=True),
        #    nn.Linear(4,2),
        #)

    def forward(self, xyz, feature):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            xyz = xyz[:, :3, :]
        ###dodecahedral
        vertices = torch.tensor([[1.44337567, 1.44337567, 1.44337567], [1.44337567, 1.44337567, -1.44337567], [1.44337567, -1.44337567, 1.44337567], [1.44337567, -1.44337567, -1.44337567],
                [-1.44337567, 1.44337567, 1.44337567], [-1.44337567, 1.44337567, -1.44337567], [-1.44337567, -1.44337567, 1.44337567], [-1.44337567, -1.44337567, -1.44337567],
                [0, 0.89205522, 2.3354309], [0, 0.89205522, -2.3354309], [0, -0.89205522, 2.3354309], [0, -0.89205522, -2.3354309],
                [2.3354309, 0, 0.89205522], [2.3354309, 0, -0.89205522], [-2.3354309, 0, 0.89205522], [-2.3354309, 0, -0.89205522],
                [0.89205522, 2.3354309, 0], [-0.89205522, 2.3354309, 0], [0.89205522, -2.3354309, 0], [-0.89205522, -2.3354309, 0]]).cuda()
        vertices = vertices.unsqueeze(0).unsqueeze(0).repeat(feature.shape[0], xyz.shape[2], 1, 1)
        feature = torch.tensor(feature).cuda()
        feature = (feature - torch.min(feature, dim=-1,keepdim=True)[0]) / (torch.max(feature, dim=-1,keepdim=True)[0] - torch.min(feature, dim=-1,keepdim=True)[0])
        feature[torch.isnan(feature)] = 0
        #feat = torch.concat([feature.unsqueeze(-1), vertices-xyz.permute(0,2,1).unsqueeze(2).repeat(1,1,20,1)], dim=-1)
        #feature = self.R(feat).flatten(start_dim=2)

        xyz = torch.tensor(xyz).cuda()

        l1_xyz, l1_points, feature = self.sa1(xyz, feature, norm)
        l2_xyz, l2_points, feature = self.sa2(l1_xyz, feature, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, feature, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        #x = F.log_softmax(x, -1)

        return x,l3_points

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()

    def forward(self, pred, gold, trans, smoothing = True):
        gold = gold.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss