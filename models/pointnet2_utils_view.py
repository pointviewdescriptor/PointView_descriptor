import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from rasterizer.rasterizer import rasterize

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def view_grouping(nsample, bin_map, xyz, new_idx):
    device = bin_map.device
    B, N, C = bin_map.shape
    _, S = new_idx.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    theta, phi = calcSphericalCoordinate(xyz)

    new_bin_map = index_points(bin_map,new_idx) #B,M,N_v
    sqrdists1 = square_distance(new_bin_map, bin_map)

    new_xyz = index_points(xyz, new_idx)#B,M,3
    sqrdists2 = square_distance(new_xyz, xyz)

    sqrdists = sqrdists1 + sqrdists2

    group_idx = torch.argsort(sqrdists)[:,:,:nsample]
    return group_idx, new_xyz, new_bin_map

def sample_and_group(npoint, depth_feat, nsample, xyz, points, radius, layer='0'):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    
    B, N, C = xyz.shape
    S = npoint
    # xyz = (xyz - torch.min(xyz, dim=-1,keepdim=True)[0]) / (torch.max(xyz, dim=-1,keepdim=True)[0] - torch.min(xyz, dim=-1,keepdim=True)[0])
    # xyz[torch.isnan(xyz)] = 0
    # depth_feat = (depth_feat - torch.min(depth_feat, dim=-1,keepdim=True)[0]) / (torch.max(depth_feat, dim=-1,keepdim=True)[0] - torch.min(depth_feat, dim=-1,keepdim=True)[0])
    # depth_feat[torch.isnan(depth_feat)] = 0
    combined_feature = torch.concat([xyz, depth_feat], dim=-1)
    fps_idx = farthest_point_sample(combined_feature, npoint) # [B, npoint, C]

    idx, new_xyz, new_bin_map = view_grouping(nsample, depth_feat, xyz, fps_idx)
    _ = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_feature = index_points(depth_feat, idx)   # [B, npoint, nsample, C]

    #############################################################
    # # visualization
    # import plotly.graph_objects as go

    # # 예제 데이터 생성 (M개의 그룹, 각 그룹에 16개의 데이터)

    # # 그룹 별로 데이터를 시각화
    # fig = go.Figure()
    
    # color = np.random.rand(3) * 255  # 각 그룹에 대해 무작위 RGB 색상 생성
    # fig.add_trace(go.Scatter3d(x=xyz[0,:,0].cpu(), y=xyz[0,:,1].cpu(), z=xyz[0,:,2].cpu(), mode='markers', marker=dict(color='rgb'+str(tuple(color)))))
    
    # # color = np.random.rand(3) * 255  # 각 그룹에 대해 무작위 RGB 색상 생성
    # # fig.add_trace(go.Scatter3d(x=new_xyz[0,:,0].cpu(), y=new_xyz[0,:,1].cpu(), z=new_xyz[0,:,2].cpu(), mode='markers', marker=dict(color='rgb'+str(tuple(color)))))
    # for k in range(512):
    #     color = np.random.rand(3) * 255  # 각 그룹에 대해 무작위 RGB 색상 생성
    #     fig.add_trace(go.Scatter3d(x=grouped_xyz[0,k,:,0].cpu(), y=grouped_xyz[0,k,:,1].cpu(), z=grouped_xyz[0,k,:,2].cpu(), mode='markers', marker=dict(color='rgb'+str(tuple(color)))))

    # # 레이아웃 설정
    # fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    # # 그래프 표시
    # fig.show()
    ###############################################################

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_depth_norm = grouped_feature - new_bin_map.view(B, S, 1, -1)

    if points is not None:
        grouped_points = index_points(points, idx)
        #new_points = torch.cat([grouped_xyz, new_xyz.view(B, S, 1, C).repeat(1,1,nsample,1),grouped_xyz_norm, torch.norm(grouped_xyz_norm, dim=-1, p=2).unsqueeze(-1), grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        #new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        new_points = torch.cat([grouped_xyz_norm, torch.norm(grouped_xyz_norm, dim=-1, p=2).unsqueeze(-1), grouped_depth_norm, grouped_points], dim=-1)
    else:
        #new_points = grouped_depth_norm # [B, npoint, nsample, C+D]
        #new_points = grouped_xyz_norm # [B, npoint, nsample, C+D]
        new_points = torch.cat([grouped_xyz_norm, torch.norm(grouped_xyz_norm, dim=-1, p=2).unsqueeze(-1), grouped_depth_norm], dim=-1) # [B, npoint, nsample, C+D]

    return new_xyz, new_points, new_bin_map

def sample_and_group_all(xyz, points, depth_feat):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)

    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        # new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1), depth_feat.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, depth_feat

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, SF, nsample, in_channel, mlp, group_all, layer):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.layer = layer
        self.SF = SF
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.group_all = group_all

        if (layer=='1'):
            last_channel = in_channel  + 20 + 1
        elif (layer=='2'):
            last_channel = in_channel + 3 + 20 + 1
        else:
            last_channel = in_channel + 3 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, feature, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = torch.tensor(xyz).permute(0, 2, 1).cuda()

        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points, new_bin_map = sample_and_group_all(xyz, points, feature)
        else:
            new_xyz, new_points, new_bin_map = sample_and_group(self.npoint, feature, self.nsample, xyz, points, self.radius, self.layer)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points, new_bin_map

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, SF, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.SF = SF
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, feature, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        feature = torch.tensor(feature).cuda()
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx, new_bin_map = view_grouping(K, feature, farthest_point_sample(xyz, S), self.SF)
            # test_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            # test_xyz = index_points(xyz, test_idx)

            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat, new_bin_map


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


def calcSphericalCoordinate(xyz):
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]

    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    pi = torch.atan2(y , x)

    return theta, pi

def knn(nsample, xyz, new_xyz):
    dist = square_distance(xyz, new_xyz)
    id = torch.topk(dist,k=nsample,dim=1,largest=False)[1]
    id = torch.transpose(id, 1, 2)
    return id

class KNN_dist(nn.Module):
    def __init__(self,k):
        super(KNN_dist, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,1),
        )
        self.k=k
    def forward(self,F,vertices):
        id = knn(self.k, vertices, vertices)
        F = index_points(F.unsqueeze(-2),id)
        v = index_points(vertices,id)
        v_0 = v[:,:,0,:].unsqueeze(-2).repeat(1,1,self.k,1)
        v_F = torch.cat((v_0, v, v_0-v,torch.norm(v_0-v,dim=-1,p=2).unsqueeze(-1)),-1)
        v_F = self.R(v_F)
        F = torch.mul(v_F.unsqueeze(-1), F)
        F = torch.sum(F,2)
        return F

class LocalGCN(nn.Module):
    def __init__(self,k,n_views):
        super(LocalGCN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)
    def forward(self,F,V):
        F = self.KNN(F, V)
        F = F.permute(0,2,1,3)
        F = self.conv(F).squeeze()
        return F

class Featuration(nn.Module):
    def __init__(self):
        super(Featuration,self).__init__()
        self.R = nn.Sequential(
            nn.Linear(4,4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(4,4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(4,1),
        )

    def forward(self,xyz,V):
        xyz = xyz.permute(0,2,1)    #B,N,3
        B,N,_ = xyz.shape
        V = V.repeat(N,1,1,1).permute(1,0,2,3)  #B,N,V,3
        xyz = xyz.repeat(V.shape[2],1,1,1).permute(1,2,0,3)  #B,N,V,3
        feature = torch.norm((xyz-V), dim=-1, p=2)
        feature = (feature - torch.min(feature, dim=-1,keepdim=True)[0]) / (torch.max(feature, dim=-1,keepdim=True)[0] - torch.min(feature, dim=-1,keepdim=True)[0])
        feature[torch.isnan(feature)] = 0
        feat = torch.concat([feature.unsqueeze(-1), V], dim=-1)
        F = self.R(feat).squeeze()
        return F