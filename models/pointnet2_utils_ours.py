import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as Functional
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




class PointCloud():
    def __init__(self, xyz, intrinsic, img_size):
        self.xyz = xyz
        self.intrinsic = intrinsic
        self.img_size = img_size

    def get_ndc(self, pose):

        K = self.intrinsic
        H = self.img_size
        W = self.img_size

        xyz_world = self.xyz # N,3
        B = xyz_world.size(0)
        n = xyz_world.size(1)
        pad = torch.ones([B, n, 1]).cuda() # (N 1)
        xyz_world = torch.cat([xyz_world, pad], dim=2).cuda() # (B, N, 4)
        #print(xyz_world.unsqueeze(2).shape)  #B,N,N_v,4
        #print(pose.permute(0,1,3,2).unsqueeze(1).shape)   #B,1,N_v,4,4
        xyz_cam = torch.matmul(xyz_world.unsqueeze(2).unsqueeze(3), pose.permute(0,1,3,2).unsqueeze(1)).squeeze(-2) # (B,N,N_v, 4)
        xyz_cam = xyz_cam[:,:,:,:3] # (B N N_v 3) 

        xyz_pix = torch.matmul(K.unsqueeze(0).unsqueeze(0), xyz_cam.permute(0,1,3,2)).permute(0,1,3,2) # B N N_v 3

        z_ndc = xyz_pix[:,:,:,2].unsqueeze(3) #  n 1
        xyz_pix = xyz_pix / (z_ndc.expand(B,n,pose.shape[1], 3))

        x_pix = xyz_pix[:,:,:,0].unsqueeze(3)
        x_ndc = 1 - (2 * x_pix) / (W - 1)
        y_pix = xyz_pix[:,:,:,1].unsqueeze(3)
        y_ndc = 1 - (2 * y_pix) / (H - 1)
        pts_ndc = torch.cat([x_ndc, y_ndc, z_ndc], dim=3)

        return xyz_cam

def create_binary_feature(id_lists, z_lists, N):
        B,H,W,V = id_lists.shape
        # 최종 결과를 저장할 텐서 초기화
        output = torch.zeros((B, N, V))

        # depth 정보를 담고 있는 z_lists를 반복
        for b in range(B):
            for v in range(V):
                valid_indices = id_lists[b, :, :, v].flatten()
                valid_mask = valid_indices != -1
                #valid_depths = 1
                valid_depths = z_lists[b, :, :, v].flatten()[valid_mask]
                valid_indices = valid_indices[valid_mask]

                # output 텐서에 깊이 정보를 할당
                output[b, valid_indices, v] = valid_depths

        return output

def create_camera_poses(camera_positions):
    # 카메라 위치 정규화
    device = camera_positions.device
    camera_positions = torch.tensor(camera_positions, dtype=torch.float32).to(device)

    camera_pose_matrices = torch.eye(4).unsqueeze(0).repeat(camera_positions.shape[0],1,1).to(device)
        
    for i in range(len(camera_positions)):
        if torch.equal(camera_positions[i], torch.tensor([0,0,1], dtype=torch.float32).to(device))|torch.equal(camera_positions[i], torch.tensor([0,0,-1], dtype=torch.float32).to(device)):
            print(camera_positions[i])
            # 카메라 포즈 생성
            translation_vector = torch.tensor([0, 0, 1.0], dtype=torch.float32).to(device)  # 원점(0, 0, 1)을 중심으로 이동
            camera_pose_matrices[i, :3, 3] = translation_vector
        
        else :
            camera_directions = -camera_positions[i] / torch.norm(camera_positions[i], dim=0, keepdim=True).to(device)

            # Z-up 방향 가정
            up_vectors = torch.tensor([0, 0, 1.0], dtype=torch.float32).to(device)

            # 카메라 방향에 대한 오른쪽 벡터 계산
            right_vectors = torch.cross(up_vectors, camera_directions)
            right_vectors = torch.nn.functional.normalize(right_vectors, dim=0)

            # 직교성을 보장하기 위해 up 벡터 재계산
            up_vectors = torch.cross(camera_directions, right_vectors)

            # 카메라 방향 벡터에서 회전 행렬 생성
            rotation_matrices = torch.stack((right_vectors, up_vectors, camera_directions), dim=0)

            camera_pose_matrices[i, :3, :3] = rotation_matrices.permute(1, 0)
            camera_pose_matrices[i, :3, 3] = torch.tensor([0,0,1])
            # camera_pose_matrices[i, :3, 3] = camera_positions[i]

    return camera_pose_matrices

def create_view_feature(points):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        new_xyz: sampled points position data, [B, C', N]
    """
    device = points.device
    B, N, C = points.shape

    vertices = torch.tensor([[1.44337567, 1.44337567, 1.44337567], [1.44337567, 1.44337567, -1.44337567], [1.44337567, -1.44337567, 1.44337567], [1.44337567, -1.44337567, -1.44337567],
                    [-1.44337567, 1.44337567, 1.44337567], [-1.44337567, 1.44337567, -1.44337567], [-1.44337567, -1.44337567, 1.44337567], [-1.44337567, -1.44337567, -1.44337567],
                    [0, 0.89205522, 2.3354309], [0, 0.89205522, -2.3354309], [0, -0.89205522, 2.3354309], [0, -0.89205522, -2.3354309],
                    [2.3354309, 0, 0.89205522], [2.3354309, 0, -0.89205522], [-2.3354309, 0, 0.89205522], [-2.3354309, 0, -0.89205522],
                    [0.89205522, 2.3354309, 0], [-0.89205522, 2.3354309, 0], [0.89205522, -2.3354309, 0], [-0.89205522, -2.3354309, 0]])
    N_v, _ = vertices.shape
    fractal_vertex = torch.Tensor(vertices)   #N'x3
    centroid = fractal_vertex.repeat(B, 1, 1).cuda()   #center of group = verttex (BxN'x3)

    # train set 
    z_list = []
    id_list = []
    image_size = 800
    intrinsic = torch.tensor([[1200, 0, 512],
                            [0, 1200, 384],
                            [0, 0, 1]]).float().cuda()


    theta, pi = calcSphericalCoordinate(centroid)   #BxNx1, BxNx1
    theta = -theta.unsqueeze(-1)
    pi = -pi.unsqueeze(-1)

    zero = torch.zeros(B,N_v,1).cuda()
    ##### Pose Inform ###
    # R = torch.cat([torch.cos(theta)*torch.cos(pi), -torch.sin(theta), torch.sin(theta)*torch.cos(pi), 
    #                 torch.cos(theta)*torch.sin(pi), torch.cos(theta), torch.sin(theta)*torch.sin(pi),
    #                 -torch.sin(theta), zero, torch.cos(theta)],2).reshape(B,N_v,3,3)
    
    # pose = torch.eye(4, dtype=torch.float32).repeat(B,N_v,1,1).cuda()
    pose = create_camera_poses(fractal_vertex).repeat(B,1,1,1).cuda()
    
    # pose[:,:,:3,:3] = R
    # pose[:,:,:,3] = torch.Tensor([0,0,4,1]).repeat(B,N_v,1).cuda()  # 카메라 위치 설정

    pc = PointCloud(points, intrinsic, image_size)
    xyz_ndc = pc.get_ndc(pose)      #B,N,N_v,3
    z_lists = []
    id_lists = []
    for k in range(B):
        z_list = []
        id_list = []
        for r in range(N_v):
            id, zbuf = rasterize(xyz_ndc[k,:,r], (image_size, image_size), 0.03)
            z_list.append(zbuf[0,:,:,0].float().cpu())
            # import matplotlib.pyplot as plt

            # # 예시로 사용할 64x64 이미지 데이터 생성
            # zg = zbuf[0,:,:,0].float().cpu()

            # # 이미지 시각화
            # plt.imshow(zg, cmap='viridis')  # cmap은 색상 맵을 나타내며, 'viridis'는 예시 중 하나입니다.
            # plt.colorbar()  # 컬러바 추가
            # plt.title('Visualization of 64x64 Image')
            # plt.show()
            id_list.append(id[0,:,:,0].long().cpu())
        z_list = torch.stack(z_list,0)
        id_list = torch.stack(id_list,0)
        z_lists.append(z_list)
        id_lists.append(id_list)

    z_lists = torch.stack(z_lists,0).permute(0,3,2,1)    #B,H,W,N_v
    id_lists = torch.stack(id_lists,0).permute(0,3,2,1)  #B,N_v,H,W
    
    binary_feature = create_binary_feature(id_lists, z_lists, N).to(device)# 배열을 파일로 저장
    return binary_feature

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

    
    
    new_bin_map = index_points(bin_map,new_idx) #B,M,N_v
    sqrdists1 = square_distance(new_bin_map, bin_map)

    new_xyz = index_points(xyz, new_idx)#B,M,3
    sqrdists2 = square_distance(new_xyz, xyz)
    
    sqrdists = sqrdists1 + sqrdists2

    group_idx = torch.argsort(sqrdists)[:,:,:nsample]
    return group_idx, new_xyz, new_bin_map

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

    def forward(self,xyz, feature):
        feature = torch.tensor(feature).cuda()
        V = torch.tensor([[1.44337567, 1.44337567, 1.44337567], [1.44337567, 1.44337567, -1.44337567], [1.44337567, -1.44337567, 1.44337567], [1.44337567, -1.44337567, -1.44337567],
                    [-1.44337567, 1.44337567, 1.44337567], [-1.44337567, 1.44337567, -1.44337567], [-1.44337567, -1.44337567, 1.44337567], [-1.44337567, -1.44337567, -1.44337567],
                    [0, 0.89205522, 2.3354309], [0, 0.89205522, -2.3354309], [0, -0.89205522, 2.3354309], [0, -0.89205522, -2.3354309],
                    [2.3354309, 0, 0.89205522], [2.3354309, 0, -0.89205522], [-2.3354309, 0, 0.89205522], [-2.3354309, 0, -0.89205522],
                    [0.89205522, 2.3354309, 0], [-0.89205522, 2.3354309, 0], [0.89205522, -2.3354309, 0], [-0.89205522, -2.3354309, 0]]).cuda()
        V = V.unsqueeze(0).repeat(feature.shape[0], xyz.shape[1], 1, 1)   #B,N,20,3

        feature = torch.tensor(feature).cuda()
        feature = (feature - torch.min(feature, dim=-1,keepdim=True)[0]) / (torch.max(feature, dim=-1,keepdim=True)[0] - torch.min(feature, dim=-1,keepdim=True)[0])
        feature[torch.isnan(feature)] = 0
        feat = torch.concat([feature.unsqueeze(-1), V-xyz.unsqueeze(2).repeat(1,1,20,1)], dim=-1)
        feature = self.R(feat).flatten(start_dim=2)
        return feature

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
    
    combined_feature = torch.concat([xyz, depth_feat], dim=-1)
    fps_idx = farthest_point_sample(combined_feature, npoint) # [B, npoint, C]

    idx, new_xyz, new_bin_map = view_grouping(nsample, depth_feat, xyz, fps_idx)
    #_ = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_feature = index_points(depth_feat, idx)   # [B, npoint, nsample, C]


    #############################################################
    # import plotly.graph_objs as go
    # import numpy as np

    # # ... xyz, new_xyz 데이터가 정의되어 있다고 가정 ...

    # # 그래프 생성
    # fig = go.Figure()

    # # 첫 번째 포인트 클라우드
    # color = np.random.rand(3) * 255  # 무작위 RGB 색상 생성
    # color_str = f'rgb({int(color[0])}, {int(color[1])}, {int(color[2])})'  # RGB 색상 문자열
    # fig.add_trace(go.Scatter3d(
    #     x=xyz[0,:,0].cpu().numpy(),  # x 좌표
    #     y=xyz[0,:,1].cpu().numpy(),  # y 좌표
    #     z=xyz[0,:,2].cpu().numpy(),  # z 좌표
    #     mode='markers',
    #     marker=dict(color='grey', size=8)
    # ))

    # # 두 번째 포인트 클라우드
    # color = np.random.rand(3) * 255  # 무작위 RGB 색상 생성
    # color_str = f'rgb({int(color[0])}, {int(color[1])}, {int(color[2])})'  # RGB 색상 문자열
    # fig.add_trace(go.Scatter3d(
    #     x=new_xyz[0,:,0].cpu().numpy(),  # x 좌표
    #     y=new_xyz[0,:,1].cpu().numpy(),  # y 좌표
    #     z=new_xyz[0,:,2].cpu().numpy(),  # z 좌표
    #     mode='markers',
    #     marker=dict(color='blue', size=8)
    # ))

    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(showbackground=False, showgrid=False, showticklabels=False, zeroline=False),
    #         yaxis=dict(showbackground=False, showgrid=False, showticklabels=False, zeroline=False),
    #         zaxis=dict(showbackground=False, showgrid=False, showticklabels=False, zeroline=False)
    #     ),
    #     paper_bgcolor='rgba(0,0,0,0)',  # 투명 배경
    #     plot_bgcolor='rgba(0,0,0,0)'
    # )

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
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, layer):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.layer = layer
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.group_all = group_all

        if (layer=='1'):
            last_channel = in_channel  + 20*2 + 1 
        elif (layer=='2'):
            last_channel = in_channel + 3 + 20*2 + 1 
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

def calcSphericalCoordinate(xyz):
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]

    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    pi = torch.atan2(y , x)

    return theta, pi



def square_distance(src, dst):
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

# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#     return centroids

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

class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(1024*self.s_views, 512*self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(512*self.s_views, 40*self.s_views))
        
    def forward(self,F,vertices,k):
        id = farthest_point_sample(vertices,self.s_views)
        vertices1 = index_points(vertices,id)
        id_knn = knn(k,vertices,vertices1)
        F = index_points(F,id_knn)
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views*F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0],k,self.s_views,40).transpose(1,2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 1024)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)

        return F_new,F_score,vertices_new

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

class NonLocalMP(nn.Module):
    def __init__(self,n_view):
        super(NonLocalMP,self).__init__()
        self.n_view=n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 1, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Conv2d(2 * 1, 1,1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, F):
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 3)
        F_i = F_i.repeat(1, 1, self.n_view, 1).unsqueeze(-1)
        F_j = F_j.repeat(1, 1, 1, self.n_view).unsqueeze(-1)
        M = torch.cat((F_i, F_j), 4)
        M = self.Relation(M)
        M = torch.sum(M,-2)
        F = torch.cat((F.unsqueeze(-1), M), -1).permute(0,3,2,1)
        F = self.Fusion(F).squeeze()
        return F
    

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

    def forward(self, xyz1, xyz2, points1, points2, feat1, feat2):
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
            dists1 = square_distance(xyz1, xyz2)
            dists2 = square_distance(feat1, feat2)
            dist = dists1 + dists2
            dist, idx = dist.sort(dim=-1)
            dist, idx = dist[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dist + 1e-8)
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
