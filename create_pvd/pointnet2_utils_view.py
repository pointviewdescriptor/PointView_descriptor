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

class PointCloud():
    def __init__(self, xyz, intrinsic, H, W):
        self.xyz = xyz
        self.intrinsic = intrinsic
        self.H = H
        self.W = W

    def get_ndc(self, pose):

        K = self.intrinsic
        H = self.H
        W = self.W

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

def create_binary_feature(depth_maps, z_lists, N):
        B,H,W,N_v = depth_maps.shape

        binary_feature = torch.zeros((B,N_v, N), dtype=torch.float32)

        for j in range(N_v):
            # Depth map 상의 좌표 계산
            pc_data = torch.zeros((B,N+1,))
            pc_data[:,depth_maps[:,:,:,j].flatten(start_dim=1)] = 1
#            pc_data[:,depth_maps[:,:,:,j].flatten(start_dim=1)] = z_lists[:,:,:,j].flatten(start_dim=1)

            # 해당 좌표에 대한 binary 값 설정
            binary_feature[:, j] = pc_data[:,:N]
        
        binary_feature = binary_feature.permute(0,2,1)
        return binary_feature

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

def create_view_feature(points ,SF):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        new_xyz: sampled points position data, [B, C', N]
    """
    device = points.device
    B, N, C = points.shape
    vertices,_,_,_ = SF

    # vertices = torch.tensor([[1.44337567, 1.44337567, 1.44337567], [1.44337567, 1.44337567, -1.44337567], [1.44337567, -1.44337567, 1.44337567], [1.44337567, -1.44337567, -1.44337567],
    #                 [-1.44337567, 1.44337567, 1.44337567], [-1.44337567, 1.44337567, -1.44337567], [-1.44337567, -1.44337567, 1.44337567], [-1.44337567, -1.44337567, -1.44337567],
    #                 [0, 0.89205522, 2.3354309], [0, 0.89205522, -2.3354309], [0, -0.89205522, 2.3354309], [0, -0.89205522, -2.3354309],
    #                 [2.3354309, 0, 0.89205522], [2.3354309, 0, -0.89205522], [-2.3354309, 0, 0.89205522], [-2.3354309, 0, -0.89205522],
    #                 [0.89205522, 2.3354309, 0], [-0.89205522, 2.3354309, 0], [0.89205522, -2.3354309, 0], [-0.89205522, -2.3354309, 0]])
    N_v, _ = vertices.shape
    # N_v, _ = fractal_vertex.shape
    fractal_vertex = torch.Tensor(vertices)   #N'x3
    centroid = fractal_vertex.repeat(B, 1, 1).cuda()   #center of group = verttex (BxN'x3)

    # train set 
    z_list = []
    id_list = []
    H = 800
    W = 800
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

    pc = PointCloud(points, intrinsic, H,W)
    xyz_ndc = pc.get_ndc(pose)      #B,N,N_v,3
    z_lists = []
    id_lists = []
    for k in range(B):
        z_list = []
        id_list = []
        for r in range(N_v):
            id, zbuf = rasterize(xyz_ndc[k,:,r], (H, W), 0.01)
            z_list.append(zbuf[0,:,:,0].float().cpu())
            # import matplotlib.pyplot as plt

            # # 예시로 사용할 64x64 이미지 데이터 생성
            # zg = zbuf[0,:,:,0].float().cpu()

            # # 이미지 시각화
            # plt.imshow(zg, cmap='viridis')  # cmap은 색상 맵을 나타내며, 'viridis'는 예시 중 하나입니다.
            # plt.colorbar()  # 컬러바 추가
            # plt.title('Visualization of 800x800 Image')
            # plt.show()
            id_list.append(id[0,:,:,0].long().cpu())
        z_list = torch.stack(z_list,0)
        id_list = torch.stack(id_list,0)
        z_lists.append(z_list)
        id_lists.append(id_list)

    z_lists = torch.stack(z_lists,0).permute(0,3,2,1)    #B,H,W,N_v
    id_lists = torch.stack(id_lists,0).permute(0,3,2,1)  #B,N_v,H,W
    
    binary_feature = create_binary_feature(id_lists, z_lists,N).to(device)# 배열을 파일로 저장
    
    return binary_feature

def calcSphericalCoordinate(xyz):
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]

    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    pi = torch.atan2(y , x)

    return theta, pi