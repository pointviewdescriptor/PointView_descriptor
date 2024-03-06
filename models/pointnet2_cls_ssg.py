import torch
import numpy as np
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F
import provider
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,num_point,normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.num_point = num_point
        self.sa1 = PointNetSetAbstraction(npoint=self.num_point//2, radius=0.2, nsample=self.num_point//32, in_channel=in_channel, mlp=[64,64,128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=self.num_point//4, radius=0.4, nsample=self.num_point//16, in_channel=128, mlp=[128,128,256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256,512,1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            # Assuming your PyTorch tensor is named point_cloud_tensor and has shape [N, 3]
            point_cloud_tensor = xyz[:,:3].permute(0,2,1)

            # Convert to NumPy array
            point_cloud_np = point_cloud_tensor.cpu().numpy().astype(np.float64)
            norm = []
            for i in range(B):
                # Create an Open3D PointCloud object
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud_np[i])
                pcd.estimate_normals()

                pcd.orient_normals_towards_camera_location()
                # pcd.orient_normals_to_align_with_direction()
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
                # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
                # o3d.visualization.draw_geometries([pcd],
                #                                   zoom=0.3412,
                #                                   front=[0.4257, -0.2125, -0.8795],
                #                                   lookat=[2.6172, 2.0475, 1.532],
                #                                   up=[-0.0694, -0.9768, 0.2024],
                #                                   point_show_normal=True)

                # print("Print a normal vector of the 0th point")
                # print(pcd.normals[0])
                # print("normal을 numpy로 바꾸고 10개 보여주기")
                # print(np.asarray(pcd.normals)[:10, :])
                norm.append(torch.tensor(pcd.normals))
            norm = torch.stack(norm, dim=0).cuda()
            norm = provider.normalize_data((norm).cpu().numpy())
            norm = torch.tensor(norm).permute(0,2,1).float().cuda()

            # norm = torch.norm(norm, p=2, dim=-1)
            # norm = (norm - torch.min(norm, dim=-1,keepdim=True)[0]) / (torch.max(norm, dim=-1,keepdim=True)[0] - torch.min(norm, dim=-1,keepdim=True)[0])
            norm[torch.isnan(norm)] = 0
            # print(norm.shape)
            # norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x, l3_points


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
#class get_loss(nn.Module):
#    def __init__(self):
#        super(get_loss, self).__init__()

#    def forward(self, pred, target, trans_feat):
#        total_loss = F.nll_loss(pred, target)

#        return total_loss
