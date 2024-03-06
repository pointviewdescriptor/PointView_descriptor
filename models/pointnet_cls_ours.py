import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils_ours import PointNetEncoder, feature_transform_reguliarzer, Featuration

class get_model(nn.Module):
    def __init__(self, class_num, args, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, class_num)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.R = nn.Sequential(
            nn.Linear(4,4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(4,4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(4,2),
        )


    def forward(self, x, feature):
        x = x[:,:3,:]
        vertices = torch.tensor([[1.44337567, 1.44337567, 1.44337567], [1.44337567, 1.44337567, -1.44337567], [1.44337567, -1.44337567, 1.44337567], [1.44337567, -1.44337567, -1.44337567],
                [-1.44337567, 1.44337567, 1.44337567], [-1.44337567, 1.44337567, -1.44337567], [-1.44337567, -1.44337567, 1.44337567], [-1.44337567, -1.44337567, -1.44337567],
                [0, 0.89205522, 2.3354309], [0, 0.89205522, -2.3354309], [0, -0.89205522, 2.3354309], [0, -0.89205522, -2.3354309],
                [2.3354309, 0, 0.89205522], [2.3354309, 0, -0.89205522], [-2.3354309, 0, 0.89205522], [-2.3354309, 0, -0.89205522],
                [0.89205522, 2.3354309, 0], [-0.89205522, 2.3354309, 0], [0.89205522, -2.3354309, 0], [-0.89205522, -2.3354309, 0]]).cuda()
        vertices = vertices.unsqueeze(0).unsqueeze(0).repeat(feature.shape[0], x.shape[2], 1, 1)
        feature = torch.tensor(feature).cuda()
        feature = (feature - torch.min(feature, dim=-1,keepdim=True)[0]) / (torch.max(feature, dim=-1,keepdim=True)[0] - torch.min(feature, dim=-1,keepdim=True)[0])
        feature[torch.isnan(feature)] = 0
        feat = torch.concat([feature.unsqueeze(-1), vertices-x.permute(0,2,1).unsqueeze(2).repeat(1,1,20,1)], dim=-1)
        feature = self.R(feat).flatten(start_dim=2).permute(0,2,1)

        x_in = torch.cat([x,feature], dim=1)
        x, trans, trans_feat = self.feat(x_in)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


