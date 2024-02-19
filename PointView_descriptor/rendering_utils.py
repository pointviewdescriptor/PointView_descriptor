import torch
from rasterizer.rasterizer import rasterize

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

def create_distance_descriptor(id_lists, z_lists, N):
        B,H,W,V = id_lists.shape
        # distance_descriptor
        distance_descriptor = torch.zeros((B, N, V))
        # Iterate through z_lists containing depth information
        for b in range(B):
            for v in range(V):
                valid_indices = id_lists[b, :, :, v].flatten()
                valid_mask = valid_indices != -1
                valid_depths = z_lists[b, :, :, v].flatten()[valid_mask]
                valid_indices = valid_indices[valid_mask]
                # Assign depth information to the output tensor
                distance_descriptor[b, valid_indices, v] = valid_depths

        return distance_descriptor

def create_camera_poses(camera_positions):
    # Normalize camera positions
    device = camera_positions.device
    camera_positions = torch.tensor(camera_positions, dtype=torch.float32).to(device)

    camera_pose_matrices = torch.eye(4).unsqueeze(0).repeat(camera_positions.shape[0],1,1).to(device)
        
    for i in range(len(camera_positions)):
        if torch.equal(camera_positions[i], torch.tensor([0,0,1], dtype=torch.float32).to(device))|torch.equal(camera_positions[i], torch.tensor([0,0,-1], dtype=torch.float32).to(device)):
            print(camera_positions[i])
            # creating camera poses
            translation_vector = torch.tensor([0, 0, 1.0], dtype=torch.float32).to(device)
            camera_pose_matrices[i, :3, 3] = translation_vector
        
        else :
            camera_directions = -camera_positions[i] / torch.norm(camera_positions[i], dim=0, keepdim=True).to(device)

            up_vectors = torch.tensor([0, 0, 1.0], dtype=torch.float32).to(device)
            right_vectors = torch.cross(up_vectors, camera_directions)
            right_vectors = torch.nn.functional.normalize(right_vectors, dim=0)
            up_vectors = torch.cross(camera_directions, right_vectors)

            rotation_matrices = torch.stack((right_vectors, up_vectors, camera_directions), dim=0)

            camera_pose_matrices[i, :3, :3] = rotation_matrices.permute(1, 0)
            camera_pose_matrices[i, :3, 3] = torch.tensor([0,0,1])

    return camera_pose_matrices

def create_descriptor(points):
    """
    Input:
        points: input points data, [B, N, C]
    Return:
        new_xyz: point-wise distance descriptor [B, N, V]
    """
    device = points.device
    B, N, C = points.shape

    #Example Dodecahedron Coordinates
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

    pose = create_camera_poses(fractal_vertex).repeat(B,1,1,1).cuda()
    
    pc = PointCloud(points, intrinsic, image_size)
    xyz_ndc = pc.get_ndc(pose)      #B,N,N_v,3
    z_lists = []
    id_lists = []
    for k in range(B):
        z_list = []
        id_list = []
        for r in range(N_v):
            id, zbuf = rasterize(xyz_ndc[k,:,r], (image_size, image_size), 0.01)
            z_list.append(zbuf[0,:,:,0].float().cpu())
            id_list.append(id[0,:,:,0].long().cpu())
        z_list = torch.stack(z_list,0)
        id_list = torch.stack(id_list,0)
        z_lists.append(z_list)
        id_lists.append(id_list)

    z_lists = torch.stack(z_lists,0).permute(0,3,2,1)    #B,H,W,N_v
    id_lists = torch.stack(id_lists,0).permute(0,3,2,1)  #B,N_v,H,W
    
    distance_descriptor = create_distance_descriptor(id_lists, z_lists, N).to(device)# 배열을 파일로 저장
    return distance_descriptor

def calcSphericalCoordinate(xyz):
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]

    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    pi = torch.atan2(y , x)

    return theta, pi