import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import math

def calcDistFromCenter(xyz):       #Calc Projection Value
    depth = np.sqrt(np.sum(xyz**2, axis=2))

    return depth

def calcSphericalCoordinate(xyz):   #find spherical coord from xyz coord
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    pi = np.arctan2(y, x)

    return theta, pi

def cvtCoord(r, theta, pi):         #cvt to xyz coord
    z = r * np.cos(theta)
    y = r * np.sin(theta) * np.sin(pi)
    x = r * np.sin(theta) * np.cos(pi)

    return x, y, z

def SphericalProjection(xyz, fractal_vertex):       #projection on spherical fractal
    B, N, _ = xyz.shape

    theta, pi = calcSphericalCoordinate(xyz)

    r = np.sqrt(np.sum(fractal_vertex[0]**2))          #radius of fractal
    
    x, y, z = cvtCoord(r, theta, pi)

    x = np.reshape(x, (B, N, 1))
    y = np.reshape(y, (B, N, 1))
    z = np.reshape(z, (B, N, 1))

    projected_xyz = np.concatenate((x, y, z), axis=2)   #BxNx3

    return projected_xyz

#not in use
def discreteToFractal(projected_pc, SF):                #discretize to fractal vertex
    vertex, near_vertex, triangles, near_tri = SF
    nvertex, _ = vertex.shape       #n(v)x3
    npoints, _ = projected_pc.shape     #Nx3

    xyz = projected_pc[:, :3]
    feature = projected_pc[:, 3]                       #depth of point cloud
    new_features = np.zeros((nvertex, 1))

    near_point_idx = int(near_vertex[0, 0])
    p1 = vertex[0]                                  #xyz of first vertex
    p2 = vertex[near_point_idx]                     #one near point xyz of first vertex
    threshold = np.sqrt(np.sum((p1-p2)**2))

    for i in range(nvertex):
        features = []
        
        for j in range(npoints):
            # if i==j:
            #     pass
            d = np.sqrt(np.sum((vertex[i]-xyz[j])**2))

            if d<=threshold:
                features.append(feature[j])
        
        if len(features)==0:
            new_features[i] = new_features[i-1]
        else:
            new_features[i] = sum(features)/len(features)
        
    new_pc = np.concatenate((vertex, new_features), axis=1)

    return new_pc

def PCtoSF(pc, SF):   #mapping Point Cloud to Spherical Fractal
    B, N, C = pc.shape

    xyz = pc[:, :, :3]
    # features = pc[:, :, 3:]
    fractal_vertex, near_vertex, triangles, near_tri = SF

    new_pc = np.zeros((B, fractal_vertex.shape[0], 3+1))

    depth = calcDistFromCenter(xyz)
    # depth = np.sqrt(np.sum(xyz**2, axis=2))
    projected_xyz = SphericalProjection(xyz, fractal_vertex)
    print(projected_xyz.shape, depth.shape)

    depth = np.reshape(depth, (B, N, 1))
    projected_pc = np.concatenate((projected_xyz, depth), axis=2) #BxNx(3+1+len(features))

    for i in range(B):
        new_pc[i] = discreteToFractal(projected_pc[i], SF)     #Nx(3+1)

    return new_pc, projected_pc           #BxNx(3+1)


#for control
def cvtIcosahedron(in_path, out_path):      #Icosahedron .obj 읽어서 nearest point 6개 찾아서 파일로 저장
    textured_mesh = o3d.io.read_triangle_mesh(in_path)
    vertices = np.asarray(textured_mesh.vertices)
    triangle = np.asarray(textured_mesh.triangles)

    edge = [[] for i in range(len(vertices))]
    tri_p = [[] for i in range(len(vertices))]

    for t in triangle:
        for idx in t:
            tri_p[idx].append(t)
            for i in t:
                if idx!=i and i not in edge[idx]:
                    edge[idx].append(i)

    with open(out_path, 'w') as f:
        for v in vertices:
            f.write('v')
            for p in v:
                f.write(' %f' % p)
            f.write("\n")

        for t in triangle:
            f.write('f')
            for idx in t:
                f.write(' %d' % idx)
            f.write("\n")

        for nnp in edge:
            f.write('np')
            for p in nnp:
                f.write(' %d' % int(p))
            f.write("\n")

        for tp in tri_p:
            f.write('tp')
            for idx in tp:
                f.write(' %d %d %d' % (idx[0], idx[1], idx[2]))
            f.write("\n")

def readIcosahedron(file_path, n_vertices):       #정리해놓은 Icosahedron 파일 읽어서 SF return
    vertices = np.zeros((n_vertices, 3))        #xyz of vertex
    near_idx = np.zeros((n_vertices, 6)) - 1    #idx of 6 neighboring vertices (n(v)x6)                                                                                                                                                    ()
    faces = np.zeros(((n_vertices-2)*2, 3))     #idx of vertices in faces (n(f)x3)
    tris = np.zeros((n_vertices, 6, 3))         #idx of vertices in 6 neighboring faces (n(v)x6x3)

    cnt_v = 0
    cnt_n = 0
    cnt_f = 0
    cnt_t = 0

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()

            if temp[0] == 'v':
                vertices[cnt_v, 0] = temp[1]
                vertices[cnt_v, 1] = temp[2]
                vertices[cnt_v, 2] = temp[3]
                cnt_v += 1

            if temp[0] == 'f':
                faces[cnt_f, 0] = temp[1]
                faces[cnt_f, 1] = temp[2]
                faces[cnt_f, 2] = temp[3]
                cnt_f += 1

            if temp[0] == 'np':
                near_idx[cnt_n, 0] = temp[1]
                near_idx[cnt_n, 1] = temp[2]
                near_idx[cnt_n, 2] = temp[3]
                near_idx[cnt_n, 3] = temp[4]
                near_idx[cnt_n, 4] = temp[5]
                if len(temp)>=7:                      #first icosahedral neighboring point -> 5
                    near_idx[cnt_n, 5] = temp[6]
                cnt_n += 1

            if temp[0] == 'tp':
                for i in range(1, len(temp)):
                    pidx = (i-1)//3                     #vertex index of each face
                    xyidx = (i-1)%3                     #face index of neighboring faces
                    tris[cnt_t, pidx, xyidx] = temp[i] 
                cnt_t += 1


    up_idx = make_upsampling_idx(faces, n_vertices)     #for accelation

    return vertices, near_idx, up_idx, tris

def make_upsampling_idx(faces, n_vertices):     #for accelation
    out_n = (n_vertices-2)*3

    edge = []
    upsam_idx = torch.zeros((out_n, 2))             #saving edge info.
    cnt = 0

    for i, t in enumerate(faces):
        p0, p1, p2 = int(t[0]), int(t[1]), int(t[2])


        if ((p0, p1) not in edge) and ((p1, p0) not in edge):
            edge.append((p0, p1))
            upsam_idx[cnt, 0] = p0
            upsam_idx[cnt, 1] = p1
            
#             output[:, :, n_vertices+cnt] = (data[:, :, p0] + data[:, :, p1]) / 2
            cnt += 1

        if ((p1, p2) not in edge) and ((p2, p1) not in edge):
            edge.append((p1, p2))
            upsam_idx[cnt, 0] = p1
            upsam_idx[cnt, 1] = p2
#             output[:, :, n_vertices+cnt] = (data[:, :, p1] + data[:, :, p2]) / 2
            cnt += 1

        if ((p2, p0) not in edge) and ((p0, p2) not in edge):
            edge.append((p2, p0))
            upsam_idx[cnt, 0] = p0
            upsam_idx[cnt, 1] = p2
            # print(cnt)
#             output[:, :, n_vertices+cnt] = (data[:, :, p2] + data[:, :, p0]) / 2
            cnt += 1

    # print(out_n, cnt)
    
    return upsam_idx