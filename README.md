# PointView-Descriptor: A Versatile Point View Descriptor of Multi-View Depth Images for 3D Shape Classification
# Abstraction
3D shape classification using multi-view depth images or 3D point clouds is a demanding task in computer vision owing to its multitude of applications in various fields such as autonomous driving, robotics, and real-time monitoring. 
Most methods use direct point cloud processing or the projection of multiple view-points to render multi-view depth images and applied a branch of learning models to classify shape categories. 
The performance of these methods fully relies on the effectiveness of  multi-view feature aggregation or exploring the relationships between local points, which limits the overall system and makes it more complex. 
In this study, we propose a structured view-based feature descriptor for multi-view capturing at the point level to avoid complicated multi-view feature aggregation or local relationship exploration between points. 
The proposed method explores the relationships between multiple-views from a visible point for a specific view. 
We first employed distance-based point descriptors to compute distance maps, called visibility depth maps from the visible points of specific views. 
We then integrated the view information with distance features by applying a multi-layer perceptron to introduce final view-based point descriptors called PointView-Descriptor. 
Subsequently, we investigated the performance of several 3D shape classification methods, including PointNet, PointNet++, DGCNN, PointMLP, and PointView-GCN with PointView-Descriptor. 
The proposed method highly improved the performance of low-complexity models without complex and time-consuming feature extraction and aggregation processes.

# Install 
The provided code was tested on Ubuntu 20.04, cuda 11.7, pytorch 1.13.0, and python 3.7 in two RTX GeForce 4090 environments.
You can run the code by downloading pytorch for your computer, and we conducted the experiment in Anaconda's virtual environment. A virtual environment can be created and run as follows:
```
conda create -n pv-des python=3.7
conda activate pv-des
cd PointView_descriptor
```
# Dataset
You can download the data set used for learning [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save it in ‘data/modelnet40_normal_resampled/’.

# Create View-Descriptor
You can create a view descriptor for a downloaded point cloud. The previously acquired data set can be downloaded [here](https://drive.google.com/file/d/1Fs2Qz9iWePmOAwf-TdslSIbAz0tUt5Ui/view?usp=drive_link), and can be created as follows.
First, You can refer to the rasterize process [here](https://github.com/seanywang0408/RadianceMapping).
```
cd create_pvd
python main.py
```
# Training
You can train and test against a variety of vanilla models.
You can choose one of [pointnet_cls, pointnet2_cls_ssg, dgcnn_cls, pointmlp_cls] next to --model.
Below is how the performance of the vanilla model is classified by pointnet++.
```
python train_classification_origin.py --model pointnet2_cls_ssg 
```
Additionally, the proposed idea can be applied to various vanilla models, and model can select one of the following [pointnet_cls_ours, pointnet2_cls_ssg_ours, dgcnn_cls_ours, pointmlp_cls_ours].
The following is how to conduct an experiment adding the proposed idea to pointnet++.
```
python train_classification.py --model pointnet2_cls_ours
```

# Reference paper and Project Codebase
[Pointnet/Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch?tab=readme-ov-file)
[dgcnn](https://github.com/WangYueFt/dgcnn.git)
[pointmlp](https://github.com/ma-xu/pointMLP-pytorch.git)
[PointView-gcn](https://github.com/SMohammadi89/PointView-GCN.git)
[RadianceMapping](https://github.com/seanywang0408/RadianceMapping)
