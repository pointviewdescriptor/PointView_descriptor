# PointView-Descriptor: A Versatile Point View Descriptors of Multi-View Depth Images for 3D Shape Classification
# Abstraction
3D shape classification using multi-view point clouds is one of the demanding tasks in computer vision owing to its multitude of applications such as autono-mous driving, robotics, and real-time monitoring. Most methods have studied multi-view capturing to render them 2D images or point clouds and applied a branch of learning models to classify shape categories. The performance of these methods fully confides in the effectiveness of multi-view feature aggregation which limits the overall system and makes it more complex. In this method, we present a structured view descriptor of multi-view capturing in point level to avoid multi-view aggregation. The proposed method explores the relationships among multi-views from a visible point for a specific view. First, the distance-based point descriptors are used to compute distance maps called confidence maps from visible points of specific views to each vertex. We use confidence maps as additional features with point information. Then view information is in-tegrated with distance features by applying a multi-layer perceptron to introduce final view-based point descriptor (PointView-Descriptor). We investigated the performance of several state-of-the-art 3D shape classification methods such as PointNet, PointNet++, and PointView-GCN with PointView-Descriptor. The proposed method dramatically reduces the system complexity and significantly improves performance. It outperforms state-of-the-art methods on ModelNet40 in instance and class levels classification.

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
```
cd create_pvd
python main.py
```
# Training
You can train and test against a variety of vanilla models.
You can choose one of [pointnet_cls, pointnet2_cls_ssg, dgcnn_cls, pointmlp_cls] next to --model.
Below is how the performance of the vanilla model is classified by pointnet++.
```
python train_classification_vanilla.py --model pointnet2_cls_ssg 
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
