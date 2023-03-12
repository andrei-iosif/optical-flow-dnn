# optical-flow-dnn
Optical flow estimation using neural networks

## Requirements
Use the following command to create the conda environment with all necessary dependencies.

```shell
conda env create -f environment.yml
```

## Datasets
Use the following commands to prepare the datasets necessary for training optical flow networks.

### FlyingChairs
```shell
wget https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip
unzip FlyingChairs.zip
cd FlyingChairs_release
wget https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt
```

### FlyingThings3D
This is a large dataset, so we will need a torrent client. For Linux, it's recommended
to use **transmission-cli**:

```shell
sudo apt install transmission-cli
```

Download the FlyingThings3D subset (RGB images and flow ground truth, ~400 GB unzipped):
```shell
transmission-cli https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_image_clean.tar.bz2.torrent -w ./FlyingThings3D
transmission-cli https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_flow.tar.bz2.torrent -w ./FlyingThings3D
```

### Sintel
```shell
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
mkdir Sintel
unzip MPI-Sintel-complete.zip -d ./Sintel/
```

### KITTI 2015
```shell
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
mkdir kitti2015
unzip data_scene_flow.zip ./kitti2015/
```

### HD1K
```shell
wget http://hci-benchmark.iwr.uni-heidelberg.de/media/downloads/hd1k_full_package.zip
```

### Virtual KITTI 2
Official website: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

Download:
```shell
mkdir VirtualKITTI
cd VirtualKITTI
wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_classSegmentation.tar
wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_forwardFlow.tar
tar -xvf vkitti_2.0.3_rgb.tar
tar -xvf vkitti_2.0.3_classSegmentation.tar
tar -xvf vkitti_2.0.3_forwardFlow.tar
```

### VIPER
Official website: https://playing-for-benchmarks.org/download/

Download:
```shell
mkdir VIPER
cd VIPER

# Download RGB images
wget -O train_img_00-77_0_jpg.zip "drive.google.com/u/0/uc?id=1-O7vWiMa3mDNFXUoYxE3vkKZQpiDXUCf&export=download&confirm=yes"
wget -O train_img_00-77_1_jpg.zip "drive.google.com/u/0/uc?id=1alD_fZja9qD7PUnk4AkD6l-jBhlCnzKr&export=download&confirm=yes"

# Download semantic labels
wget -O train_cls_00-77_0.zip "drive.google.com/u/0/uc?id=1lAbmIVuQTLZu4-hNKD20wmGn1SThvFtv&export=download&confirm=yes"
wget -O train_cls_00-77_1.zip "drive.google.com/u/0/uc?id=1KEDYhQeGQ5qOPY2RoTP1btkWmupdR2Sr&export=download&confirm=yes"

# Download flow labels
wget -O train_flow_00-09_0_npz16.zip "drive.google.com/u/0/uc?id=1rXF2FuCTBrGymo3UXT2KnSyJ8OXO_Y_y&export=download&confirm=yes"
wget -O train_flow_10-19_0_npz16.zip "drive.google.com/u/0/uc?id=1HbyFrvZBNdPN7GxvKN11gFwrQHhRE94m&export=download&confirm=yes"
wget -O train_flow_20-29_0_npz16.zip "drive.google.com/u/0/uc?id=1xB9Vg5Jp8-XHvjEaFKpjzQXpoERUoTZB&export=download&confirm=yes"
wget -O train_flow_30-39_0_npz16.zip "drive.google.com/u/0/uc?id=1vZ83ji8woRjoBPGwQciRsRTSZq3lhqyq&export=download&confirm=yes"
wget -O train_flow_40-49_0_npz16.zip "drive.google.com/u/0/uc?id=1-DA6SFtJjEtaAAfu4yi1yHU4mIkf2lyH&export=download&confirm=yes"
wget -O train_flow_50-59_0_npz16.zip "drive.google.com/u/0/uc?id=1RsY8yaFlNNcP49wyZX2UI34MZx3va7EK&export=download&confirm=yes"
wget -O train_flow_60-69_0_npz16.zip "drive.google.com/u/0/uc?id=19vKpozdNFZPNK19OocEXtx8awcpSoRU7&export=download&confirm=yes"
wget -O train_flow_60-69_0_npz16.zip "drive.google.com/u/0/uc?id=1r1wBC2asa-E4E7U59A2Dwhetz8rV_hr-&export=download&confirm=yes"
```