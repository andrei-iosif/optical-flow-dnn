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