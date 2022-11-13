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