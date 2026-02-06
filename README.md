# PDLNet
The implementation of paper "PDLNet: Learning Point Cloud Distortion for Unsupervised Cross-domain Point Cloud Segmentation in Adverse Weather".

## Installation
```Shell
# use cuda_11.6
conda create -n lidar_weather python=3.8 -y && conda activate lidar_weather
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -U openmim && mim install mmengine && mim install 'mmcv>=2.0.0rc4, <2.1.0' && mim install 'mmdet>=3.0.0, <3.2.0'
git clone https://github.com/engineerJPark/LiDARWeather.git
cd LiDARWeather && pip install -v -e .

pip install cumm-cu116 && pip install spconv-cu116
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
pip install wandb
```


## Data Preparation
```shell

└── semantickitti  
    └── sequences
        ├── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    │── 00
        │    │── ···
        │    └── 21
        ├── labels   <- contains the .label files; a .label file contains the labels of the points in a point cloud
        │    │── 00
        │    │── ···
        │    └── 10
        ├── calib
        │    │── 00
        │    │── ···
        │    └── 21
        └── semantic-kitti.yaml

python ./tools/create_data.py semantickitti --root-path ./data/semantickitti --out-dir ./data/semantickitti --extra-tag semantickitti

/SynLiDAR/
  └── 00/
    └── velodyne
      └── 000000.bin
      ├── 000001.bin
      ...
    └── labels
      └── 000000.label
      ├── 000001.label
      ...
  ...
  └── annotations.yaml
  └── read_data.py

python ./tools/create_data.py synlidar --root-path ./data/SynLiDAR --out-dir ./data/SynLiDAR --extra-tag synlidar

└── SemanticSTF/
    └── train/
        └── velodyne
        └── 000000.bin
        ├── 000001.bin
        ...
        └── labels
        └── 000000.label
        ├── 000001.label
        ...
    └── val/
        ...
    └── test/
        ...
    ...
    └── semanticstf.yaml

python ./tools/create_data.py semanticstf --root-path ./data/SemanticSTF --out-dir ./data/SemanticSTF --extra-tag semanticstf
```


## Getting Started

### Train
```python
tools/dist_train_direct.sh configs/lidarweather_minkunet/*.py 4 0,1,2,3
```

### Test
```python
python tools/test.py configs/lidarweather_minkunet/*.py work_dirs/minkunet_semantickitti/*.pth
```
