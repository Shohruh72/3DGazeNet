<p align="center">  
  <h1 align="center">Gaze Estimation: 3DGazeNet</h1>

### Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views ([arxiv](https://arxiv.org/abs/2212.02997))
![Vizualization](https://github.com/Shohruh72/3DGazeNet/blob/master/demo/demo.gif)

### Structure

- `nn.py`: Defines the U<sup>2</sup>-Net neural network architecture.
- `util.py`: Contains utility functions and classes.
- `datasets.py`: Handles data loading, preprocessing, and augmentation.
- `main.py`: The main executable script that sets up the model, performs training,testing, and inference.

### Installation

```
conda create -n PyTorch python=3.9
conda activate PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python==4.5.5.64
pip install scipy
pip install tqdm
pip install timm
```

### Dataset Preparation
1. Download the [dataset](https://drive.google.com/file/d/1erYIoTCbXk1amofJ6yTGhbpmsovWrrva/view?usp=sharing) and put it under ``Dataset/``

### Train

* Configure your dataset path in `main.py` for training
* Run `python main.py --train` for Single-GPU training
* Run `bash main.sh $ --train` for Multi-GPU training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Pretrained weight:
saved in `weights` folder

### Demo
* Configure your video path in `main.py` for visualizing the demo
* Run `python main.py --demo` for demo

#### Reference
* https://github.com/deepinsight/insightface
* Face Detection: https://github.com/Shohruh72/SCRFD
