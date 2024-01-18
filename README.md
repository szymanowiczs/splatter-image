# splatter-image
Official implementation of `Splatter Image: Ultra-Fast Single-View 3D Reconstruction'

# Using this repository

## Installation

1. Create a conda environment: 
```
conda create --name splatter-image
conda activate splatter-image
```

Install Pytorch following [official instructions](https://pytorch.org). Pytorch / Python combinations that were verified to work are: 
- Python 3.7, Pytorch 1.12.1, CUDA 11.6
- Python 3.8, Pytorch 2.1.1, CUDA 12.1

Install other requirements:
```
pip install -r requirements.txt
```

2. Install Gaussian Splatting renderer, i.e. the library for rendering a Gaussian Point cloud to an image. To do so, pull the [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) and, with your conda environment activated, run `pip install submodules/diff-gaussian-rasterization`. You will need to meet the [hardware and software requirements](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md#hardware-requirements). We did all our experimentation on an NVIDIA A6000 GPU and speed measurements on an NVIDIA V100 GPU. 

3. If you want to train on CO3D data you will need to install Pytorch3D. See instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Data

- For training / evaluating on ShapeNet-SRN classes (cars, chairs) please download the srn_\*.zip (\* = cars or chairs) from [PixelNeRF data folder](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing). Unzip the data file and change `SHAPENET_DATASET_ROOT` in `scene/srn.py` to the parent folder of the unzipped folder. For example, if your folder structure is: `/home/user/SRN/srn_cars/cars_train`, in `scene/srn.py` set  `SHAPENET_DATASET_ROOT="/home/user/SRN"`. No additional preprocessing is needed.

- For training / evaluating on CO3D download the hydrant and teddybear classes from the [CO3D release](https://ai.meta.com/datasets/co3d-downloads/). Next, set `CO3D_RAW_ROOT` to your download directory in `data_preprocessing/preoprocess_co3d.py`. Set `CO3D_OUT_ROOT` to where you want to store preprocessed data. Run `python -m data_preprocessing.preprocess_co3d` and set `CO3D_DATASET_ROOT:=CO3D_OUT_ROOT`.

## Pretrained models

Pretrained models will be released in early 2024!

## Training

Single-view models can be trained with the following command:
```
python train_network.py +dataset=[cars,chairs,hydrants,teddybears]
```

To train a 2-view model run:
```
python train_network.py +dataset=cars cam_embd=pose_pos data.input_images=2 opt.imgs_per_obj=5
```

## Evaluation

Once a model is trained evaluation can be run with 
```
python eval.py [model directory path]
```
To save renders modify variable `save_vis` and `out_folder` in eval.py.

## Code structure

Training loop is implemented in `train_network.py` and evaluation code is in `eval.py`. Datasets are implemented in `scene/srn.py` and `scene/co3d.py`. Model is implemented in `scene/gaussian_predictor.py`. The call to renderer can be found in `gaussian_renderer/__init__.py`.

## Camera conventions

Gaussian rasterizer assumes row-major order of rigid body transform matrices, i.e. that position vectors are row vectors. It also requires cameras in the COLMAP / OpenCV convention, i.e., that x points right, y down, and z away from the camera (forward).

# BibTeX

```
@inproceedings{szymanowicz23splatter,
      title={Splatter Image: Ultra-Fast Single-View 3D Reconstruction},
      author={Stanislaw Szymanowicz and Christian Rupprecht and Andrea Vedaldi},
      year={2023},
      booktitle={arXiv},
}
```

# Acknowledgements

S. Szymanowicz is supported by an EPSRC Doctoral Training Partnerships Scholarship (DTP) EP/R513295/1 and the Oxford-Ashton Scholarship.
A. Vedaldi is supported by ERC-CoG UNION 101001212.
We thank Eldar Insafutdinov for his help with installation requirements.
