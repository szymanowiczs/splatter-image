# Functions for reading data from .txt files (ShapeNet-SRN) 
# and .npy files (CO3D)

import os
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import focal2fov, fov2focal
import numpy as np
from pathlib import Path

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


def readCamerasFromTxt(rgb_paths, pose_paths, idxs):
    cam_infos = []
    # Transform fov from degrees to radians
    fovx = 51.98948897809546 * 2 * np.pi / 360

    for idx in idxs:
        cam_name = pose_paths[idx]
        # SRN cameras are camera-to-world transforms
        # no need to change from SRN camera axes (x right, y down, z away) 
        # it's the same as COLMAP (x right, y down, z forward)
        c2w = np.loadtxt(cam_name, dtype=np.float32).reshape(4, 4)

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = rgb_paths[idx]
        image_name = Path(cam_name).stem
        # SRN images already are RGB with white background
        image = Image.open(image_path)

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
        
    return cam_infos

def readCamerasFromNpy(folder_path, 
                       w2c_Rs_rmo=None, 
                       w2c_Ts_rmo=None, 
                       focals_folder_path=None):
    # Set every_5th_in for the testing set
    cam_infos = []
    # Transform fov from degrees to radians
    fname_order_path = os.path.join(folder_path, "frame_order.txt")
    c2w_T_rmo_path = os.path.join(folder_path, "c2w_T_rmo.npy")
    c2w_R_rmo_path = os.path.join(folder_path, "c2w_R_rmo.npy")
    if focals_folder_path is None:
        focals_folder_path = folder_path
    focal_lengths_path = os.path.join(focals_folder_path, "focal_lengths.npy")

    with open(fname_order_path, "r") as f:
        fnames = f.readlines()
    fnames = [fname.split("\n")[0] for fname in fnames]

    if w2c_Ts_rmo is None:
        c2w_T_rmo = np.load(c2w_T_rmo_path)
    if w2c_Rs_rmo is None:
        c2w_R_rmo = np.load(c2w_R_rmo_path)
    focal_lengths = np.load(focal_lengths_path)[:, 0, :]

    camera_transform_matrix = np.eye(4)
    camera_transform_matrix[0, 0] *= -1
    camera_transform_matrix[1, 1] *= -1
    
    # assume shape 128 x 128
    image_side = 128

    for f_idx, fname in enumerate(fnames):

        w2c_template = np.eye(4)
        if w2c_Rs_rmo is None:
            w2c_R = np.transpose(c2w_R_rmo[f_idx])
        else:
            w2c_R = w2c_Rs_rmo[f_idx]
        if w2c_Ts_rmo is None:
            w2c_T = - np.matmul(c2w_T_rmo[f_idx], w2c_R)
        else:
            w2c_T = w2c_Ts_rmo[f_idx]
        w2c_template[:3, :3] = w2c_R
        # at this point the scene scale is approx. that of shapenet cars
        w2c_template[3:, :3] = w2c_T

        # Pytorch3D cameras have (x left, y right, z away axes)
        # need to transform to COLMAP / OpenCV (x right, y down, z forward)
        # transform axes and transpose to column major order
        w2c = np.transpose(np.matmul(w2c_template, camera_transform_matrix))

        # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_name = fname.split(".png")

        focal_lengths_ndc = focal_lengths[f_idx]
        focal_lengths_px = focal_lengths_ndc * image_side / 2

        FovY = focal2fov(focal_lengths_px[1], image_side) 
        FovX = focal2fov(focal_lengths_px[0], image_side)

        cam_infos.append(CameraInfo(uid=fname, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                        image_path=None, image_name=image_name, width=image_side, height=image_side))
        
    return cam_infos
