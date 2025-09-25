import os
import glob
import json
import math
import torch
import torchvision
import numpy as np

from PIL import Image

from .shared_dataset import SharedDataset

from utils.graphics_utils import getProjectionMatrix, fov2focal
from utils.camera_utils import get_loop_cameras

class Zero123PlusDataset(SharedDataset):
    def __init__(self,
                 cfg,
                 dataset_name = "train"
                 ) -> None:

        super(Zero123PlusDataset).__init__()
        self.cfg = cfg
        self.root_dir = cfg.data.root_dir

        # Get all object directories
        self.paths = []
        for obj_dir in sorted(os.listdir(self.root_dir)):
            obj_path = os.path.join(self.root_dir, obj_dir)
            if os.path.isdir(obj_path):
                # Check if we have 6 views (001-006) and corresponding .npy files
                required_files = []
                for i in range(1, 7):  # 001-006
                    png_file = os.path.join(obj_path, f"{i:03d}.png")
                    npy_file = os.path.join(obj_path, f"{i:03d}.npy")
                    if os.path.exists(png_file) and os.path.exists(npy_file):
                        required_files.append(png_file)

                if len(required_files) == 6:  # Only include objects with all 6 views
                    self.paths.append(obj_dir)

        # Split dataset for training and validation
        total_objects = len(self.paths)
        self.dataset_name = dataset_name
        if self.dataset_name == "val" or dataset_name == "vis":
            # Use last 10% as validation
            self.paths = self.paths[math.floor(total_objects * 0.9):]
        elif self.dataset_name == "test":
            # Use middle 10% as test
            start_idx = math.floor(total_objects * 0.45)
            end_idx = math.floor(total_objects * 0.55)
            self.paths = self.paths[start_idx:end_idx]
        else:
            # Use first 80% as training
            self.paths = self.paths[:math.floor(total_objects * 0.8)]

        if cfg.data.subset != -1:
            self.paths = self.paths[:cfg.data.subset]

        print('============= Zero123Plus Dataset length: %d =============' % len(self.paths))

        # Use 35mm FOV as specified
        fov_degrees = 35.0
        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=fov_degrees * np.pi / 180,
            fovY=fov_degrees * np.pi / 180).transpose(0,1)

        self.image_side_target = self.cfg.data.training_resolution

        # Camera coordinate conversion (OpenGL to COLMAP/OpenCV)
        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)

        self.imgs_per_obj_train = self.cfg.opt.imgs_per_obj

    def __len__(self):
        return len(self.paths)

    def load_imgs_and_convert_cameras(self, obj_path, num_views):
        """
        Load the images, camera matrices and projection matrices for a given object
        """
        # Black background for transparent images
        bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        world_view_transforms = []
        view_world_transforms = []

        camera_centers = []
        imgs = []
        fg_masks = []

        # Available view indices (001-006)
        available_indices = list(range(1, 7))

        # For validation, use fixed views for reproducibility
        if self.dataset_name != "train":
            view_indices = available_indices[:num_views]
        else:
            # For training, randomly sample views
            if num_views <= len(available_indices):
                view_indices = np.random.choice(available_indices, num_views, replace=False)
            else:
                view_indices = available_indices

            # Ensure we include conditioning views first if multiple input images
            if self.cfg.data.input_images > 1:
                # First input_images views are conditioning views
                conditioning_indices = view_indices[:self.cfg.data.input_images]
                all_indices = np.concatenate([conditioning_indices, view_indices])
                view_indices = all_indices

        # Load the images and cameras
        for view_idx in view_indices:
            # Load image
            img_path = os.path.join(obj_path, f"{view_idx:03d}.png")
            img = Image.open(img_path)

            # Resize to training resolution
            img = torchvision.transforms.functional.resize(img,
                                                self.cfg.data.training_resolution,
                                                interpolation=torchvision.transforms.InterpolationMode.LANCZOS)
            img = torchvision.transforms.functional.pil_to_tensor(img) / 255.0

            # Handle transparency
            if img.shape[0] == 4:  # RGBA
                fg_masks.append(img[3:, ...])
                imgs.append(img[:3, ...] * img[3:, ...] + bg_color * (1 - img[3:, ...]))
            else:  # RGB
                # Create a mask for non-black pixels (assuming black background)
                mask = (img.sum(dim=0, keepdim=True) > 0.1).float()
                fg_masks.append(mask)
                imgs.append(img)

            # Load camera matrix (.npy files store world-to-camera matrix)
            cam_path = os.path.join(obj_path, f"{view_idx:03d}.npy")
            w2c_matrix = torch.tensor(np.load(cam_path)).float()  # 3x4

            # Convert to 4x4 homogeneous matrix
            w2c_cmo = torch.cat([w2c_matrix, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0)

            # Transform from OpenGL to COLMAP convention
            w2c_cmo = torch.matmul(self.opengl_to_colmap, w2c_cmo)

            # Convert to row-major order for rendering
            world_view_transform = w2c_cmo.transpose(0, 1)
            view_world_transform = w2c_cmo.inverse().transpose(0, 1)
            camera_center = view_world_transform[3, :3].clone()

            world_view_transforms.append(world_view_transform)
            view_world_transforms.append(view_world_transform)
            camera_centers.append(camera_center)

        imgs = torch.stack(imgs)
        fg_masks = torch.stack(fg_masks)
        world_view_transforms = torch.stack(world_view_transforms)
        view_world_transforms = torch.stack(view_world_transforms)
        camera_centers = torch.stack(camera_centers)

        # Set focal length for 35mm FOV and 512x512 resolution
        focal_length = fov2focal(35.0, self.cfg.data.training_resolution)
        focals_pixels = torch.full((imgs.shape[0], 2), fill_value=focal_length)
        pps_pixels = torch.full((imgs.shape[0], 2), fill_value=self.cfg.data.training_resolution / 2.0)

        # Normalize camera distance to 2.0 (standard for the training)
        if torch.norm(camera_centers[0]) > 1e-5:
            translation_scaling_factor = 2.0 / torch.norm(camera_centers[0])
            world_view_transforms[:, 3, :3] *= translation_scaling_factor
            view_world_transforms[:, 3, :3] *= translation_scaling_factor
            camera_centers *= translation_scaling_factor

        full_proj_transforms = world_view_transforms.bmm(self.projection_matrix.unsqueeze(0).expand(
            world_view_transforms.shape[0], 4, 4))

        return {"gt_images": imgs,
                "world_view_transforms": world_view_transforms,
                "view_to_world_transforms": view_world_transforms,
                "full_proj_transforms": full_proj_transforms,
                "camera_centers": camera_centers,
                "focals_pixels": focals_pixels,
                "pps_pixels": pps_pixels,
                "fg_masks": fg_masks}

    def get_example_id(self, index):
        return self.paths[index]

    def __getitem__(self, index):
        obj_path = os.path.join(self.root_dir, self.paths[index])

        if self.dataset_name == "vis":
            # For visualization, load all 6 views
            num_views = 6
        else:
            if self.dataset_name == "train":
                num_views = self.imgs_per_obj_train
            else:
                num_views = 6  # Use all views for validation

        images_and_camera_poses = self.load_imgs_and_convert_cameras(obj_path, num_views)
        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses