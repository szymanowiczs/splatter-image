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

OBJAVERSE_ROOT = "/tmp/dummy_objaverse" # Dummy path for zero123plus training
OBJAVERSE_LVIS_ANNOTATION_PATH = "/tmp/dummy_annotation.json" # Dummy path for zero123plus training
# assert OBJAVERSE_ROOT is not None, "Update dataset path"
# assert OBJAVERSE_LVIS_ANNOTATION_PATH is not None, "Update filtering .json path"

class ObjaverseDataset(SharedDataset):
    def __init__(self,
                 cfg,
                 dataset_name = "train"
                 ) -> None:

        super(ObjaverseDataset).__init__()
        self.cfg = cfg
        self.root_dir = OBJAVERSE_ROOT

        # load the file names
        with open(OBJAVERSE_LVIS_ANNOTATION_PATH) as f:
            self.paths = json.load(f)

        # split the dataset for training and validation
        total_objects = len(self.paths)
        self.dataset_name = dataset_name
        if self.dataset_name == "val" or dataset_name == "vis":
            # validation or visualisation on Objaverse
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        elif self.dataset_name == "test":
            raise NotImplementedError # Objaverse does not have separate test subset
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        
        if cfg.data.subset != -1:
            self.paths = self.paths[:cfg.data.subset]

        print('============= length of dataset %d =============' % len(self.paths))

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)

        self.image_side_target = self.cfg.data.training_resolution
        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)

        self.imgs_per_obj_train = self.cfg.opt.imgs_per_obj

    def __len__(self):
        return len(self.paths)
       
    def load_imgs_and_convert_cameras(self, paths, num_views):
        """
        Load the images, camera matrices and projection matrices for a given object 
        """
        bg_color = torch.tensor([1., 1., 1.], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        world_view_transforms = []
        view_world_transforms = []

        camera_centers = []
        imgs = []
        fg_masks = []

        # validation dataset is used for scoring - fix cond frame for reproducibility
        # in trainng need to randomly sample the conditioning frame
        if self.dataset_name != "train":
            indexes = torch.arange(num_views)
        else:
            indexes = torch.randperm(len(paths))[:num_views]
            indexes = torch.cat([indexes[:self.cfg.data.input_images], indexes], dim=0)

        # load the images and cameras
        for i in indexes:
            # read to [0, 1] FloatTensor and resize to training_resolution
            img = Image.open(paths[i])
            # resize to the training resolution
            img = torchvision.transforms.functional.resize(img,
                                                self.cfg.data.training_resolution,
                                                interpolation=torchvision.transforms.InterpolationMode.LANCZOS)
            img = torchvision.transforms.functional.pil_to_tensor(img) / 255.0
            # set background
            fg_masks.append(img[3:, ...])
            imgs.append(img[:3, ...] * img[3:, ...] + bg_color * (1 - img[3:, ...]))

            # .npy files store world-to-camera matrix in column major order
            w2c_cmo = torch.tensor(np.load(paths[i].replace('png', 'npy'))).float() # 3x4
            w2c_cmo = torch.cat([w2c_cmo, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0) # 4x4
            # camera poses in .npy files are in OpenGL convention: 
            #     x right, y up, z into the camera (backward),
            # need to transform to COLMAP / OpenCV:
            #     x right, y down, z away from the camera (forward)
            w2c_cmo = torch.matmul(self.opengl_to_colmap, w2c_cmo)
            # need row major oder
            world_view_transform = w2c_cmo.transpose(0, 1)
            view_world_transform = w2c_cmo.inverse().transpose(0, 1)
            camera_center = view_world_transform[3, :3].clone()

            world_view_transforms.append(world_view_transform)
            view_world_transforms.append(view_world_transform)
            # full_proj_transforms.append(full_proj_transform)
            camera_centers.append(camera_center)

        imgs = torch.stack(imgs)
        fg_masks = torch.stack(fg_masks)
        world_view_transforms = torch.stack(world_view_transforms)
        view_world_transforms = torch.stack(view_world_transforms)
        camera_centers = torch.stack(camera_centers)
        focals_pixels = torch.full((imgs.shape[0], 2),
                                   fill_value=fov2focal(self.cfg.data.fov,
                                                        self.cfg.data.training_resolution))
        pps_pixels = torch.zeros((imgs.shape[0], 2))

        # fix the distance of the source camera to the object / world center
        assert torch.norm(camera_centers[0]) > 1e-5, \
            "Camera is at {} from center".format(torch.norm(camera_centers[0]))
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

    def load_loop(self, paths, num_imgs_in_loop):
        world_view_transforms = []
        view_world_transforms = []
        camera_centers = []
        imgs = []

        gt_imgs_and_cameras = self.load_imgs_and_convert_cameras(paths, len(paths))
        loop_cameras_c2w_cmo = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop)

        for src_idx in range(self.cfg.data.input_images):
            imgs.append(gt_imgs_and_cameras["gt_images"][src_idx])
            camera_centers.append(gt_imgs_and_cameras["camera_centers"][src_idx])
            world_view_transforms.append(gt_imgs_and_cameras["world_view_transforms"][src_idx])
            view_world_transforms.append(gt_imgs_and_cameras["view_to_world_transforms"][src_idx])

        for loop_camera_c2w_cmo in loop_cameras_c2w_cmo:
            view_world_transform = torch.from_numpy(loop_camera_c2w_cmo).transpose(0, 1)
            world_view_transform = torch.from_numpy(loop_camera_c2w_cmo).inverse().transpose(0, 1)
            camera_center = view_world_transform[3, :3].clone()

            camera_centers.append(camera_center)
            world_view_transforms.append(world_view_transform)
            view_world_transforms.append(view_world_transform)

            # use the closest camera as reference gt image
            closest_gt_idx = torch.argmin(torch.norm(
                gt_imgs_and_cameras["camera_centers"] - camera_center.unsqueeze(0), dim=-1)).item()
            imgs.append(gt_imgs_and_cameras["gt_images"][closest_gt_idx])

        imgs = torch.stack(imgs)
        world_view_transforms = torch.stack(world_view_transforms)
        view_world_transforms = torch.stack(view_world_transforms)
        camera_centers = torch.stack(camera_centers)

        full_proj_transforms = world_view_transforms.bmm(self.projection_matrix.unsqueeze(0).expand(
            world_view_transforms.shape[0], 4, 4))

        focals_pixels = torch.full((imgs.shape[0], 2),
                                   fill_value=fov2focal(self.cfg.data.fov,
                                                        self.cfg.data.training_resolution))
        pps_pixels = torch.zeros((imgs.shape[0], 2))

        return {"gt_images": imgs.to(memory_format=torch.channels_last),
                "world_view_transforms": world_view_transforms,
                "view_to_world_transforms": view_world_transforms,
                "full_proj_transforms": full_proj_transforms,
                "camera_centers": camera_centers,
                "focals_pixels": focals_pixels,
                "pps_pixels": pps_pixels}

    def get_example_id(self, index):
        example_id = self.paths[index]
        return example_id

    def __getitem__(self, index):
        # load the rendered images
        filename = os.path.join(self.root_dir, self.paths[index])
        paths = glob.glob(filename + '/*.png')

        if self.dataset_name == "vis":
            images_and_camera_poses = self.load_loop(paths, 200)
        else:
            if self.dataset_name == "train":
                num_views = self.imgs_per_obj_train
            else:
                num_views = len(paths)
            try:
                images_and_camera_poses = self.load_imgs_and_convert_cameras(paths, num_views)
            except:
                print("Found an error with path {}, loading from \
                      8e348d4d2f2949cf88bd896a92a4364d instead").format(self.paths[index])
                filename = os.path.join(self.root_dir, '8e348d4d2f2949cf88bd896a92a4364d')
                paths = glob.glob(filename + '/*.png')
                num_views = len(paths)
                images_and_camera_poses = self.load_imgs_and_convert_cameras(paths, num_views)

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses
