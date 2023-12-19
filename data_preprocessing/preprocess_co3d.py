import torch
import torchvision
import numpy as np

import math
import os
import tqdm
from PIL import Image
from omegaconf import DictConfig

from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.tools.config import expand_args_fields

from .co3d_normalisation import normalize_sequence

CO3D_RAW_ROOT = None # change to where your CO3D data resides
CO3D_OUT_ROOT = None # change to your folder here

assert CO3D_RAW_ROOT is not None, "Change CO3D_RAW_ROOT to where your raw CO3D data resides"
assert CO3D_OUT_ROOT is not None, "Change CO3D_OUT_ROOT to where you want to save the processed CO3D data"

def update_scores(top_scores, top_names, new_score, new_name):
    for sc_idx, sc in enumerate(top_scores):
        if new_score > sc:
            # shift scores and names to the right, start from the end
            for sc_idx_next in range(len(top_scores)-1, sc_idx, -1):
                top_scores[sc_idx_next] = top_scores[sc_idx_next - 1]
                top_names[sc_idx_next] = top_names[sc_idx_next - 1]
            top_scores[sc_idx] = new_score
            top_names[sc_idx] = new_name
            break
    return top_scores, top_names

def main(dataset_name, category):

    subset_name = "fewview_dev"

    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_root=CO3D_RAW_ROOT,
        dataset_JsonIndexDataset_args=DictConfig(
            {"remove_empty_masks": False, "load_point_clouds": True}
        ),
    ).get_dataset_map()

    created_dataset = dataset_map[dataset_name]

    sequence_names = [k for k in created_dataset.seq_annots.keys()]

    bkgd = 0.0 # black background

    out_folder_path = os.path.join(CO3D_OUT_ROOT, "co3d_{}_for_gs".format(category), 
                                   dataset_name)
    os.makedirs(out_folder_path, exist_ok=True)

    bad_sequences = []
    camera_Rs_all_sequences = {}
    camera_Ts_all_sequences = {}

    for sequence_name in tqdm.tqdm(sequence_names):

        folder_outname = os.path.join(out_folder_path, sequence_name)

        frame_idx_gen = created_dataset.sequence_indices_in_order(sequence_name)
        frame_idxs = []
        focal_lengths_this_sequence = []
        rgb_full_this_sequence = []
        rgb_fg_this_sequence = []
        fname_order = []

        # Preprocess cameras with Viewset Diffusion protocol
        normalized_cameras, _, _, _, _ = normalize_sequence(created_dataset, sequence_name, 1.2)
            
        camera_Rs_all_sequences[sequence_name] = normalized_cameras.R
        camera_Ts_all_sequences[sequence_name] = normalized_cameras.T

        while True:
            try:
                frame_idx = next(frame_idx_gen)
                frame_idxs.append(frame_idx)
            except StopIteration:
                break
        
        # Preprocess images
        for frame_idx in frame_idxs:
            # Read the original uncropped image
            frame = created_dataset[frame_idx]
            rgb_image = torchvision.transforms.functional.pil_to_tensor(
                Image.open(frame.image_path)).float() / 255.0
            # ============= Foreground mask =================
            # Initialise the foreground mask at the original resolution
            fg_probability = torch.zeros_like(rgb_image)[:1, ...]
            # Find size of the valid region in the 800x800 image (non-padded)
            resized_image_mask_boundary_y = torch.where(frame.mask_crop > 0)[1].max() + 1
            resized_image_mask_boundary_x = torch.where(frame.mask_crop > 0)[2].max() + 1
            # Resize the foreground mask to the original scale
            x0, y0, box_w, box_h = frame.crop_bbox_xywh
            resized_mask = torchvision.transforms.functional.resize(
                frame.fg_probability[:, :resized_image_mask_boundary_y, :resized_image_mask_boundary_x],
                (box_h, box_w),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                )
            # Fill in the foreground mask at the original scale in the correct location based
            # on where it was cropped
            fg_probability[:, y0:y0+box_h, x0:x0+box_w] = resized_mask

            # ============== Crop around principal point ================
            # compute location of principal point in Pytorch3D NDC coordinate system in pixels
            # scaling * 0.5 is due to the NDC min and max range being +- 1
            principal_point_cropped = frame.camera.principal_point * 0.5 * frame.image_rgb.shape[1]
            # compute location of principal point from top left corer, i.e. in image grid coords
            scaling_factor = max(box_h, box_w) / 800
            principal_point_x = (frame.image_rgb.shape[2] * 0.5 - principal_point_cropped[0, 0]) * scaling_factor + x0
            principal_point_y = (frame.image_rgb.shape[1] * 0.5 - principal_point_cropped[0, 1]) * scaling_factor + y0
            # Get the largest center-crop that fits in the foreground
            max_half_side = get_max_box_side(
                frame.image_size_hw, principal_point_x, principal_point_y)
            # After this transformation principal point is at (0, 0)
            rgb = crop_image_at_non_integer_locations(rgb_image, max_half_side, 
                                                      principal_point_x, principal_point_y)          
            fg_probability_cc = crop_image_at_non_integer_locations(fg_probability, max_half_side, 
                                                      principal_point_x, principal_point_y)
            assert frame.image_rgb.shape[1] == frame.image_rgb.shape[2], "Expected square images"
            
            # =============== Resize to 128 and save =======================
            # Resize raw rgb
            pil_rgb = torchvision.transforms.functional.to_pil_image(rgb)
            pil_rgb = torchvision.transforms.functional.resize(pil_rgb,
                                               128,
                                               interpolation=torchvision.transforms.InterpolationMode.LANCZOS)
            rgb = torchvision.transforms.functional.pil_to_tensor(pil_rgb) / 255.0
            # Resize mask
            fg_probability_cc = torchvision.transforms.functional.resize(fg_probability_cc,
                                               128,
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            # Save rgb
            rgb_full_this_sequence.append(rgb[:3, ...])
            # Save masked rgb
            rgb_fg = rgb[:3, ...] * fg_probability_cc + bkgd * (1-fg_probability_cc)
            rgb_fg_this_sequence.append(rgb_fg)

            fname_order.append("{:05d}.png".format(frame_idx))

            # ============== Intrinsics transformation =================
            # Transform focal length according to the crop
            # Focal length is in NDC conversion so we do not need to change it when resizing
            # We should transform focal length to non-cropped image and then back to cropped but
            # the scaling factor of the full non-cropped image cancels out.
            transformed_focal_lengths = frame.camera.focal_length * max(box_h, box_w) / (2 * max_half_side)
            focal_lengths_this_sequence.append(transformed_focal_lengths)

        os.makedirs(folder_outname, exist_ok=True)
        focal_lengths_this_sequence = torch.stack(focal_lengths_this_sequence)

        
        if torch.all(torch.logical_not( torch.stack(rgb_full_this_sequence).isnan() ) ) and \
            torch.all(torch.logical_not( torch.stack(rgb_fg_this_sequence).isnan() ) ) and \
            torch.all(torch.logical_not( focal_lengths_this_sequence.isnan() ) ):
        
            np.save(os.path.join(folder_outname, "images_full.npy"), torch.stack(rgb_full_this_sequence).numpy())
            np.save(os.path.join(folder_outname, "images_fg.npy"), torch.stack(rgb_fg_this_sequence).numpy())
            np.save(os.path.join(folder_outname, "focal_lengths.npy"), focal_lengths_this_sequence.numpy())

            with open(os.path.join(folder_outname, "frame_order.txt"), "w+") as f:
                f.writelines([fname + "\n" for fname in fname_order])
        else:
            print("Warning! bad sequence {}".format(sequence_name))
            bad_sequences.append(sequence_name)


    # convert camera data to numpy archives and save
    for dict_to_save, dict_name in zip([camera_Rs_all_sequences,
                                        camera_Ts_all_sequences],
                                       ["camera_Rs",
                                        "camera_Ts"]):

        np.savez(os.path.join(out_folder_path, dict_name+".npz"),
                 **{k: v.detach().cpu().numpy() for k, v in dict_to_save.items()})

    return bad_sequences

def get_max_box_side(hw, principal_point_x, principal_point_y):
    # assume images are always padded on the right - find where the image ends
    # find the largest center crop we can make
    max_x = hw[1] # x-coord of the rightmost boundary
    min_x = 0.0 # x-coord of the leftmost boundary
    max_y = hw[0] # y-coord of the top boundary
    min_y = 0.0 # y-coord of the bottom boundary

    max_half_w = min(principal_point_x - min_x, max_x - principal_point_x) 
    max_half_h = min(principal_point_y - min_y, max_y - principal_point_y) 
    max_half_side = min(max_half_h, max_half_w)

    return max_half_side

def crop_image_at_non_integer_locations(img, 
                                        max_half_side: float, 
                                        principal_point_x: float, 
                                        principal_point_y: float):
    """
    Crops the image so that its center is at the principal point.
    The boundaries are specified by half of the image side. 
    """
    # number of pixels that the image spans. We don't want to resize
    # at this stage. However, the boundaries might be such that
    # the crop side is not an integer. Therefore there will be
    # minimal resizing, but it's extent will be sub-pixel.
    # We don't apply low-pass filtering at this stage and cropping is
    # done with bilinear sampling 
    max_pixel_number = math.floor(2 * max_half_side)
    half_pixel_side = 0.5 / max_pixel_number
    x_locations = torch.linspace(principal_point_x - max_half_side + half_pixel_side,
                                 principal_point_x + max_half_side - half_pixel_side,
                                 max_pixel_number)
    y_locations = torch.linspace(principal_point_y - max_half_side + half_pixel_side,
                                 principal_point_y + max_half_side - half_pixel_side,
                                 max_pixel_number)
    grid_locations = torch.stack(torch.meshgrid(x_locations, y_locations, indexing='ij'), dim=-1).transpose(0, 1)
    grid_locations[:, :, 1] = ( grid_locations[:, :, 1] - img.shape[1] / 2 ) / ( img.shape[1] / 2 )
    grid_locations[:, :, 0] = ( grid_locations[:, :, 0] - img.shape[2] / 2 ) / ( img.shape[2] / 2 )
    image_crop = torch.nn.functional.grid_sample(img.unsqueeze(0), grid_locations.unsqueeze(0))
    return image_crop.squeeze(0)

if __name__ == "__main__":
    for category in ["teddybear", "hydrant"]:
        for split in ["train", "val", "test"]:
            bad_sequences_val = main(split, category)
