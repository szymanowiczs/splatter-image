"""
Save a couple images to grids with cond, render cond, novel render, novel gt
Also save images to a render video
"""
import glob
import os
from PIL import Image
import numpy as np
import torch

from matplotlib import pyplot as plt
from utils.sh_utils import eval_sh
from einops import rearrange

def gridify():

    out_folder = "grids_objaverse"
    os.makedirs(out_folder, exist_ok=True)

    folder_paths = glob.glob("/scratch/shared/beegfs/stan/scaling_splatter_image/objaverse/*")
    # pixelnerf_root = "/scratch/shared/beegfs/stan/splatter_image/pixelnerf/teddybears"
    folder_paths_test = sorted([fpath for fpath in folder_paths if "gt" not in fpath], key= lambda x: int(os.path.basename(x).split("_")[0]))
    """folder_paths_test = [folder_paths_test[i] for i in [5, 7, 12, 15,
                                                        18, 19, 30, 33,
                                                        37, 42, 43, 44,
                                                        48, 51, 64, 66,
                                                        70, 74, 78, 85, 
                                                        89, 91, 92]]"""

    # Initialize variables for grid dimensions
    num_examples_row = 6
    rows = num_examples_row
    num_per_ex = 2
    cols = num_examples_row * num_per_ex # 7 * 2
    im_res = 128

    for im_idx in range(100):
        print("Doing frame {}".format(im_idx))
        # for im_name in ["xyz", "colours", "opacity", "scaling"]:
        grid = np.zeros((rows*im_res, cols*im_res, 3), dtype=np.uint8)

        # Iterate through the folders in the out_folder
        for f_idx, folder_path_test in enumerate(folder_paths_test[:num_examples_row*num_examples_row]):
            # if im_name == "xyz":
            #     print(folder_path_test)
            row_idx = f_idx // num_examples_row
            col_idx = f_idx % num_examples_row
            im_path = os.path.join(folder_path_test, "{:05d}.png".format(im_idx))
            im_path_gt = os.path.join(folder_path_test + "_gt", "{:05d}.png".format(im_idx))
            """im_path_pixelnerf = os.path.join(pixelnerf_root, os.path.basename(folder_path_test),
                                             "{:06d}.png".format(im_idx))"""

            # im_path = os.path.join(folder_path_test, "{}.png".format(im_name))
            try:
                im = np.array(Image.open(im_path))
                im_gt = np.array(Image.open(im_path_gt))
                #im_pn = np.array(Image.open(im_path_pixelnerf))
                grid[row_idx*im_res: (row_idx+1)*im_res,
                 col_idx * num_per_ex *im_res: (col_idx * num_per_ex+1)*im_res, : ] = im[:, :, :3]
                grid[row_idx*im_res: (row_idx+1)*im_res,
                 (col_idx * num_per_ex + 1) *im_res: (col_idx* num_per_ex +2)*im_res, : ] = im_gt[:, :, :3]
                """grid[row_idx*im_res: (row_idx+1)*im_res,
                 (col_idx * num_per_ex + 2) *im_res: (col_idx* num_per_ex +3)*im_res, : ] = im_pn[:, :, :3]"""
            except FileNotFoundError:
                a = 0
        im_out = Image.fromarray(grid)
        im_out.save(os.path.join(out_folder, "{:05d}.png".format(im_idx)))
        # im_out.save(os.path.join(out_folder, "{}.png".format(im_name)))

def comparisons():

    out_root = "hydrants_comparisons"
    os.makedirs(out_root, exist_ok=True)

    folder_paths = glob.glob("/users/stan/pixel-nerf/full_eval_hydrant/*")
    folder_paths_test = sorted(folder_paths)
    folder_paths_ours_root = "/scratch/shared/beegfs/stan/out_hydrants_with_lpips_ours"

    # Initialize variables for grid dimensions
    rows = 3
    cols = 1
    im_res = 128

    for f_idx, folder_path_test in enumerate(folder_paths_test):

        example_id = "_".join(os.path.basename(folder_path_test).split("_")[1:])
        out_folder = os.path.join(out_root, example_id)
        os.makedirs(out_folder, exist_ok=True)
        num_images = len([p for p in glob.glob(os.path.join(folder_path_test, "*.png")) if "gt" not in p])

        grid = np.zeros((rows*im_res, cols*im_res, 3), dtype=np.uint8)

        for im_idx in range(num_images):

            im_path_pixelnerf = os.path.join(folder_path_test, "{:06d}.png".format(im_idx+1))
            im_path_ours = os.path.join(folder_paths_ours_root, example_id, "{:05d}.png".format(im_idx))
            im_path_gt = os.path.join(folder_paths_ours_root, example_id + "_gt", "{:05d}.png".format(im_idx))
            # im_path = os.path.join(folder_path_test, "{}.png".format(im_name))

            im_pn = np.array(Image.open(im_path_pixelnerf))
            im_ours = np.array(Image.open(im_path_ours))
            im_gt = np.array(Image.open(im_path_gt))

            grid[:im_res, :, :] = im_pn
            grid[im_res:2*im_res, :, :] = im_ours
            grid[2*im_res:3*im_res, :, :] = im_gt

            im_out = Image.fromarray(grid)
            im_out.save(os.path.join(out_folder, "{:05d}.png".format(im_idx)))

def vis_image_preds(image_preds: dict, folder_out: str):
    """
    Visualises network's image predictions.
    Args:
        image_preds: a dictionary of xyz, opacity, scaling, rotation, features_dc and features_rest
    """
    image_preds_reshaped = {}
    ray_dirs = (image_preds["xyz"].detach().cpu() / torch.norm(image_preds["xyz"].detach().cpu(), dim=-1, keepdim=True)).reshape(128, 128, 3)

    for k, v in image_preds.items():
        image_preds_reshaped[k] = v
        if k == "xyz":
            image_preds_reshaped[k] = (image_preds_reshaped[k] - torch.min(image_preds_reshaped[k], dim=0, keepdim=True)[0]) / (
                torch.max(image_preds_reshaped[k], dim=0, keepdim=True)[0] - torch.min(image_preds_reshaped[k], dim=0, keepdim=True)[0]
            )
        if k == "scaling":
            image_preds_reshaped["scaling"] = (image_preds_reshaped["scaling"] - torch.min(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0]) / (
                torch.max(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0] - torch.min(image_preds_reshaped["scaling"], dim=0, keepdim=True)[0]
            )
        if k != "features_rest":
            image_preds_reshaped[k] = image_preds_reshaped[k].reshape(128, 128, -1).detach().cpu()
        else:
            image_preds_reshaped[k] = image_preds_reshaped[k].reshape(128, 128, 3, 3).detach().cpu().permute(0, 1, 3, 2)
        if k == "opacity":
            image_preds_reshaped[k] = image_preds_reshaped[k].expand(128, 128, 3) 


    colours = torch.cat([image_preds_reshaped["features_dc"].unsqueeze(-1), image_preds_reshaped["features_rest"]], dim=-1)
    colours = eval_sh(1, colours, ray_dirs)

    plt.imsave(os.path.join(folder_out, "colours.png"),
               colours.numpy())
    plt.imsave(os.path.join(folder_out, "opacity.png"),
               image_preds_reshaped["opacity"].numpy())
    plt.imsave(os.path.join(folder_out, "xyz.png"), 
               (image_preds_reshaped["xyz"] * image_preds_reshaped["opacity"]+ 1 - image_preds_reshaped["opacity"]).numpy())
    plt.imsave(os.path.join(folder_out, "scaling.png"), 
               (image_preds_reshaped["scaling"] * image_preds_reshaped["opacity"] + 1 - image_preds_reshaped["opacity"]).numpy())

if __name__ == "__main__":
    gridify()