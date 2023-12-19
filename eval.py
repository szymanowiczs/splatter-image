import json
import os
import sys
import tqdm
from omegaconf import OmegaConf

import lpips as lpips_lib

import torch
import torchvision
from torch.utils.data import DataLoader

from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from scene.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn

class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        return psnr, ssim, lpips

@torch.no_grad()
def evaluate_dataset(model, dataloader, device, model_cfg, save_vis=0, out_folder=None
                     ):
    """
    Runs evaluation on the dataset passed in the dataloader. 
    Computes, prints and saves PSNR, SSIM, LPIPS.
    Args:
        save_vis: how many examples will have visualisations saved
    """

    if save_vis > 0:

        os.makedirs(out_folder, exist_ok=True)

    with open("scores.txt", "w+") as f:
        f.write("")

    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # instantiate metricator
    metricator = Metricator(device)

    psnr_all_examples_novel = []
    ssim_all_examples_novel = []
    lpips_all_examples_novel = []

    psnr_all_examples_cond = []
    ssim_all_examples_cond = []
    lpips_all_examples_cond = []

    for d_idx, data in enumerate(tqdm.tqdm(dataloader)):
        psnr_all_renders_novel = []
        ssim_all_renders_novel = []
        lpips_all_renders_novel = []
        psnr_all_renders_cond = []
        ssim_all_renders_cond = []
        lpips_all_renders_cond = []

        data = {k: v.to(device) for k, v in data.items()}

        rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]

        if model_cfg.data.category == "hydrants" or model_cfg.data.category == "teddybears":
            focals_pixels_pred = data["focals_pixels"][:, :model_cfg.data.input_images, ...]
        else:
            focals_pixels_pred = None

        if model_cfg.data.origin_distances:
            input_images = torch.cat([data["gt_images"][:, :model_cfg.data.input_images, ...],
                                      data["origin_distances"][:, :model_cfg.data.input_images, ...]],
                                      dim=2)
        else:
            input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]

        example_id = dataloader.dataset.get_example_id(d_idx)
        if d_idx < save_vis:

            out_example_gt = os.path.join(out_folder, "{}_".format(d_idx) + example_id + "_gt")
            out_example = os.path.join(out_folder, "{}_".format(d_idx) + example_id)

            os.makedirs(out_example_gt, exist_ok=True)
            os.makedirs(out_example, exist_ok=True)

        # batch has length 1, the first image is conditioning
        reconstruction = model(input_images,
                               data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                               rot_transform_quats,
                               focals_pixels_pred)

        for r_idx in range( data["gt_images"].shape[1]):
            if "focals_pixels" in data.keys():
                focals_pixels_render = data["focals_pixels"][0, r_idx]
            else:
                focals_pixels_render = None
            image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
                                     data["world_view_transforms"][0, r_idx],
                                     data["full_proj_transforms"][0, r_idx], 
                                     data["camera_centers"][0, r_idx],
                                     background,
                                     model_cfg,
                                     focals_pixels=focals_pixels_render)["render"]

            if d_idx < save_vis:
                # vis_image_preds(reconstruction, out_example)
                torchvision.utils.save_image(image, os.path.join(out_example, '{0:05d}'.format(r_idx) + ".png"))
                torchvision.utils.save_image(data["gt_images"][0, r_idx, ...], os.path.join(out_example_gt, '{0:05d}'.format(r_idx) + ".png"))

            # exclude non-foreground images from metric computation
            if not torch.all(data["gt_images"][0, r_idx, ...] == 0):
                psnr, ssim, lpips = metricator.compute_metrics(image, data["gt_images"][0, r_idx, ...])
                if r_idx < model_cfg.data.input_images:
                    psnr_all_renders_cond.append(psnr)
                    ssim_all_renders_cond.append(ssim)
                    lpips_all_renders_cond.append(lpips)
                else:
                    psnr_all_renders_novel.append(psnr)
                    ssim_all_renders_novel.append(ssim)
                    lpips_all_renders_novel.append(lpips)

        psnr_all_examples_cond.append(sum(psnr_all_renders_cond) / len(psnr_all_renders_cond))
        ssim_all_examples_cond.append(sum(ssim_all_renders_cond) / len(ssim_all_renders_cond))
        lpips_all_examples_cond.append(sum(lpips_all_renders_cond) / len(lpips_all_renders_cond))

        psnr_all_examples_novel.append(sum(psnr_all_renders_novel) / len(psnr_all_renders_novel))
        ssim_all_examples_novel.append(sum(ssim_all_renders_novel) / len(ssim_all_renders_novel))
        lpips_all_examples_novel.append(sum(lpips_all_renders_novel) / len(lpips_all_renders_novel))

        with open("scores.txt", "a+") as f:
            f.write("{}_".format(d_idx) + example_id + \
                    " " + str(psnr_all_examples_novel[-1]) + \
                    " " + str(ssim_all_examples_novel[-1]) + \
                    " " + str(lpips_all_examples_novel[-1]) + "\n")

    scores = {"PSNR_cond": sum(psnr_all_examples_cond) / len(psnr_all_examples_cond),
              "SSIM_cond": sum(ssim_all_examples_cond) / len(ssim_all_examples_cond),
              "LPIPS_cond": sum(lpips_all_examples_cond) / len(lpips_all_examples_cond),
              "PSNR_novel": sum(psnr_all_examples_novel) / len(psnr_all_examples_novel),
              "SSIM_novel": sum(ssim_all_examples_novel) / len(ssim_all_examples_novel),
              "LPIPS_novel": sum(lpips_all_examples_novel) / len(lpips_all_examples_novel)}

    return scores

@torch.no_grad()
def main(experiment_path, device_idx, split='val', save_vis=0):
    
    # set device and random seed
    device = torch.device("cuda:{}".format(device_idx))
    torch.cuda.set_device(device)

    # load cfg
    training_cfg = OmegaConf.load(os.path.join(experiment_path, ".hydra", "config.yaml"))

    # load model
    model = GaussianSplatPredictor(training_cfg)
    ckpt_loaded = torch.load(os.path.join(experiment_path, "model_latest.pth"), map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model = model.to(device)
    model.eval()
    print('Loaded model!')

    # instantiate dataset loader
    dataset = get_dataset(training_cfg, split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            persistent_workers=True, pin_memory=True, num_workers=1)
    
    scores = evaluate_dataset(model, dataloader, device, training_cfg, save_vis=save_vis)
    print(scores)
    return scores

if __name__ == "__main__":

    experiment_path = sys.argv[1]
    split = 'test' 
    out_folder = 'out'
    scores = main(experiment_path, 0, split=split, save_vis=0, out_folder=out_folder)
    with open(os.path.join(experiment_path, 
                           "{}_scores.json".format(split)), "w+") as f:
        json.dump(scores, f, indent=4)