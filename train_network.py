#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import hydra
import os
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader

from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, l2_loss
import lpips as lpips_lib

from eval import evaluate_dataset
from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from scene.dataset_factory import get_dataset


@hydra.main(version_base=None, config_path='configs', config_name="default_config")
def main(cfg: DictConfig):

    vis_dir = os.getcwd()

    dict_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    if os.path.isdir(os.path.join(vis_dir, "wandb")):
        run_name_path = glob.glob(os.path.join(vis_dir, "wandb", "latest-run", "run-*"))[0]
        print("Got run name path {}".format(run_name_path))
        run_id = os.path.basename(run_name_path).split("run-")[1].split(".wandb")[0]
        print("Resuming run with id {}".format(run_id))
        wandb_run = wandb.init(project=cfg.wandb.project, resume=True,
                        id = run_id, config=dict_cfg)

    else:
        wandb_run = wandb.init(project=cfg.wandb.project, reinit=True,
                        config=dict_cfg)

    first_iter = 0
    device = safe_state(cfg)

    gaussian_predictor = GaussianSplatPredictor(cfg)
    gaussian_predictor.to(device)

    l = []
    if cfg.model.network_with_offset:
        l.append({'params': gaussian_predictor.network_with_offset.parameters(), 
         'lr': cfg.opt.base_lr})
    if cfg.model.network_without_offset:
        l.append({'params': gaussian_predictor.network_wo_offset.parameters(), 
         'lr': cfg.opt.base_lr})
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, 
                                 betas=cfg.opt.betas)

    if cfg.opt.step_lr_at != -1:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=cfg.opt.step_lr_at,
                                                    gamma=0.1)

    # Resuming training
    if os.path.isfile(os.path.join(vis_dir, "model_latest.pth")):
        print('Loading an existing model from ', os.path.join(vis_dir, "model_latest.pth"))
        checkpoint = torch.load(os.path.join(vis_dir, "model_latest.pth"),
                                map_location=device) 
        try:
            gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                               strict=False)
            print("Warning, model mismatch - was this expected?")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        first_iter = checkpoint["iteration"]
        best_PSNR = checkpoint["best_PSNR"] 
        print('Loaded model')
    # Resuming from checkpoint
    elif cfg.opt.pretrained_ckpt is not None:
        pretrained_ckpt_dir = os.path.join(cfg.opt.pretrained_ckpt, "model_latest.pth")
        checkpoint = torch.load(pretrained_ckpt_dir,
                                map_location=device) 
        try:
            gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                               strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # first_iter = checkpoint["iteration"]
        best_PSNR = checkpoint["best_PSNR"] 
        print('Loaded model from a pretrained checkpoint')

    if cfg.opt.ema.use:
        ema = EMA(gaussian_predictor, 
                  beta=cfg.opt.ema.beta,
                  update_every=cfg.opt.ema.update_every,
                  update_after_step=cfg.opt.ema.update_after_step)
        ema = ema.to(device)

    if cfg.opt.loss == "l2":
        loss_fn = l2_loss
    elif cfg.opt.loss == "l1":
        loss_fn = l1_loss

    if cfg.opt.lambda_lpips != 0:
        lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
    if cfg.opt.start_lpips_after == 0:
        lambda_lpips = cfg.opt.lambda_lpips
    else:
        lambda_lpips = 0.0
    lambda_l12 = 1.0 - lambda_lpips

    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    dataset = get_dataset(cfg, "train")
    dataloader = DataLoader(dataset, 
                            batch_size=cfg.opt.batch_size,
                            shuffle=True)

    val_dataset = get_dataset(cfg, "val")
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                persistent_workers=True,
                                pin_memory=True)

    test_dataset = get_dataset(cfg, "test")
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1,
                                 shuffle=True)
    gaussian_predictor.train()

    print("Beginning training")
    first_iter += 1
    best_PSNR = 0.0
    dataloader_iterator = iter(dataloader)
    for iteration in range(first_iter, cfg.opt.iterations + 1):        

        if iteration == cfg.opt.start_lpips_after:
            lambda_lpips = cfg.opt.lambda_lpips
            lambda_l12 = 1.0 - lambda_lpips

        try:
            data = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            data = next(dataloader_iterator)        
        data = {k: v.to(device) for k, v in data.items()}

        rot_transform_quats = data["source_cv2wT_quat"][:, :cfg.data.input_images]

        if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
            focals_pixels_pred = data["focals_pixels"][:, :cfg.data.input_images, ...]
            input_images = torch.cat([data["gt_images"][:, :cfg.data.input_images, ...],
                            data["origin_distances"][:, :cfg.data.input_images, ...]],
                            dim=2)
        else:
            focals_pixels_pred = None
            input_images = data["gt_images"][:, :cfg.data.input_images, ...]

        gaussian_splats = gaussian_predictor(input_images,
                                             data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                                             rot_transform_quats,
                                             focals_pixels_pred)


        if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
            # regularize very big gaussians
            if len(torch.where(gaussian_splats["scaling"] > 20)[0]) > 0:
                big_gaussian_reg_loss = torch.mean(
                    gaussian_splats["scaling"][torch.where(gaussian_splats["scaling"] > 20)] * 0.1)
                print('Regularising {} big Gaussians on iteration {}'.format(
                    len(torch.where(gaussian_splats["scaling"] > 20)[0]), iteration))
            else:
                big_gaussian_reg_loss = 0.0
            # regularize very small Gaussians
            if len(torch.where(gaussian_splats["scaling"] < 1e-5)[0]) > 0:
                small_gaussian_reg_loss = torch.mean(
                    -torch.log(gaussian_splats["scaling"][torch.where(gaussian_splats["scaling"] < 1e-5)]) * 0.1)
                print('Regularising {} small Gaussians on iteration {}'.format(
                    len(torch.where(gaussian_splats["scaling"] < 1e-5)[0]), iteration))
            else:
                small_gaussian_reg_loss = 0.0
        # Render
        l12_loss_sum = 0.0
        lpips_loss_sum = 0.0
        rendered_images = []
        gt_images = []
        for b_idx in range(data["gt_images"].shape[0]):
            # image at index 0 is training, remaining images are targets
            # Rendering is done sequentially because gaussian rasterization code
            # does not support batching
            gaussian_splat_batch = {k: v[b_idx].contiguous() for k, v in gaussian_splats.items()}
            for r_idx in range(cfg.data.input_images, data["gt_images"].shape[1]):
                if "focals_pixels" in data.keys():
                    focals_pixels_render = data["focals_pixels"][b_idx, r_idx].cpu()
                else:
                    focals_pixels_render = None
                image = render_predicted(gaussian_splat_batch, 
                                    data["world_view_transforms"][b_idx, r_idx],
                                    data["full_proj_transforms"][b_idx, r_idx],
                                    data["camera_centers"][b_idx, r_idx],
                                    background,
                                    cfg,
                                    focals_pixels=focals_pixels_render)["render"]
                # Put in a list for a later loss computation
                rendered_images.append(image)
                gt_image = data["gt_images"][b_idx, r_idx]
                gt_images.append(gt_image)
        rendered_images = torch.stack(rendered_images, dim=0)
        gt_images = torch.stack(gt_images, dim=0)
        # Loss computation
        l12_loss_sum = loss_fn(rendered_images, gt_images) 
        if cfg.opt.lambda_lpips != 0 and iteration > cfg.opt.start_lpips_after:
            lpips_loss_sum = torch.mean(
                lpips_fn(rendered_images * 2 - 1, gt_images * 2 - 1),
                )

        total_loss = l12_loss_sum * lambda_l12 + lpips_loss_sum * lambda_lpips
        if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
            total_loss = total_loss + big_gaussian_reg_loss + small_gaussian_reg_loss

        total_loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        if cfg.opt.step_lr_at != -1:
            scheduler.step()

        if cfg.opt.ema.use:
            ema.update()

        gaussian_predictor.eval()

        # ========= Logging =============
        with torch.no_grad():
            if iteration % cfg.logging.loss_log == 0:
                wandb.log({"training_loss": np.log10(total_loss.item() + 1e-8)}, step=iteration)
                if cfg.opt.lambda_lpips != 0:
                    wandb.log({"training_l12_loss": np.log10(l12_loss_sum.item() + 1e-8)}, step=iteration)
                    if iteration > cfg.opt.start_lpips_after:
                        wandb.log({"training_lpips_loss": np.log10(lpips_loss_sum.item() + 1e-8)}, step=iteration)
                    else:
                        wandb.log({"training_lpips_loss": np.log10(lpips_loss_sum + 1e-8)}, step=iteration)
                if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                    if type(big_gaussian_reg_loss) == float:
                        brl_for_log = big_gaussian_reg_loss
                    else:
                        brl_for_log = big_gaussian_reg_loss.item()
                    if type(small_gaussian_reg_loss) == float:
                        srl_for_log = small_gaussian_reg_loss
                    else:
                        srl_for_log = small_gaussian_reg_loss.item()
                    wandb.log({"reg_loss_big": np.log10(brl_for_log + 1e-8)}, step=iteration)
                    wandb.log({"reg_loss_small": np.log10(srl_for_log + 1e-8)}, step=iteration)

            if iteration % cfg.logging.render_log == 0 or iteration == 1:
                wandb.log({"render": wandb.Image(image.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy())}, step=iteration)
                wandb.log({"gt": wandb.Image(gt_image.permute(1, 2, 0).detach().cpu().numpy())}, step=iteration)
            if iteration % cfg.logging.loop_log == 0 or iteration == 1:
                # torch.cuda.empty_cache()
                vis_data = next(iter(test_dataloader))
                vis_data = {k: v.to(device) for k, v in vis_data.items()}

                rot_transform_quats = vis_data["source_cv2wT_quat"][:, :cfg.data.input_images]

                if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                    focals_pixels_pred = vis_data["focals_pixels"][:, :cfg.data.input_images, ...]
                    input_images = torch.cat([vis_data["gt_images"][:, :cfg.data.input_images, ...],
                                              vis_data["origin_distances"][:, :cfg.data.input_images, ...]],
                                              dim=2)
                else:
                    focals_pixels_pred = None
                    input_images = vis_data["gt_images"][:, :cfg.data.input_images, ...]

                gaussian_splats_vis = gaussian_predictor(input_images,
                                                    vis_data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                                                    rot_transform_quats,
                                                    focals_pixels_pred)

                test_loop = []
                test_loop_gt = []
                for r_idx in range(vis_data["gt_images"].shape[1]):
                    # We don't change the input or output of the network, just the rendering cameras
                    if "focals_pixels" in vis_data.keys():
                        focals_pixels_render = vis_data["focals_pixels"][0, r_idx]
                    else:
                        focals_pixels_render = None
                    test_image = render_predicted({k: v[0].contiguous() for k, v in gaussian_splats_vis.items()}, 
                                         vis_data["world_view_transforms"][0, r_idx], 
                                         vis_data["full_proj_transforms"][0, r_idx], 
                                         vis_data["camera_centers"][0, r_idx],
                                         background,
                                         cfg,
                                         focals_pixels=focals_pixels_render)["render"]
                    test_loop_gt.append((np.clip(vis_data["gt_images"][0, r_idx].detach().cpu().numpy(), 0, 1)*255).astype(np.uint8))
                    test_loop.append((np.clip(test_image.detach().cpu().numpy(), 0, 1)*255).astype(np.uint8))
    
                wandb.log({"rot": wandb.Video(np.asarray(test_loop), fps=20, format="mp4")},
                    step=iteration)
                wandb.log({"rot_gt": wandb.Video(np.asarray(test_loop_gt), fps=20, format="mp4")},
                    step=iteration)

        fnames_to_save = []
        # Find out which models to save
        if (iteration + 1) % cfg.logging.ckpt_iterations == 0:
            fnames_to_save.append("model_latest.pth")
        if (iteration + 1) % cfg.logging.val_log == 0:
            torch.cuda.empty_cache()
            print("\n[ITER {}] Validating".format(iteration + 1))
            if cfg.opt.ema.use:
                scores = evaluate_dataset(
                    ema, 
                    val_dataloader, 
                    device=device,
                    model_cfg=cfg)
            else:
                scores = evaluate_dataset(
                    gaussian_predictor, 
                    val_dataloader, 
                    device=device,
                    model_cfg=cfg)
            wandb.log(scores, step=iteration+1)
            # save models - if the newest psnr is better than the best one,
            # overwrite best_model. Always overwrite the latest model. 
            if scores["PSNR_novel"] > best_PSNR:
                fnames_to_save.append("model_best.pth")
                best_PSNR = scores["PSNR_novel"]
                print("\n[ITER {}] Saving new best checkpoint PSNR:{:.2f}".format(
                    iteration + 1, best_PSNR))
            torch.cuda.empty_cache()

        # ============ Model saving =================
        for fname_to_save in fnames_to_save:
            ckpt_save_dict = {
                            "iteration": iteration,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": total_loss.item(),
                            "best_PSNR": best_PSNR
                            }
            if cfg.opt.ema.use:
                ckpt_save_dict["model_state_dict"] = ema.ema_model.state_dict()                  
            else:
                ckpt_save_dict["model_state_dict"] = gaussian_predictor.state_dict() 
            torch.save(ckpt_save_dict, os.path.join(vis_dir, fname_to_save))

        gaussian_predictor.train()

    wandb_run.finish()

def training_report(gaussian_splats, data):
    raise NotImplementedError
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()