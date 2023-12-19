from omegaconf import OmegaConf
import os
import sys

import torch
from torch.utils.data import DataLoader

from scene.gaussian_predictor import GaussianSplatPredictor
from scene.srn import SRNDataset
from utils.sh_utils import eval_sh

@torch.no_grad()
def test_sh_transform(experiment_path, device):

    # ================== Get data ==================
    device = torch.device("cuda:{}".format(device))
    torch.cuda.set_device(device)
    training_cfg = OmegaConf.load(os.path.join(experiment_path, ".hydra", "config.yaml"))

    # load model
    model = GaussianSplatPredictor(training_cfg)
    ckpt_loaded = torch.load(os.path.join(experiment_path, "model_best.pth"), map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"], strict=True)
    model = model.to(device)
    # model.load_state_dict(checkpoint)
    model.eval()
    print('Loaded model!')

    dataset = SRNDataset(training_cfg, dataset_name='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            persistent_workers=True, pin_memory=True, num_workers=4)
    data_batch = next(iter(dataloader))
    data_batch = {k: v.to(device) for k, v in data_batch.items()}

    # ============= Get random SH predictions and ray dirs =============
    shs = torch.randn((10, 1000, 3, (1 + 1) ** 2), 
                      dtype=torch.float32, device="cuda")

    shs_rest = shs[:, :, :, :3]
    ray_dirs = torch.randn((10, 1000, 3), 
                      dtype=torch.float32, device="cuda")
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
    # compute colour values evaluated from random ray dirs
    cs_vanilla = eval_sh(1, torch.cat([shs[:, :, :, :1], 
                                       shs_rest], dim=-1), 
                         ray_dirs)
    
    # transform SH and random ray dirs with an arbitrary camera transform
    shs_rest_transformed = model.transform_SHs(shs_rest.transpose(2, 3),
                                               data_batch["view_to_world_transforms"][0, :10, :3, :3])
    ray_dirs_transformed = torch.bmm(ray_dirs, data_batch["view_to_world_transforms"][0, :10, :3, :3])

    # compute resulting colour values
    cs_transformed = eval_sh(1, torch.cat([shs[:, :, :, :1],
                                           shs_rest_transformed.transpose(2, 3)], dim=-1),
                             ray_dirs_transformed)
    
    assert torch.mean((cs_transformed - cs_vanilla).abs()) < 1e-6

if __name__ == "__main__":
    test_sh_transform(sys.argv[1], sys.argv[2])