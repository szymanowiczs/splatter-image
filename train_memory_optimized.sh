#!/bin/bash

cd /root/projects/MultiSplat/splatter-image

echo "Starting Zero123Plus training with memory optimization..."

# 内存优化配置：降低分辨率、批次大小、注意力分辨率
python train_network.py \
  dataset=zero123plus \
  data.training_resolution=256 \
  data.input_images=6 \
  opt.iterations=50000 \
  opt.batch_size=1 \
  opt.imgs_per_obj=6 \
  model.attention_resolutions=[64] \
  model.base_dim=96 \
  logging.ckpt_iterations=10000 \
  logging.val_log=10000 \
  logging.loss_log=100 \
  logging.loop_log=5000 \
  logging.render_log=5000 \
  general.mixed_precision=true