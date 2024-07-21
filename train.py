import sys
sys.path.append('../arctic/')

import torch.utils
import torch.utils.data
import common.tb_utils as tb_utils
from datasets.handy_light_dataset import HandyLightDataset
from models.wrapper import HAMERWrapper
from common.tb_utils import push_images
from common.torch_utils import reset_all_seeds
from easydict import EasyDict
import torch
import torchvision
import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

# args setup
args = {'method': 'hamer_light', 'exp_key': '0d7f459de', 'extraction_mode': '', 'img_feat_version': '', 'window_size': 11, 'eval': False, 'debug': False, 'agent_id': 0, 'load_from': '', 'load_ckpt': '', 'infer_ckpt': '', 'resume_ckpt': '', 'fast_dev_run': False, 'trainsplit': 'minitrain', 'valsplit': 'minival', 'run_on': '', 'setup': 'p2', 'log_every': 50, 'eval_every_epoch': 1, 'lr_dec_epoch': [], 'num_epoch': 100, 'lr': 1e-05, 'lr_dec_factor': 10, 'lr_decay': 0.1, 'num_exp': 1, 'acc_grad': 1, 'batch_size': 2, 'test_batch_size': 16, 'num_workers': 1, 'eval_on': '', 'mute': False, 'no_vis': False, 'cluster': False, 'cluster_node': '', 'bid': 21, 'temp_loader': False, 'gpu_ids': [0], 'gpu_arch': 'ampere', 'gpu_min_mem': 20000, 'n_freq_pos_enc': 4, 'use_gt_bbox': True, 'separate_hands': False, 'pos_enc': 'center+corner_latent', 'img_res': 224, 'img_res_ds': 224, 'logger': 'tensorboard', 'backbone': 'resnet50', 'vis_every': 100, 'regress_center_corner': False, 'flip_prob': 0.0, 'bbox_scale': 2.5, 'pretrained': 'hamer', 'val_dataset': 'egoexo', 'tf_decoder': False, 'use_glb_feat': True, 'use_grasp_loss': True, 'use_glb_feat_w_grasp': False, 'use_render_seg_loss': False, 'use_gt_hand_mask': False, 'use_depth_loss': False, 'no_crops': False, 'no_intrx': False, 'focal_length': 1000.0, 'rot_factor': 30.0, 'noise_factor': 0.4, 'scale_factor': 0.25, 'img_norm_mean': [0.485, 0.456, 0.406], 'img_norm_std': [0.229, 0.224, 0.225], 'pin_memory': True, 'shuffle_train': True, 'seed': 1, 'grad_clip': 150.0, 'use_gt_k': False, 'speedup': True, 'max_dist': 0.1, 'ego_image_scale': 0.3, 'project': 'arctic', 'interface_p': None, 'ckpt_p': '', 'log_dir': './logs/0d7f459de', 'args_p': './logs/0d7f459de/args.json', 'gpu': 'A40', 'experiment': None}
# args = {'window_size': 11, 'img_res': 224, 'img_res_ds': 224, 'use_gt_bbox': True, 'use_obj_bbox': True, 'flip_prob': 0.0, 'focal_length': 1000.0, 'rot_factor': 30.0, 'noise_factor': 0.4, 'scale_factor': 0.25, 'use_gt_k': False, 
#             'setup': 'p2', 'debug': False, 'ego_image_scale': 0.3, # for arctic
#             'regress_center_corner': False, 'eval_every_epoch': 1, 'grad_clip': 150.0,# for hamer
#             'img_norm_mean': [0.485, 0.456, 0.406], 'img_norm_std': [0.229, 0.224, 0.225],  # img mean and norm
#             'pos_enc': 'center+corner_latent', 'n_freq_pos_enc':4, 'speedup':True}    # kpe encoding
args = EasyDict(args)
args.method = "hamer_light"
args.setup = "p2"

torch.set_float32_matmul_precision('medium')    # TODO: what is this

def main():
    # preperation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tb_utils.log_exp_meta(args)
    reset_all_seeds(1)
    
    # model
    model = HAMERWrapper(args, push_images_fn=push_images)
    model.set_training_flags()
    # dataset
    dataset = HandyLightDataset(args, split='minitrain')
    
    
    logging.info(f"Dataset size: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=args.shuffle_train,
        )
    
    # trainer
    ckpt_callback = ModelCheckpoint(
        monitor="loss__val",
        verbose=True,
        save_top_k=3, #og: 5
        mode="min",
        every_n_epochs=1,
        save_last=True,
        dirpath=os.path.join(args.log_dir, "checkpoints"),
    )
    
    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)
    model_summary_cb = ModelSummary(max_depth=3)
    callbacks = [ckpt_callback, pbar_cb, model_summary_cb]
    
    pl_logger = pl.loggers.TensorBoardLogger(args.log_dir)
    trainer = pl.Trainer(
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.acc_grad,
        devices=-1, # og: 1
        accelerator="gpu",
        logger=pl_logger,
        min_epochs=args.num_epoch,
        max_epochs=args.num_epoch,
        callbacks=callbacks,
        log_every_n_steps=args.log_every,
        default_root_dir=args.log_dir,
        check_val_every_n_epoch=args.eval_every_epoch,
        num_sanity_val_steps=(not args.debug),
        enable_model_summary=False,
        strategy='ddp_find_unused_parameters_true', # DDP doesn't work without this
    )
    
    trainer.fit(model, dataloader, dataloader)

    
if __name__ == "__main__":
    main()