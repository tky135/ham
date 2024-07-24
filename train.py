import sys
sys.path.append('../arctic/')
import torch.utils
import torch.utils.data
import common.tb_utils as tb_utils
from datasets.handy_light_dataset import HandyLightDataset
from models.wrapper import HAMERWrapper
from common.tb_utils import push_images
from common.torch_utils import reset_all_seeds
from hamer.configs import dataset_config
from easydict import EasyDict
import torch
import torchvision
import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from utils import load_tars_as_webdataset

# args setup from aditya's coda
args = {'method': 'hamer_light', 'exp_key': '0d7f459de', 'extraction_mode': '', 'img_feat_version': '', 'window_size': 11, 'eval': False, 'debug': False, 'agent_id': 0, 'load_from': '', 'load_ckpt': '', 'infer_ckpt': '', 'resume_ckpt': '', 'fast_dev_run': False, 'trainsplit': 'train', 'valsplit': 'minival', 'run_on': '', 'setup': 'p2', 'log_every': 50, 'eval_every_epoch': 1, 'lr_dec_epoch': [], 'num_epoch': 100, 'lr': 1e-05, 'lr_dec_factor': 10, 'lr_decay': 0.1, 'num_exp': 1, 'acc_grad': 1, 'batch_size': 2, 'test_batch_size': 16, 'num_workers': 0, 'eval_on': '', 'mute': False, 'no_vis': False, 'cluster': False, 'cluster_node': '', 'bid': 21, 'temp_loader': False, 'gpu_ids': [0], 'gpu_arch': 'ampere', 'gpu_min_mem': 20000, 'n_freq_pos_enc': 4, 'use_gt_bbox': True, 'separate_hands': False, 'pos_enc': 'center+corner_latent', 'img_res': 224, 'img_res_ds': 224, 'logger': 'tensorboard', 'backbone': 'resnet50', 'vis_every': 100, 'regress_center_corner': False, 'flip_prob': 0.0, 'bbox_scale': 2.5, 'pretrained': 'hamer', 'val_dataset': 'egoexo', 'tf_decoder': False, 'use_glb_feat': True, 'use_grasp_loss': True, 'use_glb_feat_w_grasp': False, 'use_render_seg_loss': False, 'use_gt_hand_mask': False, 'use_depth_loss': False, 'no_crops': False, 'no_intrx': False, 'focal_length': 1000.0, 'rot_factor': 30.0, 'noise_factor': 0.4, 'scale_factor': 0.25, 'img_norm_mean': [0.485, 0.456, 0.406], 'img_norm_std': [0.229, 0.224, 0.225], 'pin_memory': True, 'shuffle_train': True, 'seed': 1, 'grad_clip': 150.0, 'use_gt_k': False, 'speedup': True, 'max_dist': 0.1, 'ego_image_scale': 0.3, 'project': 'arctic', 'interface_p': None, 'ckpt_p': '', 'log_dir': './logs/0d7f459de', 'args_p': './logs/0d7f459de/args.json', 'gpu': 'A40', 'experiment': None}
# args = {'window_size': 11, 'img_res': 224, 'img_res_ds': 224, 'use_gt_bbox': True, 'use_obj_bbox': True, 'flip_prob': 0.0, 'focal_length': 1000.0, 'rot_factor': 30.0, 'noise_factor': 0.4, 'scale_factor': 0.25, 'use_gt_k': False, 
#             'setup': 'p2', 'debug': False, 'ego_image_scale': 0.3, # for arctic
#             'regress_center_corner': False, 'eval_every_epoch': 1, 'grad_clip': 150.0,# for hamer
#             'img_norm_mean': [0.485, 0.456, 0.406], 'img_norm_std': [0.229, 0.224, 0.225],  # img mean and norm
#             'pos_enc': 'center+corner_latent', 'n_freq_pos_enc':4, 'speedup':True}    # kpe encoding
args = EasyDict(args)
args.method = "hamer_light"
args.setup = "p2"



# cfg setup from hamer
cfg = {'task_name': 'train', 'tags': ['dev'], 'train': True, 'test': False, 'ckpt_path': None, 'seed': None, 'DATASETS': {'SUPPRESS_KP_CONF_THRESH': 0.3, 'FILTER_NUM_KP': 4, 'FILTER_NUM_KP_THRESH': 0.0, 'FILTER_REPROJ_THRESH': 31000, 'SUPPRESS_BETAS_THRESH': 3.0, 'SUPPRESS_BAD_POSES': False, 'POSES_BETAS_SIMULTANEOUS': True, 'FILTER_NO_POSES': False, 'TRAIN': {'INTERHAND26M-TRAIN': {'WEIGHT': 0.25}, 'RHD-TRAIN': {'WEIGHT': 0.05}}, 'VAL': {'FREIHAND-TRAIN': {'WEIGHT': 1.0}, 'MTC-TRAIN': {'WEIGHT': 1.0}}, 'MOCAP': 'FREIHAND-MOCAP', 'BETAS_REG': True, 'CONFIG': {'SCALE_FACTOR': 0.3, 'ROT_FACTOR': 30, 'TRANS_FACTOR': 0.02, 'COLOR_SCALE': 0.2, 'ROT_AUG_RATE': 0.6, 'TRANS_AUG_RATE': 0.5, 'DO_FLIP': False, 'FLIP_AUG_RATE': 0.0, 'EXTREME_CROP_AUG_RATE': 0.0, 'EXTREME_CROP_AUG_LEVEL': 1}}, 'trainer': {'_target_': 'pytorch_lightning.Trainer', 'default_root_dir': '${paths.output_dir}', 'accelerator': 'gpu', 'devices': 2, 'deterministic': False, 'num_sanity_val_steps': 0, 'log_every_n_steps': '${GENERAL.LOG_STEPS}', 'val_check_interval': '${GENERAL.VAL_STEPS}', 'precision': 16, 'max_steps': '${GENERAL.TOTAL_STEPS}', 'limit_val_batches': 1, 'strategy': 'ddp_find_unused_parameters_true', 'num_nodes': 1, 'sync_batchnorm': True}, 'paths': {'root_dir': '${oc.env:PROJECT_ROOT}', 'data_dir': '/data01/adityap9/datasets/hamer', 'log_dir': './logs', 'output_dir': '${hydra:runtime.output_dir}', 'work_dir': '${hydra:runtime.cwd}'}, 'extras': {'ignore_warnings': False, 'enforce_tags': True, 'print_config': False}, 'exp_name': 'hamer', 'MANO': {'DATA_DIR': '/data01/adityap9/datasets/hamer/_DATA/data/', 'MODEL_PATH': '${MANO.DATA_DIR}/mano', 'GENDER': 'neutral', 'NUM_HAND_JOINTS': 15, 'MEAN_PARAMS': '${MANO.DATA_DIR}/mano_mean_params.npz', 'CREATE_BODY_POSE': False}, 'EXTRA': {'FOCAL_LENGTH': 5000, 'NUM_LOG_IMAGES': 4, 'NUM_LOG_SAMPLES_PER_IMAGE': 8, 'PELVIS_IND': 0}, 'GENERAL': {'TOTAL_STEPS': 1000000, 'LOG_STEPS': 1000, 'VAL_STEPS': 1000, 'CHECKPOINT_STEPS': 1000, 'CHECKPOINT_SAVE_TOP_K': 1, 'NUM_WORKERS': 0, 'PREFETCH_FACTOR': 2}, 'TRAIN': {'LR': 1e-05, 'WEIGHT_DECAY': 0.0001, 'BATCH_SIZE': 8, 'LOSS_REDUCTION': 'mean', 'NUM_TRAIN_SAMPLES': 2, 'NUM_TEST_SAMPLES': 64, 'POSE_2D_NOISE_RATIO': 0.01, 'SMPL_PARAM_NOISE_RATIO': 0.005}, 'MODEL': {'IMAGE_SIZE': 256, 'IMAGE_MEAN': [0.485, 0.456, 0.406], 'IMAGE_STD': [0.229, 0.224, 0.225], 'BACKBONE': {'TYPE': 'vit', 'PRETRAINED_WEIGHTS': '/data01/adityap9/datasets/hamer/hamer_training_data/vitpose_backbone.pth'}, 'MANO_HEAD': {'TYPE': 'transformer_decoder', 'IN_CHANNELS': 2048, 'TRANSFORMER_DECODER': {'depth': 6, 'heads': 8, 'mlp_dim': 1024, 'dim_head': 64, 'dropout': 0.0, 'emb_dropout': 0.0, 'norm': 'layer', 'context_dim': 1280}}, 'POS_ENC': {'TYPE': 'center+corner_latent', 'N_FREQ': 4, 'DEC': False}, 'USE_GT_F': True, 'USE_GT_P': True, 'USE_GLB_FEAT': False}, 'LOSS_WEIGHTS': {'KEYPOINTS_3D': 0.05, 'KEYPOINTS_2D': 0.01, 'GLOBAL_ORIENT': 0.001, 'HAND_POSE': 0.001, 'BETAS': 0.0005, 'ADVERSARIAL': 0.0005, 'KEYPOINTS_3D_ABS': 0.0, 'CAM_T': 0.05}}
cfg = EasyDict(cfg) # this will do the job recursively
torch.set_float32_matmul_precision('medium')    # TODO: what is this

### Load hamer dataset
DATASET = "FREIHAND-TRAIN"
dataset_cfg = dataset_config() # from hamer/configs
url = dataset_cfg.get(DATASET).get("URLS")
iter_dataset = load_tars_as_webdataset(cfg, url, train=True, resampled=False, epoch_size=dataset_cfg.get(DATASET).get("epoch_size"))  # train only decide whether to shuffle and augment
hamer_loader = torch.utils.data.DataLoader(iter_dataset, batch_size=2, num_workers=0, pin_memory=True, shuffle=False)

def main():
    # preperation
    tb_utils.log_exp_meta(args)
    reset_all_seeds(1)
    
    # model
    model = HAMERWrapper(args, push_images_fn=push_images)
    model.set_training_flags()
    # dataset
    dataset = HandyLightDataset(args, split='train')
    
    valdataset = HandyLightDataset(args, split='minival')
    valloader = torch.utils.data.DataLoader(
            dataset=valdataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=False,
        )
    
    
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
    
    trainer.fit(model, hamer_loader, hamer_loader)

    
if __name__ == "__main__":
    main()