import torch.utils
import torch.utils.data
from datasets.handy_light_dataset import HandyLightDataset
from models.wrapper import HAMERWrapper
from arctic.common.tb_utils import push_images
from easydict import EasyDict
import torch
import torchvision

# args setup
# {'method': 'hamer_light', 'exp_key': '0d7f459de', 'extraction_mode': '', 'img_feat_version': '', 'window_size': 11, 'eval': False, 'debug': False, 'agent_id': 0, 'load_from': '', 'load_ckpt': '', 'infer_ckpt': '', 'resume_ckpt': '', 'fast_dev_run': False, 'trainsplit': 'minitrain', 'valsplit': 'minival', 'run_on': '', 'setup': 'p2', 'log_every': 50, 'eval_every_epoch': 1, 'lr_dec_epoch': [], 'num_epoch': 100, 'lr': 1e-05, 'lr_dec_factor': 10, 'lr_decay': 0.1, 'num_exp': 1, 'acc_grad': 1, 'batch_size': 2, 'test_batch_size': 16, 'num_workers': 1, 'eval_on': '', 'mute': False, 'no_vis': False, 'cluster': False, 'cluster_node': '', 'bid': 21, 'temp_loader': False, 'gpu_ids': [0], 'gpu_arch': 'ampere', 'gpu_min_mem': 20000, 'n_freq_pos_enc': 4, 'use_gt_bbox': True, 'separate_hands': False, 'pos_enc': 'center+corner_latent', 'img_res': 224, 'img_res_ds': 224, 'logger': 'tensorboard', 'backbone': 'resnet50', 'vis_every': 100, 'regress_center_corner': False, 'flip_prob': 0.0, 'bbox_scale': 2.5, 'pretrained': 'hamer', 'val_dataset': 'egoexo', 'tf_decoder': False, 'use_glb_feat': True, 'use_grasp_loss': True, 'use_glb_feat_w_grasp': False, 'use_render_seg_loss': False, 'use_gt_hand_mask': False, 'use_depth_loss': False, 'no_crops': False, 'no_intrx': False, 'focal_length': 1000.0, 'rot_factor': 30.0, 'noise_factor': 0.4, 'scale_factor': 0.25, 'img_norm_mean': [0.485, 0.456, 0.406], 'img_norm_std': [0.229, 0.224, 0.225], 'pin_memory': True, 'shuffle_train': True, 'seed': 1, 'grad_clip': 150.0, 'use_gt_k': False, 'speedup': True, 'max_dist': 0.1, 'ego_image_scale': 0.3, 'project': 'arctic', 'interface_p': None, 'ckpt_p': '', 'log_dir': './logs/0d7f459de', 'args_p': './logs/0d7f459de/args.json', 'gpu': 'A40', 'experiment': None}
args = {'window_size': 11, 'img_res': 224, 'img_res_ds': 224, 'use_gt_bbox': True, 'use_obj_bbox': True, 'flip_prob': 0.0, 'focal_length': 1000.0, 'rot_factor': 30.0, 'noise_factor': 0.4, 'scale_factor': 0.25, 'use_gt_k': False, 
            'setup': 'p2', 'debug': False, 'ego_image_scale': 0.3, # for arctic
            'regress_center_corner': False, # for hamer
            'img_norm_mean': [0.485, 0.456, 0.406], 'img_norm_std': [0.229, 0.224, 0.225],  # img mean and norm
            'pos_enc': 'center+corner_latent', 'n_freq_pos_enc':4, 'speedup':True}    # kpe encoding
args = EasyDict(args)
args.experiment = None
args.exp_key = "xxxxxxx"
args.method = "hamer_light"
args.setup = "p2"

# dataset
handy_light_dataset = HandyLightDataset(args, split='train')
dataloader = torch.utils.data.DataLoader(handy_light_dataset, batch_size=1, shuffle=False, num_workers=0)

model = HAMERWrapper(args, push_images_fn=push_images)

for i, data in enumerate(dataloader):
    output = model(*data, mode="train")
    import ipdb; ipdb.set_trace()
