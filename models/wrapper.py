from cv2.gapi import flip
from arctic.common.comet_utils import push_images
from arctic.src.callbacks.loss.loss_arctic_sf import compute_loss_light
from arctic.src.callbacks.process.process_arctic import process_data_light
from arctic.src.callbacks.vis.visualize_arctic import visualize_all
from arctic.src.models.hamer_local.model import HAMER
from arctic.src.models.generic.wrapper import GenericWrapper
import cv2
# from generic wrapper
import numpy as np
import torch
import os
import arctic.common.data_utils as data_utils
import arctic.common.ld_utils as ld_utils
from arctic.common.body_models import MANODecimator
from arctic.common.xdict import xdict
from arctic.common.data_utils import unormalize_kp2d
from copy import deepcopy
from hamer.utils.render_openpose import render_hand_keypoints

def mul_loss_dict(loss_dict):
    for key, val in loss_dict.items():
        loss, weight = val
        loss_dict[key] = loss * weight
    return loss_dict

class HAMERWrapper(GenericWrapper):
    def __init__(self, args, push_images_fn):
        super().__init__(args, push_images_fn)
        self.model = HAMER(
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )
        self.process_fn = process_data_light
        self.loss_fn = compute_loss_light
        self.metric_dict = [
            "mrrpe.rl",
            "mpjpe.ra",
            "mpjpe.pa.ra",
            'pix_err',
        ]

        if args.get('val_dataset', None) == 'epic':
            self.metric_dict = [
                "pix_err",
            ]

        self.vis_fns = [visualize_all]

        self.num_vis_train = 1
        self.num_vis_val = 1

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)

    def forward(self, inputs, targets, meta_info, mode):
        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l,
            # "arti_head": self.model.arti_head,
            "mesh_sampler": MANODecimator(),
            "object_sampler": self.object_sampler,
        }
        # models["arti_head"].object_tensors.to(self.device) # added by aditya

        self.set_flags(mode)
        inputs = xdict(inputs)
        targets = xdict(targets)
        meta_info = xdict(meta_info)
        # inputs_og = deepcopy(inputs)
        # targets_og = deepcopy(targets)
        # meta_info_og = deepcopy(meta_info)
        with torch.no_grad():
            inputs, targets, meta_info = self.process_fn(
                models, inputs, targets, meta_info, mode, self.args
            )
            # inputs_og, targets_og, meta_info_og = process_data(
            #     models, inputs_og, targets_og, meta_info_og, mode, self.args
            # )
        
        ### visualization to test input data
        
        # prepare for visualization
        if not os.path.exists('__test__'):
            os.makedirs('__test__')
        img_idx_str = meta_info['imgname'][0].split('/')[-1].split('.')[0]
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        # plot image
        img = inputs['img'][0].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        img = img * 255
        img = img.type(torch.uint8)
        img = img.numpy().transpose(1, 2, 0)
        cv2.imwrite(f"__test__/{img_idx_str}.png", img[:,:,::-1])

        # plot 2d keypoints with image
        keypoints_2d = targets['mano.j2d.norm.r'].cpu()
        keypoints_2d = (keypoints_2d + 1) * self.args.img_res / 2
        keypoints_2d = keypoints_2d[0]
        # keypoints_2d = (keypoints_2d + 0.5) * 224
        keypoints_2d = torch.cat([keypoints_2d, torch.ones(keypoints_2d.shape[0], 1)], dim=-1)
        keypoints_2d = keypoints_2d.cpu().numpy()
        keypoints_2d = keypoints_2d[mano_to_openpose]
        output_img = render_hand_keypoints(img, keypoints_2d)
        cv2.imwrite(f"__test__/{img_idx_str}_2dj.png", output_img[:, :, ::-1])
        
        K = meta_info['intrinsics'].cpu()[0]
        keypoints_3d = targets['mano.j3d.cam.r'].cpu()
        keypoints_3d = keypoints_3d[0]
        
        proj_keypoints_3d = torch.matmul(K, keypoints_3d.t()).t()
        proj_keypoints_3d = proj_keypoints_3d[:, :2] / proj_keypoints_3d[:, 2:]
        proj_keypoints_3d = torch.cat([proj_keypoints_3d, torch.ones(proj_keypoints_3d.shape[0], 1)], dim=-1)
        proj_keypoints_3d = proj_keypoints_3d.numpy()[mano_to_openpose]
        output_img = render_hand_keypoints(img, proj_keypoints_3d)
        cv2.imwrite(f"__test__/{img_idx_str}_3dj.png", output_img[:, :, ::-1])
        import ipdb ; ipdb.set_trace()
        
        
        
        
        
        
        
        

        # move_keys = ["object.v_len"]
        # for key in move_keys:
        #     meta_info[key] = targets[key]
        meta_info["mano.faces.r"] = self.mano_r.faces
        meta_info["mano.faces.l"] = self.mano_l.faces
        pred = self.model(inputs, meta_info, targets=targets) # targets only for debugging

        ####### extra code for debugging flip augmentation #######
        # seems to be working fine
        # import os
        # from PIL import Image, ImageDraw
        # # import pytorch3d.transforms.rotation_conversions as rot_conv
        # flip_cam_l = targets['mano.cam_t.wp.r'].cpu()*torch.Tensor([[1,-1,1]])
        # pose_l_aa = targets['mano.pose.r'].cpu()
        # pose_l_aa[:,1::3] *= -1
        # pose_l_aa[:,2::3] *= -1
        # shape_l = targets['mano.beta.r'].cpu()
        # K = meta_info['intrinsics'].cpu()
        # tmp_mano_l = self.model.mano_l.cpu()
        # out_l = tmp_mano_l(rotmat=pose_l_aa, shape=shape_l, K=K, cam=flip_cam_l)

        # flip_cam_r = targets['mano.cam_t.wp.l'].cpu()*torch.Tensor([[1,-1,1]])
        # pose_r_aa = targets['mano.pose.l'].cpu()
        # pose_r_aa[:,1::3] *= -1
        # pose_r_aa[:,2::3] *= -1
        # shape_r = targets['mano.beta.l'].cpu()
        # tmp_mano_r = self.model.mano_r.cpu()
        # out_r = tmp_mano_r(rotmat=pose_r_aa, shape=shape_r, K=K, cam=flip_cam_r)
        
        # save_dir = 'logs/ig_hands_debug/flip'
        # os.makedirs(save_dir, exist_ok=True)
        # bz = inputs['img'].shape[0]
        # size = 5
        # for idx in range(bz):
        #     tmp = inputs['img'][idx].unsqueeze(0)
        #     f_img = data_utils.denormalize_images(tmp)[0].cpu()
        #     pil_f_img = Image.fromarray((255*f_img.permute(1,2,0).numpy()).astype(np.uint8))
        #     pil_f_img.save(os.path.join(save_dir, 'f_img_v_{}.png'.format(idx)))
        #     draw_f = ImageDraw.Draw(pil_f_img)

        #     if meta_info['is_flipped'][idx] == 0:
        #         j2d_l =  targets['mano.j2d.norm.l'][idx]
        #         j2d = (j2d_l+1)*112
        #         for j in j2d:
        #             draw_f.ellipse([j[0]-size//2,j[1]-size//2,j[0]+size//2,j[1]+size//2], outline='cyan')
        #         pil_f_img.save(os.path.join(save_dir, 'f_img_j_{}.png'.format(idx)))

        #         j2d_r =  targets['mano.j2d.norm.r'][idx]
        #         j2d = (j2d_r+1)*112
        #         for j in j2d:
        #             draw_f.ellipse([j[0]-size//2,j[1]-size//2,j[0]+size//2,j[1]+size//2], outline='blue')
        #         pil_f_img.save(os.path.join(save_dir, 'f_img_j_{}.png'.format(idx)))
        #     else:
        #         j2d_l = (out_l['j2d.norm.l'][idx]+1)*112
        #         for j in j2d_l:
        #             draw_f.ellipse([j[0]-size//2,j[1]-size//2,j[0]+size//2,j[1]+size//2], outline='yellow')
        #         pil_f_img.save(os.path.join(save_dir, 'f_img_j_{}.png'.format(idx)))

        #         j2d_r = (out_r['j2d.norm.r'][idx]+1)*112
        #         for j in j2d_r:
        #             draw_f.ellipse([j[0]-size//2,j[1]-size//2,j[0]+size//2,j[1]+size//2], outline='red')
        #         pil_f_img.save(os.path.join(save_dir, 'f_img_j_{}.png'.format(idx)))
        ########################################

        loss_dict = self.loss_fn(
            pred=pred, gt=targets, meta_info=meta_info, args=self.args
        )
        loss_dict = {k: (loss_dict[k][0].mean(), loss_dict[k][1]) for k in loss_dict}
        loss_dict_unweighted = loss_dict.copy()
        loss_dict = mul_loss_dict(loss_dict)

        # flip the input img for visualization
        if sum(meta_info["is_flipped"]) > 0:
            flip_img = torch.flip(inputs["img"], dims=[3])
            flip_img = flip_img.contiguous()
            mixed_img = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,flip_img.shape[1], flip_img.shape[2], flip_img.shape[3]), flip_img, inputs['img'])
            inputs.overwrite('img', mixed_img)
        
        # temporary fix
        all_loss = 0.0
        for k in loss_dict:
            if 'loss' in k:
                all_loss += loss_dict[k]
        loss_dict['loss'] = all_loss
        # loss_dict["loss"] = sum(loss_dict[k] for k in loss_dict)

        # conversion for vis and eval
        keys = list(pred.keys())
        for key in keys:
            # denormalize 2d keypoints
            if "2d.norm" in key:
                denorm_key = key.replace(".norm", "")
                assert key in targets.keys(), f"Do not have key {key}"

                val_pred = pred[key]
                val_gt = targets[key]

                val_denorm_pred = data_utils.unormalize_kp2d(
                    val_pred, self.args.img_res
                )
                val_denorm_gt = data_utils.unormalize_kp2d(val_gt, self.args.img_res)

                pred[denorm_key] = val_denorm_pred
                targets[denorm_key] = val_denorm_gt

        if mode == "train":
            return {"out_dict": (inputs, targets, meta_info, pred), "loss": loss_dict, "loss_unweighted": loss_dict_unweighted}

        if mode == "vis":
            vis_dict = xdict()
            vis_dict.merge(inputs.prefix("inputs."))
            vis_dict.merge(pred.prefix("pred."))
            vis_dict.merge(targets.prefix("targets."))
            vis_dict.merge(meta_info.prefix("meta_info."))
            vis_dict = vis_dict.detach()
            return vis_dict

        # evaluate metrics
        metrics_all = self.evaluate_metrics(
            pred, targets, meta_info, self.metric_dict
        ).to_torch()
        out_dict = xdict()
        out_dict["imgname"] = meta_info["imgname"]
        out_dict.merge(ld_utils.prefix_dict(metrics_all, "metric."))

        if mode == "extract":
            mydict = xdict()
            mydict.merge(inputs.prefix("inputs."))
            mydict.merge(pred.prefix("pred."))
            mydict.merge(targets.prefix("targets."))
            mydict.merge(meta_info.prefix("meta_info."))
            mydict = mydict.detach()
            return mydict
        return out_dict, loss_dict
