import json, pickle, os, cv2, math
import os.path as op

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import sys
sys.path.append('/home/kt19/projects/kaiyuan/hamer/arctic')
import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
from common.object_tensors import ObjectTensors
from src.datasets.dataset_utils import get_valid, pad_jts2d


class HandyLightDataset(Dataset):
    def __getitem__(self, index):
        if self.rand_idx is not None:
            index = self.rand_idx
        imgname = self.imgnames[index]
        data = self.getitem(imgname)
        return data

    def getitem(self, imgname, load_rgb=True):
        args = self.args
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_cam = seq_data["cam_coord"]
        data_2d = seq_data["2d"]
        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
        vidx, is_valid, right_valid, left_valid = get_valid(
            data_2d, data_cam, vidx, view_idx, imgname
        )

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # hands
        joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints3d_r = data_cam["joints.right"][vidx, view_idx].copy()

        joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
        joints3d_l = data_cam["joints.left"][vidx, view_idx].copy()

        pose_r = data_params["pose_r"][vidx].copy()
        betas_r = data_params["shape_r"][vidx].copy()
        pose_l = data_params["pose_l"][vidx].copy()
        betas_l = data_params["shape_l"][vidx].copy()

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()
        # NOTE:
        # kp2d, kp3d are in undistored space
        # thus, results for evaluation is in the undistorted space (non-curved)
        # dist parameters can be used for rendering in visualization

        # objects
        # bbox2d = pad_jts2d(data_2d["bbox3d"][vidx, view_idx].copy())
        # bbox3d = data_cam["bbox3d"][vidx, view_idx].copy()
        # bbox2d_t = bbox2d[:8]
        # bbox2d_b = bbox2d[8:]
        # bbox3d_t = bbox3d[:8]
        # bbox3d_b = bbox3d[8:]

        # kp2d = pad_jts2d(data_2d["kp3d"][vidx, view_idx].copy())
        # kp3d = data_cam["kp3d"][vidx, view_idx].copy()
        # kp2d_t = kp2d[:16]
        # kp2d_b = kp2d[16:]
        # kp3d_t = kp3d[:16]
        # kp3d_b = kp3d[16:]

        # obj_radian = data_params["obj_arti"][vidx].copy()

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        # LOADING END

        # SPEEDUP PROCESS
        (
            joints2d_r,
            joints2d_l,
            bbox,
        ) = dataset_utils.transform_2d_for_speedup_light(
            speedup,
            is_egocam,
            joints2d_r,
            joints2d_l,
            bbox,
            args.ego_image_scale,
        )
        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace(
                "/arctic_data/", "/data/arctic_data/data/"
            ).replace("/data/data/", "/data/")
            # imgname = imgname.replace("/arctic_data/", "/data/arctic_data/")
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        # if not is_egocam: # scale reference is changed from 200 to 300 for allocentric images
        #     scale = scale * 1.5
        # self.aug_data = False # temporarily disable data augmentation for debugging

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )
        # kp2d_b = data_utils.j2d_processing(
        #     kp2d_b, center, scale, augm_dict, args.img_res
        # )
        # kp2d_t = data_utils.j2d_processing(
        #     kp2d_t, center, scale, augm_dict, args.img_res
        # )
        # bbox2d_b = data_utils.j2d_processing(
        #     bbox2d_b, center, scale, augm_dict, args.img_res
        # )
        # bbox2d_t = data_utils.j2d_processing(
        #     bbox2d_t, center, scale, augm_dict, args.img_res
        # )
        # bbox2d = np.concatenate((bbox2d_t, bbox2d_b), axis=0)
        # kp2d = np.concatenate((kp2d_t, kp2d_b), axis=0)

        # if args.get('use_render_seg_loss', False): # dummy mask loading to balance load b/w dataloaders (debugging I/O bottleneck)
        #     dummy_path = '/data02/adityap9/projects/hands23.beta/vis/visor_hand_masks/masks/2_0_P03_101_frame_0000038953.png'
        #     dummy_mask, _ = read_img(dummy_path, (2800, 2000, 3))
        #     dummy_mask, _ = read_img(dummy_path, (2800, 2000, 3))

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )

            # bbox format below: [x0, y0, w, h] for right_bbox, left_bbox
            if 'train' in self.split or args.use_gt_bbox:
                # compute bbox from ground truth joints2d during training
                j2d_r_pix = ((joints2d_r[...,:2]+1)/2)*(args.img_res-1)
                j2d_l_pix = ((joints2d_l[...,:2]+1)/2)*(args.img_res-1)
                right_bbox = np.array([j2d_r_pix[...,0].min(), j2d_r_pix[...,1].min(), j2d_r_pix[...,0].max(), j2d_r_pix[...,1].max()]).clip(0, args.img_res-1)
                left_bbox = np.array([j2d_l_pix[...,0].min(), j2d_l_pix[...,1].min(), j2d_l_pix[...,0].max(), j2d_l_pix[...,1].max()]).clip(0, args.img_res-1)
                right_bbox = np.array([right_bbox[0], right_bbox[1], right_bbox[2]-right_bbox[0], right_bbox[3]-right_bbox[1]]).astype(np.int16)
                left_bbox = np.array([left_bbox[0], left_bbox[1], left_bbox[2]-left_bbox[0], left_bbox[3]-left_bbox[1]]).astype(np.int16)
                right_bbox_og = right_bbox.copy()
                left_bbox_og = left_bbox.copy()
                if right_bbox[2] == 0 or right_bbox[3] == 0: 
                    right_bbox = None # no right hand in the image
                    right_bbox_og = np.array([0, 0, args.img_res-1, args.img_res-1])
                if left_bbox[2] == 0 or left_bbox[3] == 0: 
                    left_bbox = None # no left hand in the image
                    left_bbox_og = np.array([0, 0, args.img_res-1, args.img_res-1])
            else:
                # get bounding box information from frankmocap
                hand_bbox = self.bbox_dict[imgname]
                right_bbox, left_bbox = hand_bbox['bbox_right'], hand_bbox['bbox_left']
                if args.bbox_detector == 'frankmocap': # these are computed at 840 resolution
                    down_scale = args.img_res / 840
                    if right_bbox is not None: right_bbox = (right_bbox * down_scale).astype(np.int16)
                    if left_bbox is not None: left_bbox = (left_bbox * down_scale).astype(np.int16)
            
            if self.aug_data:
                right_bbox, left_bbox = data_utils.jitter_bbox(right_bbox), data_utils.jitter_bbox(left_bbox)
                if right_bbox is not None:
                    new_right_bbox = np.array([right_bbox[0], right_bbox[1], right_bbox[0]+right_bbox[2], right_bbox[1]+right_bbox[3]]).astype(np.int16).clip(0, args.img_res-1)
                    if (new_right_bbox[2]-new_right_bbox[0]) == 0 or (new_right_bbox[3]-new_right_bbox[1]) == 0: right_bbox = None
                if left_bbox is not None:
                    new_left_bbox = np.array([left_bbox[0], left_bbox[1], left_bbox[0]+left_bbox[2], left_bbox[1]+left_bbox[3]]).astype(np.int16).clip(0, args.img_res-1)
                    if (new_left_bbox[2]-new_left_bbox[0]) == 0 or (new_left_bbox[3]-new_left_bbox[1]) == 0: left_bbox = None
            
            def crop_and_pad(img, bbox, scale=1):
                if bbox is None: # resize the image and return
                    img_crop = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [args.img_res/2, args.img_res/2, args.img_res, args.img_res], 
                            1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0].transpose(2,0,1)
                    img_crop = np.clip(img_crop, 0, 1)
                    new_bbox = np.array([0, 0, args.img_res-1, args.img_res-1])
                    return img_crop, new_bbox

                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
                x_mid, y_mid, width, height = (x0+x1)//2, (y0+y1)//2, x1-x0, y1-y0 
                size = max(width, height)
                img_crop = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [x_mid, y_mid, size*scale, size*scale], 1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0]
                img_crop = np.clip(img_crop, 0, 1)
                new_bbox = np.array([x_mid-(size*scale)//2, y_mid-(size*scale)//2, x_mid+(size*scale)//2, y_mid+(size*scale)//2]).clip(0, args.img_res-1).astype(np.int16)
                return img_crop.transpose(2,0,1), new_bbox # np.array([x0, y0, x1, y1])
            
            bbox_scale = 1.5
            if args.get('bbox_scale', None) is not None:
                bbox_scale = args.bbox_scale
            # bbox format below; [x0, y0, x1, y1] for r_bbox, l_bbox
            r_img, r_bbox = crop_and_pad(img, right_bbox, scale=bbox_scale)
            l_img, l_bbox = crop_and_pad(img, left_bbox, scale=bbox_scale)
            norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
            norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

            img_ds = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [args.img_res/2, args.img_res/2, args.img_res, args.img_res], 
                            1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0].transpose(2,0,1)
            img_ds = np.clip(img_ds, 0, 1)
            img_ds = torch.from_numpy(img_ds).float()
            norm_img = self.normalize_img(img_ds)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        inputs["r_img"] = norm_r_img
        inputs["l_img"] = norm_l_img
        inputs["r_bbox"] = r_bbox
        inputs["l_bbox"] = l_bbox

        if augm_dict["flip"] == 1:
            inputs['img'] = torch.flip(norm_img, dims=[2])
            inputs['r_img'] = torch.flip(norm_r_img, dims=[2])
            inputs['l_img'] = torch.flip(norm_l_img, dims=[2])
            n_r_bbox = np.array([args.img_res-1, 0, args.img_res-1, 0]) + np.array([-1, 1, -1, 1]) * l_bbox
            n_l_bbox = np.array([args.img_res-1, 0, args.img_res-1, 0]) + np.array([-1, 1, -1, 1]) * r_bbox
            inputs['r_bbox'] = np.array([n_r_bbox[2], n_r_bbox[1], n_r_bbox[0], n_r_bbox[3]])
            inputs['l_bbox'] = np.array([n_l_bbox[2], n_l_bbox[1], n_l_bbox[0], n_l_bbox[3]])

        if args.use_gt_bbox:
            inputs['r_bbox_og'] = right_bbox_og
            inputs['l_bbox_og'] = left_bbox_og

        meta_info["imgname"] = imgname
        rot_r = data_cam["rot_r_cam"][vidx, view_idx]
        rot_l = data_cam["rot_l_cam"][vidx, view_idx]

        pose_r = np.concatenate((rot_r, pose_r), axis=0)
        pose_l = np.concatenate((rot_l, pose_l), axis=0)

        # hands
        targets["mano.pose.r"] = torch.from_numpy(
            data_utils.pose_processing(pose_r, augm_dict)
        ).float()
        targets["mano.pose.l"] = torch.from_numpy(
            data_utils.pose_processing(pose_l, augm_dict)
        ).float()
        targets["mano.beta.r"] = torch.from_numpy(betas_r).float()
        targets["mano.beta.l"] = torch.from_numpy(betas_l).float()
        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

        # object
        # targets["object.kp3d.full.b"] = torch.from_numpy(kp3d_b[:, :3]).float()
        # targets["object.kp2d.norm.b"] = torch.from_numpy(kp2d_b[:, :2]).float()
        # targets["object.kp3d.full.t"] = torch.from_numpy(kp3d_t[:, :3]).float()
        # targets["object.kp2d.norm.t"] = torch.from_numpy(kp2d_t[:, :2]).float()

        # targets["object.bbox3d.full.b"] = torch.from_numpy(bbox3d_b[:, :3]).float()
        # targets["object.bbox2d.norm.b"] = torch.from_numpy(bbox2d_b[:, :2]).float()
        # targets["object.bbox3d.full.t"] = torch.from_numpy(bbox3d_t[:, :3]).float()
        # targets["object.bbox2d.norm.t"] = torch.from_numpy(bbox2d_t[:, :2]).float()
        # targets["object.radian"] = torch.FloatTensor(np.array(obj_radian))

        # targets["object.kp2d.norm"] = torch.from_numpy(kp2d[:, :2]).float()
        # targets["object.bbox2d.norm"] = torch.from_numpy(bbox2d[:, :2]).float()

        # compute RT from cano space to augmented space
        # this transform match j3d processing
        obj_idx = self.obj_names.index(obj_name)
        # meta_info["kp3d.cano"] = self.kp3d_cano[obj_idx] / 1000  # meter
        # kp3d_cano = meta_info["kp3d.cano"].numpy()
        # kp3d_target = targets["object.kp3d.full.b"][:, :3].numpy()

        # rotate canonical kp3d to match original image
        # R, _ = tf.solve_rigid_tf_np(kp3d_cano, kp3d_target)
        # obj_rot = (
        #     rot.batch_rot2aa(torch.from_numpy(R).float().view(1, 3, 3)).view(3).numpy()
        # )

        # # multiply rotation from data augmentation
        # obj_rot_aug = rot.rot_aa(obj_rot, augm_dict["rot"])
        # targets["object.rot"] = torch.FloatTensor(obj_rot_aug).view(1, 3)

        # full image camera coord
        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])
        # targets["object.kp3d.full.b"] = torch.FloatTensor(kp3d_b[:, :3])

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        fixed_focal_length = args.focal_length
        intrx = data_utils.get_aug_intrix(
            intrx,
            fixed_focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k
        else:
            intrx = intrx.numpy() # make format consistent with gt intrinsics used in egocentric setting
        
        intrx_for_enc = intrx.copy()
        if args.get('no_intrx', False):
            intrx_for_enc = np.eye(3)
            intrx_for_enc[0,0] = args.img_res / 2
            intrx_for_enc[1,1] = args.img_res / 2
            intrx_for_enc[0,2] = args.img_res / 2
            intrx_for_enc[1,2] = args.img_res / 2

        if args.pos_enc is not None: # compute positional encoding for different pixels
            L = args.n_freq_pos_enc
            
            if 'center' in args.pos_enc or 'perspective_correction' in args.pos_enc:
                # center of left & right image
                r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
                l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
                r_angle_x, r_angle_y = np.arctan2(r_center[0]-intrx_for_enc[0,2], intrx_for_enc[0,0]), np.arctan2(r_center[1]-intrx_for_enc[1,2], intrx_for_enc[1,1])
                l_angle_x, l_angle_y = np.arctan2(l_center[0]-intrx_for_enc[0,2], intrx_for_enc[0,0]), np.arctan2(l_center[1]-intrx_for_enc[1,2], intrx_for_enc[1,1])
                r_angle = np.array([r_angle_x, r_angle_y]).astype(np.float32)
                l_angle = np.array([l_angle_x, l_angle_y]).astype(np.float32)
                inputs["r_center_angle"], inputs["l_center_angle"] = r_angle, l_angle
                targets['center.r'], targets['center.l'] = r_angle, l_angle
                # move this computation to model.forward(), takes too much time on cpu
                # r_center_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1)*r_angle.reshape(1,-1)), np.cos(2**np.arange(L).reshape(-1,1)*r_angle.reshape(1,-1))], axis=-1).flatten()
                # l_center_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1)*l_angle.reshape(1,-1)), np.cos(2**np.arange(L).reshape(-1,1)*l_angle.reshape(1,-1))], axis=-1).flatten()
                # r_center_pos_enc = torch.from_numpy(r_center_pos_enc).float()
                # l_center_pos_enc = torch.from_numpy(l_center_pos_enc).float()
                # inputs['r_center_pos_enc'], inputs['l_center_pos_enc'] = r_center_pos_enc, l_center_pos_enc

            if 'corner' in args.pos_enc:
                # corners of left & right image
                r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
                l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
                r_corner = np.stack([r_corner[:,0]-intrx_for_enc[0,2], r_corner[:,1]-intrx_for_enc[1,2]], axis=-1)
                l_corner = np.stack([l_corner[:,0]-intrx_for_enc[0,2], l_corner[:,1]-intrx_for_enc[1,2]], axis=-1)
                r_angle = np.arctan2(r_corner, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).flatten().astype(np.float32)
                l_angle = np.arctan2(l_corner, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).flatten().astype(np.float32)
                inputs["r_corner_angle"], inputs["l_corner_angle"] = r_angle, l_angle
                targets['corner.r'], targets['corner.l'] = r_angle, l_angle
                # move this computation to model.forward(), takes too much time on cpu
                # r_corner_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1)*r_angle.reshape(1,-1)), np.cos(2**np.arange(L).reshape(-1,1)*r_angle.reshape(1,-1))], axis=-1).flatten()
                # l_corner_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1)*l_angle.reshape(1,-1)), np.cos(2**np.arange(L).reshape(-1,1)*l_angle.reshape(1,-1))], axis=-1).flatten()
                # r_corner_pos_enc = torch.from_numpy(r_corner_pos_enc).float()
                # l_corner_pos_enc = torch.from_numpy(l_corner_pos_enc).float()
                # inputs['r_corner_pos_enc'], inputs['l_corner_pos_enc'] = r_corner_pos_enc, l_corner_pos_enc

            if 'dense' in args.pos_enc:
                # dense positional encoding for all pixels
                r_x_grid, r_y_grid = range(r_bbox[0], r_bbox[2]+1), range(r_bbox[1], r_bbox[3]+1)
                r_x_grid, r_y_grid = np.meshgrid(r_x_grid, r_y_grid, indexing='ij') # torch doesn't support batched meshgrid
                l_x_grid, l_y_grid = range(l_bbox[0], l_bbox[2]+1), range(l_bbox[1], l_bbox[3]+1)
                l_x_grid, l_y_grid = np.meshgrid(l_x_grid, l_y_grid, indexing='ij')
                r_pix = np.stack([r_x_grid-intrx_for_enc[0,2], r_y_grid-intrx_for_enc[1,2]], axis=-1)
                l_pix = np.stack([l_x_grid-intrx_for_enc[0,2], l_y_grid-intrx_for_enc[1,2]], axis=-1)
                # r_pix = np.stack([r_x_grid, r_y_grid], axis=-1) # analogous to virtual depth?
                # l_pix = np.stack([l_x_grid, l_y_grid], axis=-1)
                r_angle = np.arctan2(r_pix, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).transpose(2,0,1).astype(np.float32)
                l_angle = np.arctan2(l_pix, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).transpose(2,0,1).astype(np.float32)
                # r_angle = r_pix.transpose(2,0,1).astype(np.float32) # analagous to translation?
                # l_angle = l_pix.transpose(2,0,1).astype(np.float32)
                # inputs["r_dense_angle"], inputs["l_dense_angle"] = r_angle, l_angle
                r_angle_fdim = np.zeros((r_angle.shape[0], args.img_res, args.img_res))
                l_angle_fdim = np.zeros((l_angle.shape[0], args.img_res, args.img_res))
                r_angle_fdim[:r_angle.shape[0], :r_angle.shape[1], :r_angle.shape[2]] = r_angle
                l_angle_fdim[:l_angle.shape[0], :l_angle.shape[1], :l_angle.shape[2]] = l_angle
                inputs["r_dense_angle"], inputs["l_dense_angle"] = r_angle_fdim.astype(np.float32), l_angle_fdim.astype(np.float32)
                r_mask = np.zeros((args.img_res, args.img_res))
                r_mask[:r_angle.shape[1], :r_angle.shape[2]] = 1
                l_mask = np.zeros((args.img_res, args.img_res))
                l_mask[:l_angle.shape[1], :l_angle.shape[2]] = 1
                inputs["r_dense_mask"], inputs["l_dense_mask"] = r_mask.astype(np.float32), l_mask.astype(np.float32)
                # move this computation to model.forward(), takes too much time on cpu
                # r_c, r_w, r_h = r_angle.shape
                # r_dense_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1,1,1)*r_angle.reshape(1,r_c,r_w,r_h)), 
                #                     np.cos(2**np.arange(L).reshape(-1,1,1,1)*r_angle.reshape(1,r_c,r_w,r_h))], axis=2).reshape(-1, r_w, r_h)
                # l_c, l_w, l_h = l_angle.shape
                # l_dense_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1,1,1)*l_angle.reshape(1,l_c,l_w,l_h)),
                #                     np.cos(2**np.arange(L).reshape(-1,1,1,1)*l_angle.reshape(1,l_c,l_w,l_h))], axis=2).reshape(-1, l_w, l_h)
                # r_dense_pos_enc = torch.from_numpy(r_dense_pos_enc).float()
                # l_dense_pos_enc = torch.from_numpy(l_dense_pos_enc).float()
                # r_dense_pos_enc = F.interpolate(r_dense_pos_enc.unsqueeze(0), size=(args.img_res_ds, args.img_res_ds), mode='bilinear', align_corners=True).squeeze(0)
                # l_dense_pos_enc = F.interpolate(l_dense_pos_enc.unsqueeze(0), size=(args.img_res_ds, args.img_res_ds), mode='bilinear', align_corners=True).squeeze(0)
                # inputs['r_dense_pos_enc'], inputs['l_dense_pos_enc'] = r_dense_pos_enc, l_dense_pos_enc

            if 'cam_conv' in args.pos_enc:
                # dense positional encoding for all pixels
                r_x_grid, r_y_grid = range(r_bbox[0], r_bbox[2]+1), range(r_bbox[1], r_bbox[3]+1)
                r_x_grid, r_y_grid = np.meshgrid(r_x_grid, r_y_grid, indexing='ij') # torch doesn't support batched meshgrid
                l_x_grid, l_y_grid = range(l_bbox[0], l_bbox[2]+1), range(l_bbox[1], l_bbox[3]+1)
                l_x_grid, l_y_grid = np.meshgrid(l_x_grid, l_y_grid, indexing='ij')
                r_pix = np.stack([r_x_grid-intrx_for_enc[0,2], r_y_grid-intrx_for_enc[1,2]], axis=-1)
                l_pix = np.stack([l_x_grid-intrx_for_enc[0,2], l_y_grid-intrx_for_enc[1,2]], axis=-1)
                r_pix_transp = r_pix.transpose(2,0,1).astype(np.float32)
                l_pix_transp = l_pix.transpose(2,0,1).astype(np.float32)
                r_pix_centered = np.stack([2*r_x_grid/args.img_res-1, 2*r_y_grid/args.img_res-1], axis=-1).transpose(2,0,1).astype(np.float32)
                l_pix_centered = np.stack([2*l_x_grid/args.img_res-1, 2*l_y_grid/args.img_res-1], axis=-1).transpose(2,0,1).astype(np.float32)
                r_angle = np.arctan2(r_pix, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).transpose(2,0,1).astype(np.float32)
                l_angle = np.arctan2(l_pix, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).transpose(2,0,1).astype(np.float32)
                
                # (w,h) format, check which is correct
                r_angle_fdim = np.zeros((r_angle.shape[0]+r_pix_transp.shape[0]+r_pix_centered.shape[0], args.img_res, args.img_res))
                r_angle_fdim[:r_angle.shape[0], :r_angle.shape[1], :r_angle.shape[2]] = r_angle
                r_angle_fdim[r_angle.shape[0]:r_angle.shape[0]+r_pix_transp.shape[0], :r_angle.shape[1], :r_angle.shape[2]] = r_pix_transp
                r_angle_fdim[r_angle.shape[0]+r_pix_transp.shape[0]:, :r_angle.shape[1], :r_angle.shape[2]] = r_pix_centered

                l_angle_fdim = np.zeros((l_angle.shape[0]+l_pix_transp.shape[0]+r_pix_centered.shape[0], args.img_res, args.img_res))
                l_angle_fdim[:l_angle.shape[0], :l_angle.shape[1], :l_angle.shape[2]] = l_angle
                l_angle_fdim[l_angle.shape[0]:l_angle.shape[0]+l_pix_transp.shape[0], :l_angle.shape[1], :l_angle.shape[2]] = l_pix_transp
                l_angle_fdim[l_angle.shape[0]+l_pix_transp.shape[0]:, :l_angle.shape[1], :l_angle.shape[2]] = l_pix_centered

                inputs["r_dense_angle"], inputs["l_dense_angle"] = r_angle_fdim.astype(np.float32), l_angle_fdim.astype(np.float32)

                r_mask = np.zeros((args.img_res, args.img_res))
                r_mask[:r_angle.shape[1], :r_angle.shape[2]] = 1
                l_mask = np.zeros((args.img_res, args.img_res))
                l_mask[:l_angle.shape[1], :l_angle.shape[2]] = 1
                inputs["r_dense_mask"], inputs["l_dense_mask"] = r_mask.astype(np.float32), l_mask.astype(np.float32)

            if 'sinusoidal_cc' in args.pos_enc:
                # center of left & right image
                r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
                l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
                r_angle = 2*r_center/args.img_res - 1
                l_angle = 2*l_center/args.img_res - 1
                inputs["r_center_angle"], inputs["l_center_angle"] = r_angle, l_angle
                targets['center.r'], targets['center.l'] = r_angle, l_angle

                # corners of left & right image
                r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
                l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
                r_angle = 2*r_corner/args.img_res - 1
                l_angle = 2*l_corner/args.img_res - 1
                r_angle = r_angle.flatten()
                l_angle = l_angle.flatten()
                inputs["r_corner_angle"], inputs["l_corner_angle"] = r_angle, l_angle
                targets['corner.r'], targets['corner.l'] = r_angle, l_angle

            if 'pcl' in args.pos_enc: # https://github.com/yu-frank/PerspectiveCropLayers/blob/main/src/pcl_demo.ipynb

                # define functions for PCL
                def virtualCameraRotationFromPosition(position):
                    x, y, z = position[0], position[1], position[2]
                    n1x = math.sqrt(1 + x ** 2)
                    d1x = 1 / n1x
                    d1xy = 1 / math.sqrt(1 + x ** 2 + y ** 2)
                    d1xy1x = 1 / math.sqrt((1 + x ** 2 + y ** 2) * (1 + x ** 2))
                    R_virt2orig = np.array([d1x, -x * y * d1xy1x, x * d1xy,
                                            0*x,      n1x * d1xy, y * d1xy,
                                        -x * d1x,     -y * d1xy1x, 1 * d1xy]).reshape(3,3)
                    return R_virt2orig

                def bK_virt(p_position, K_c, bbox_size_img, focal_at_image_plane=True, slant_compensation=True):
                    p_length = np.linalg.norm(p_position)
                    focal_length_factor = 1
                    if focal_at_image_plane:
                        focal_length_factor *= p_length
                    if slant_compensation:
                        sx = 1.0 / math.sqrt(p_position[0]**2+p_position[2]**2)  # this is cos(phi)
                        sy = math.sqrt(p_position[0]**2+1) / math.sqrt(p_position[0]**2+p_position[1]**2 + 1)  # this is cos(theta)
                        bbox_size_img = np.array(bbox_size_img) * np.array([sx,sy])

                    f_orig = np.diag(K_c)[:2]
                    f_compensated = focal_length_factor * f_orig / bbox_size_img # dividing by the target bbox_size_img will make the coordinates normalized to 0..1, as needed for the perspective grid sample function; an alternative would be to make the grid_sample operate on pixel coordinates
                    K_virt = np.zeros((3,3))
                    K_virt[2,2] = 1
                    # Note, in unit image coordinates ranging from 0..1
                    K_virt[0, 0] = f_compensated[0]
                    K_virt[1, 1] = f_compensated[1]
                    K_virt[:2, 2] = 0.5
                    return K_virt

                def perspective_grid(P_virt2orig, image_pixel_size, crop_pixel_size_wh, transform_to_pytorch=False):
                    # create a grid of linearly increasing indices (one for each pixel, going from 0..1)
                    xs = torch.linspace(0, 1, crop_pixel_size_wh[0])
                    ys = torch.linspace(0, 1, crop_pixel_size_wh[1])

                    rs, cs = torch.meshgrid([xs, ys])
                    # cs = ys.view(1, -1).repeat(xs.size(0), 1)
                    # rs = xs.view(-1, 1).repeat(1, ys.size(0))
                    zs = torch.ones(rs.shape)  # init homogeneous coordinate to 1
                    pv = torch.stack([rs, cs, zs])

                    # expand along batch dimension
                    grid = pv.expand([3, crop_pixel_size_wh[0], crop_pixel_size_wh[1]])

                    # linearize the 2D grid to a single dimension, to apply transformation
                    bpv_lin = grid.view([3, -1])

                    # do the projection
                    bpv_lin_orig = torch.matmul(P_virt2orig, bpv_lin)
                    eps = 0.00000001
                    bpv_lin_orig_p = bpv_lin_orig[:2, :] / (eps + bpv_lin_orig[2:3, :]) # projection, divide homogeneous coord

                    # go back from linear to two–dimensional outline of points
                    bpv_orig = bpv_lin_orig_p.view(2, crop_pixel_size_wh[0], crop_pixel_size_wh[1])

                    # the sampling function assumes the position information on the last dimension
                    bpv_orig = bpv_orig.permute([2, 1, 0])

                    # the transformed points will be in pixel coordinates ranging from 0 up to the image width/height (unmapped from the original intrinsics matrix)
                    # but the pytorch grid_sample function assumes it in -1,..,1; the direction is already correct (assuming negative y axis, which is also assumed by bytorch)
                    if transform_to_pytorch:
                        bpv_orig /= image_pixel_size # map to 0..1
                        bpv_orig *= 2 # to 0...2
                        bpv_orig -= 1 # to -1...1

                    return bpv_orig

                r_c = (inputs['r_bbox'][:2]+inputs['r_bbox'][2:])/2
                l_c = (inputs['l_bbox'][:2]+inputs['l_bbox'][2:])/2
                r_w, r_h = inputs['r_bbox'][2]-inputs['r_bbox'][0], inputs['r_bbox'][3]-inputs['r_bbox'][1]
                l_w, l_h = inputs['l_bbox'][2]-inputs['l_bbox'][0], inputs['l_bbox'][3]-inputs['l_bbox'][1]
                K_inv = np.linalg.inv(intrx)
                r_pos = np.matmul(K_inv, np.array([r_c[0], r_c[1], 1]))
                l_pos = np.matmul(K_inv, np.array([l_c[0], l_c[1], 1]))
                
                def pcl_layer(p_pos, K_c, w_c, h_c):
                    # get rotation from orig to new coordinate frame
                    R_virt2orig = virtualCameraRotationFromPosition(p_pos)
                    # determine target frame
                    K_virt = bK_virt(p_pos, K_c, [w_c, h_c])
                    K_virt_inv = np.linalg.inv(K_virt)
                    # projective transformation orig to virtual camera
                    P_virt2orig = np.matmul(K_c, np.matmul(R_virt2orig, K_virt_inv))

                    # convert to torch
                    P_virt2orig = torch.from_numpy(P_virt2orig).float()
                    K_virt = torch.from_numpy(K_virt).float()
                    R_virt2orig = torch.from_numpy(R_virt2orig).float()

                    grid_perspective = perspective_grid(P_virt2orig, args.img_res, [w_c, h_c], transform_to_pytorch=True)
                    return grid_perspective, R_virt2orig

                r_size = max(r_w, r_h)
                if r_size == 0: r_size = args.img_res
                r_grid_perspective, R_virt2orig_r = pcl_layer(r_pos, intrx.copy(), r_size, r_size)
                l_size = max(l_w, l_h)
                if l_size == 0: l_size = args.img_res
                l_grid_perspective, R_virt2orig_l = pcl_layer(l_pos, intrx.copy(), l_size, l_size)
                    
                n_r_img = F.grid_sample(inputs['img'].unsqueeze(0), r_grid_perspective.unsqueeze(0))[0]
                n_l_img = F.grid_sample(inputs['img'].unsqueeze(0), l_grid_perspective.unsqueeze(0))[0]

                # resize n_r_img, n_l_img to args.img_res
                n_r_img = F.interpolate(n_r_img.unsqueeze(0), size=(args.img_res, args.img_res), mode='bilinear', align_corners=True)[0]
                n_l_img = F.interpolate(n_l_img.unsqueeze(0), size=(args.img_res, args.img_res), mode='bilinear', align_corners=True)[0]

                inputs['r_img'] = n_r_img
                inputs['l_img'] = n_l_img
                inputs['r_rot'] = R_virt2orig_r
                inputs['l_rot'] = R_virt2orig_l

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")
        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        meta_info['dataset'] = 'handy'
        # meta_info["sample_index"] = index

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = 1
        meta_info['is_j3d_loss'] = 1
        meta_info['is_beta_loss'] = 1
        meta_info['is_pose_loss'] = 1
        meta_info['is_cam_loss'] = 1

        meta_info['is_grasp_loss'] = 0
        targets["grasp.l"] = 8 # no grasp
        targets["grasp.r"] = 8 # no grasp
        targets['grasp_valid_r'] = 0
        targets['grasp_valid_l'] = 0

        # root and at least 3 joints inside image
        targets["is_valid"] = float(is_valid)
        targets["left_valid"] = float(left_valid) * float(is_valid)
        targets["right_valid"] = float(right_valid) * float(is_valid)
        targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

        if args.get('use_render_seg_loss', False):
            meta_info['is_mask_loss'] = 0
            # dummy values
            targets['render.r'] = torch.zeros((1, args.img_res_ds, args.img_res_ds))
            targets['render.l'] = torch.zeros((1, args.img_res_ds, args.img_res_ds))
            targets['render_valid_r'] = 0
            targets['render_valid_l'] = 0

        if args.get('use_depth_loss', False):
            meta_info['is_depth_loss'] = 0
            targets['depth.r'] = torch.zeros((args.img_res, args.img_res))
            targets['depth.l'] = torch.zeros((args.img_res, args.img_res))

        return inputs, targets, meta_info

    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
        assert len(imgnames) == len(set(imgnames))
        imgnames = dataset_utils.downsample(imgnames, split)
        self.imgnames = imgnames

    def _load_data(self, args, split, seq, **kwargs):
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        # during inference, turn off
        if seq is not None:
            self.aug_data = False
        
        if args.method == "frankmocap":
            self.normalize_img = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        else:
            self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            self.mode = "train"
        elif "val" in split:
            self.mode = "val"
        elif "test" in split:
            self.mode = "test"

        setup = kwargs.get('setup', None)
        if setup is None:
            setup = args.setup

        short_split = split.replace("mini", "").replace("tiny", "").replace("small", "")
        data_p = op.join(
            f"/data02/adityap9/datasets/arctic/data/arctic_data/data/splits/{setup}_{short_split}.npy"
        )
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()

        self.data = data["data_dict"]
        self.imgnames = data["imgnames"]

        # out_file = 'logs/debug/imgnames_p1_val.txt'
        # for img in self.imgnames:
        #     with open(out_file, 'a') as f:
        #         f.write(img+'\n')
        # exit()

        with open("/data02/adityap9/datasets/arctic/data/arctic_data/data/meta/misc.json", "r") as f:
            misc = json.load(f)

        if not args.use_gt_bbox and 'val' in self.split: # only during validation
            # load bboxes estimated from frankmocap
            if args.bbox_detector == 'frankmocap':
                bbox_file = "/home/adityap9/projects/ihoi/output/debug/arctic_bboxes/hand_bbox_{}_{}.pkl".format(setup, short_split)
            elif args.bbox_detector == 'arctic':
                bbox_file = "/data02/adityap9/projects/arctic/logs/bbox_hands_debug/bbox_{}_{}_working.pkl".format(setup, short_split)
            with open(bbox_file, "rb") as f:
                bbox_dict = pickle.load(f)
            self.bbox_dict = bbox_dict

        # unpack
        subjects = list(misc.keys())
        intris_mat = {}
        world2cam = {}
        image_sizes = {}
        ioi_offset = {}
        for subject in subjects:
            world2cam[subject] = misc[subject]["world2cam"]
            intris_mat[subject] = misc[subject]["intris_mat"]
            image_sizes[subject] = misc[subject]["image_size"]
            ioi_offset[subject] = misc[subject]["ioi_offset"]

        self.world2cam = world2cam
        self.intris_mat = intris_mat
        self.image_sizes = image_sizes
        self.ioi_offset = ioi_offset

        object_tensors = ObjectTensors()
        self.kp3d_cano = object_tensors.obj_tensors["kp_bottom"]
        self.obj_names = object_tensors.obj_tensors["names"]
        self.egocam_k = None

    def __init__(self, args, split, seq=None, **kwargs):
        ###### debugging rendering
        self.rand_idx = None
        if kwargs.get('rend', False):
            self.rand_idx = 1000
        ######
        self._load_data(args, split, seq, **kwargs)
        self._process_imgnames(seq, split)
        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )

    def __len__(self):
        if self.args.debug:
            return 1
        return len(self.imgnames)

    def getitem_eval(self, imgname, load_rgb=True):
        args = self.args
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        # SPEEDUP PROCESS
        bbox = dataset_utils.transform_bbox_for_speedup(
            speedup,
            is_egocam,
            bbox,
            args.ego_image_scale,
        )
        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace("/arctic_data/", "/data/arctic_data/data/")
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]
        self.aug_data = False

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )

            # get bounding box information from frankmocap
            hand_bbox = self.bbox_dict[imgname]
            right_bbox, left_bbox = hand_bbox['bbox_right'], hand_bbox['bbox_left']
            if self.aug_data:
                right_bbox, left_bbox = data_utils.jitter_bbox(right_bbox), data_utils.jitter_bbox(left_bbox)
            
            def crop_and_pad(img, bbox):
                if bbox is None: return img, np.zeros(4)
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
                x_mid, y_mid, size = (x0+x1)//2, (y0+y1)//2, max(bbox[2], bbox[3])
                img_crop = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [x_mid, y_mid, size, size], 1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0]
                return img_crop.transpose(2,0,1), np.array([x0, y0, x1, y1])
            r_img, r_bbox = crop_and_pad(img, right_bbox)
            l_img, l_bbox = crop_and_pad(img, left_bbox)
            norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
            norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

            img_ds = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [args.img_res/2, args.img_res/2, args.img_res, args.img_res], 1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0].transpose(2,0,1)
            img_ds = torch.from_numpy(img_ds).float()
            norm_img = self.normalize_img(img_ds)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        inputs["r_img"] = norm_r_img
        inputs["l_img"] = norm_l_img
        meta_info["imgname"] = imgname

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k

        if args.pos_enc is not None: # compute positional encoding for different pixels
            L = args.n_freq_pos_enc
            
            # center of left & right image
            r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
            l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
            r_angle_x, r_angle_y = np.arctan2(r_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(r_center[1]-intrx[1,2], intrx[1,1])
            l_angle_x, l_angle_y = np.arctan2(l_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(l_center[1]-intrx[1,2], intrx[1,1])
            r_angle, l_angle = np.array([r_angle_x, r_angle_y]), np.array([l_angle_x, l_angle_y])
            r_center_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1)*r_angle.reshape(1,-1)), np.cos(2**np.arange(L).reshape(-1,1)*r_angle.reshape(1,-1))], axis=-1).flatten()
            l_center_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1)*l_angle.reshape(1,-1)), np.cos(2**np.arange(L).reshape(-1,1)*l_angle.reshape(1,-1))], axis=-1).flatten()
            r_center_pos_enc = torch.from_numpy(r_center_pos_enc).float()
            l_center_pos_enc = torch.from_numpy(l_center_pos_enc).float()
            inputs['r_center_pos_enc'], inputs['l_center_pos_enc'] = r_center_pos_enc, l_center_pos_enc

            # corners of left & right image
            r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
            l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
            r_corner = np.stack([r_corner[:,0]-intrx[0,2], r_corner[:,1]-intrx[1,2]], axis=-1)
            l_corner = np.stack([l_corner[:,0]-intrx[0,2], l_corner[:,1]-intrx[1,2]], axis=-1)
            r_angle = np.arctan2(r_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten()
            l_angle = np.arctan2(l_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten()
            r_corner_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1)*r_angle.reshape(1,-1)), np.cos(2**np.arange(L).reshape(-1,1)*r_angle.reshape(1,-1))], axis=-1).flatten()
            l_corner_pos_enc = np.stack([np.sin(2**np.arange(L).reshape(-1,1)*l_angle.reshape(1,-1)), np.cos(2**np.arange(L).reshape(-1,1)*l_angle.reshape(1,-1))], axis=-1).flatten()
            r_corner_pos_enc = torch.from_numpy(r_corner_pos_enc).float()
            l_corner_pos_enc = torch.from_numpy(l_corner_pos_enc).float()
            inputs['r_corner_pos_enc'], inputs['l_corner_pos_enc'] = r_corner_pos_enc, l_corner_pos_enc

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")

        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        return inputs, targets, meta_info