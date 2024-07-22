# copied from image_dataset.py


from typing import Optional, Tuple, List
import torch.utils
from hamer.datasets import HAMERDataModule, Dataset
from hamer.datasets.utils import get_example, expand_to_aspect_ratio
import cv2
import numpy as np
import torch
from yacs.config import CfgNode
import copy
import pickle
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256
DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
FLIP_KEYPOINT_PERMUTATION = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

intrx_path = '/data11/adityap9/projects/hamer/logs/debug/intrx.pkl'
with open(intrx_path, 'rb') as f:
    intrx = pickle.load(f)  # dict_keys(['ho3d', 'h2o3d', 'dexycb', 'freihand', 'interhand', 'rhd', 'mtc'])

def load_tars_as_webdataset(cfg: CfgNode, urls: str|List[str], train: bool,
        resampled=False,
        epoch_size=None,
        cache_dir=None,
        **kwargs) -> Dataset:
    """
    Loads the dataset from a webdataset tar file.
    """
    # url: /data01/adityap9/datasets/hamer/hamer_training_data/dataset_tars/interhand26m-train/{000000..001056}.tar

    IMG_SIZE = cfg.MODEL.IMAGE_SIZE
    BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
    MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
    STD = 255. * np.array(cfg.MODEL.IMAGE_STD)

    def split_data(source):
        for item in source:
            # dict_keys(['__key__', '__url__', 'data.pyd', 'jpg'])
            datas = item['data.pyd']
            for data in datas:
                if 'detection.npz' in item:
                    det_idx = data['extra_info']['detection_npz_idx']
                    mask = item['detection.npz']['masks'][det_idx]
                else:
                    mask = np.ones_like(item['jpg'][:,:,0], dtype=bool)
                yield {
                    '__key__': item['__key__'],
                    'jpg': item['jpg'],
                    'data.pyd': data,
                    'mask': mask,
                }

    def suppress_bad_kps(item, thresh=0.0):
        if thresh > 0:
            kp2d = item['data.pyd']['keypoints_2d']
            kp2d_conf = np.where(kp2d[:, 2] < thresh, 0.0, kp2d[:, 2])
            item['data.pyd']['keypoints_2d'] = np.concatenate([kp2d[:,:2], kp2d_conf[:,None]], axis=1)
        return item

    def filter_numkp(item, numkp=4, thresh=0.0):
        kp_conf = item['data.pyd']['keypoints_2d'][:, 2]
        return (kp_conf > thresh).sum() > numkp

    def filter_reproj_error(item, thresh=10**4.5):
        losses = item['data.pyd'].get('extra_info', {}).get('fitting_loss', np.array({})).item()
        reproj_loss = losses.get('reprojection_loss', None)
        return reproj_loss is None or reproj_loss < thresh

    def filter_bbox_size(item, thresh=1):
        bbox_size_min = item['data.pyd']['scale'].min().item() * 200.
        return bbox_size_min > thresh

    def filter_no_poses(item):
        return (item['data.pyd']['has_hand_pose'] > 0)

    def supress_bad_betas(item, thresh=3):
        has_betas = item['data.pyd']['has_betas']
        if thresh > 0 and has_betas:
            betas_abs = np.abs(item['data.pyd']['betas'])
            if (betas_abs > thresh).any():
                item['data.pyd']['has_betas'] = False
        return item

    def supress_bad_poses(item):
        has_hand_pose = item['data.pyd']['has_hand_pose']
        if has_hand_pose:
            hand_pose = item['data.pyd']['hand_pose']
            pose_is_probable = poses_check_probable(torch.from_numpy(hand_pose)[None, 3:], amass_poses_hist100_smooth).item()
            if not pose_is_probable:
                item['data.pyd']['has_hand_pose'] = False
        return item

    def poses_betas_simultaneous(item):
        # We either have both hand_pose and betas, or neither
        has_betas = item['data.pyd']['has_betas']
        has_hand_pose = item['data.pyd']['has_hand_pose']
        item['data.pyd']['has_betas'] = item['data.pyd']['has_hand_pose'] = np.array(float((has_hand_pose>0) and (has_betas>0)))
        return item

    def set_betas_for_reg(item):
        # Always have betas set to true
        has_betas = item['data.pyd']['has_betas']
        betas = item['data.pyd']['betas']

        if not (has_betas>0):
            item['data.pyd']['has_betas'] = np.array(float((True)))
            item['data.pyd']['betas'] = betas * 0
        return item

    def set_intrx(item): # check if this is correct
        # item: {'__key__', 'jpg', 'data.pyd', 'mask'}
        name = item['__key__']
        strings = name.split('/')
        dataset_name = strings[0].split('-')[0]
        if 'ho3d' in dataset_name or 'h2o3d' in dataset_name:
            index = '/'.join(strings[1:])
        elif 'dex' in dataset_name:
            index = strings[-2]
            dataset_name = 'dexycb'
        elif 'freihand' in dataset_name:
            index = str(int(strings[-1])%32560).zfill(8) # total unique images in freihand = 32560
        elif 'interhand' in dataset_name:
            index = '/'.join(strings[1:])
            dataset_name = 'interhand'
        elif 'rhd' in dataset_name:
            index = strings[-1]
        elif 'mtc' in dataset_name:
            index = strings[1]+'/'+strings[-1].split('_')[1]
        item['data.pyd']['intrx'] = intrx[dataset_name][index].astype(np.float32)
        return item
    def process_webdataset_tar_item(item, train, 
                                    augm_config=None, 
                                    MEAN=DEFAULT_MEAN, 
                                    STD=DEFAULT_STD, 
                                    IMG_SIZE=DEFAULT_IMG_SIZE,
                                    BBOX_SHAPE=None,
                                    use_skimage_antialias=False,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    pos_enc=None,
                                    use_gt_p=False,
                                    ):
        # Read data from item
        key = item['__key__']
        image = item['jpg']

        # write the image for testing
        # name = key.split('/')[-1]

        # cv2.imwrite(f'__test__/{name}_orig.png', image[:,:,::-1])
        # print(image.shape)
        # raise Exception
        data = item['data.pyd']
        mask = item['mask']

        keypoints_2d = data['keypoints_2d']
        keypoints_3d = data['keypoints_3d']
        center = data['center']
        scale = data['scale']
        hand_pose = data['hand_pose']
        betas = data['betas']
        right = data['right']
        has_hand_pose = data['has_hand_pose']
        has_betas = data['has_betas']
        # image_file = data['image_file']
        K = data['intrx'].copy()
        
        

        # if use_gt_p:
        #     K[0,2] = image.shape[1] / 2
        #     K[1,2] = image.shape[0] / 2

        # Process data
        orig_keypoints_2d = keypoints_2d.copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        if bbox_size < 1:
            breakpoint()


        mano_params = {'global_orient': hand_pose[:3],
                    'hand_pose': hand_pose[3:],
                    'betas': betas
                    }

        has_mano_params = {'global_orient': has_hand_pose,
                        'hand_pose': has_hand_pose,
                        'betas': has_betas
                        }

        mano_params_is_axis_angle = {'global_orient': True,
                                    'hand_pose': True,
                                    'betas': False
                                    }

        augm_config = copy.deepcopy(augm_config)
        # Crop image and (possibly) perform data augmentation
        img_rgba = np.concatenate([image, mask.astype(np.uint8)[:,:,None]*255], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size, trans, pos_emb, data_full = get_example(img_rgba,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    mano_params, has_mano_params,
                                                                                                    FLIP_KEYPOINT_PERMUTATION,
                                                                                                    IMG_SIZE, IMG_SIZE,
                                                                                                    MEAN, STD, train, right, augm_config,
                                                                                                    is_bgr=False, return_trans=True,
                                                                                                    use_skimage_antialias=use_skimage_antialias,
                                                                                                    border_mode=border_mode,
                                                                                                    pos_enc=pos_enc,
                                                                                                    K=K,
                                                                                                    )
        img_patch = img_patch_rgba[:3,:,:]
        mask_patch = (img_patch_rgba[3,:,:] / 255.0).clip(0,1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)
        # if key not in ImageDataset.key_dict:
        #     if "interhand" in key:
        #         if "interhand_x" not in ImageDataset.record_dict:
        #             ImageDataset.record_dict["interhand_x"] = []
        #             ImageDataset.record_dict["interhand_y"] = []
        #         angle_x, angle_y = pos_emb['center_angle'][0], pos_emb['center_angle'][1]
        #         ImageDataset.record_dict["interhand_x"].append(angle_x)
        #         ImageDataset.record_dict["interhand_y"].append(angle_y)
        #     elif "rhd" in key:
        #         if "rhd_x" not in ImageDataset.record_dict:
        #             ImageDataset.record_dict["rhd_x"] = []
        #             ImageDataset.record_dict["rhd_y"] = []
        #         angle_x, angle_y = pos_emb['center_angle'][0], pos_emb['center_angle'][1]
        #         ImageDataset.record_dict["rhd_x"].append(angle_x)
        #         ImageDataset.record_dict["rhd_y"].append(angle_y)
        #     elif "freihand" in key:
        #         if "freihand_x" not in ImageDataset.record_dict:
        #             ImageDataset.record_dict["freihand_x"] = []
        #             ImageDataset.record_dict["freihand_y"] = []
        #         angle_x, angle_y = pos_emb['center_angle'][0], pos_emb['center_angle'][1]
        #         ImageDataset.record_dict["freihand_x"].append(angle_x)
        #         ImageDataset.record_dict["freihand_y"].append(angle_y)
        #     elif "mtc" in key:
        #         if "mtc_x" not in ImageDataset.record_dict:
        #             ImageDataset.record_dict["mtc_x"] = []
        #             ImageDataset.record_dict["mtc_y"] = []
        #         angle_x, angle_y = pos_emb['center_angle'][0], pos_emb['center_angle'][1]
        #         ImageDataset.record_dict["mtc_x"].append(angle_x)
        #         ImageDataset.record_dict["mtc_y"].append(angle_y)
        #     else:
        #         raise Exception("Unknown dataset " + key)
        #     ImageDataset.key_dict.add(key)
        #     # if len(ImageDataset.key_dict) >= 50000:
        #         # save record_dict
        #     with open('record_dict.pkl', 'wb') as f:
        #         pickle.dump(ImageDataset.record_dict, f)
        #     # raise Exception("done")
        #     # ImageDataset.key_dict.add(key)
        # print(key)
        # print(image.shape)
        # print(K)
        # print(pos_emb)
        # print("=====================================", flush=True)
        full_size = max(image.shape[:2])
        scale_factor = IMG_SIZE / full_size
        mean_focal_length = 0.5 * (K[0,0] + K[1,1])
        focal_length = mean_focal_length * scale_factor

        item = {}

        item['img'] = img_patch
        item['mask'] = mask_patch

        if data_full is not None:
            item['img_full'] = data_full['img'][:3]
            item['box_center_full'] = data_full['box_center']
            item['box_size_full'] = data_full['box_size']
            crop_intrx = data_full['K_crop'].copy()
            # scale intrinsics by scale_factor
            crop_intrx[0,0] *= scale_factor
            crop_intrx[1,1] *= scale_factor
            crop_intrx[0,2] *= scale_factor
            crop_intrx[1,2] *= scale_factor
            item['K_crop'] = crop_intrx
            item['K'] = data['intrx'].copy()
            rescale_intrx = data['intrx'].copy()
            rescale_intrx[0,0] *= scale_factor
            rescale_intrx[1,1] *= scale_factor
            rescale_intrx[0,2] -= (image.shape[1]/2 - full_size/2)
            rescale_intrx[1,2] -= (image.shape[0]/2 - full_size/2)
            rescale_intrx[0,2] *= scale_factor
            rescale_intrx[1,2] *= scale_factor
            item['K_rescale'] = rescale_intrx
            crop_rescale_intrx = data_full['K_crop'].copy()
            factor = IMG_SIZE / data_full['box_size']
            crop_rescale_intrx[0,0] *= factor
            crop_rescale_intrx[1,1] *= factor
            crop_rescale_intrx[0,2] *= factor
            crop_rescale_intrx[1,2] *= factor
            item['K_crop_rescale'] = crop_rescale_intrx
        
        # item['img_og'] = image
        # item['mask_og'] = mask
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = center.copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['mano_params'] = mano_params
        item['has_mano_params'] = has_mano_params
        item['mano_params_is_axis_angle'] = mano_params_is_axis_angle
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = key
        item['right'] = right
        # item['idx'] = idx
        item['focal_length'] = focal_length
        if pos_emb is not None:
            item['pos_emb'] = pos_emb

        return item

    def adapt_for_arctic(item):
        """
        dict_keys(['img', 'mask', 'img_full', 'box_center_full', 'box_size_full', 'K_crop', 'K', 'K_rescale', 'K_crop_rescale', 'keypoints_2d', 'keypoints_3d', 'orig_keypoints_2d', 'box_center', 'box_size', 'img_size', 'mano_params', 'has_mano_params', 'mano_params_is_axis_angle', '_scale', '_trans', 'imgname', 'right', 'focal_length', 'pos_emb', '__key__'])
        """
        img = item['img']
        keypoints_2d = item['keypoints_2d']
        keypoints_3d = item['keypoints_3d']
        mano_params = item['mano_params']
        imgname = item['imgname']
        pos_emb = item['pos_emb']
        
    # Load the dataset
    # if epoch_size is not None:
    #     resampled = True    # this can cause confusion
    #corrupt_filter = lambda sample: (sample['__key__'] not in CORRUPT_KEYS)
    import webdataset as wds
    ### this is where webdataset is created
    dataset = wds.WebDataset(urls,
                            nodesplitter=wds.split_by_node,
                            shardshuffle=True,
                            resampled=resampled,
                            cache_dir=cache_dir,
                            ) #.select(corrupt_filter)

    # # check keys, for debugging only
    # all_keys = []
    # for i, item in enumerate(dataset): # doesn't have length, iterates indefinitely
    #     all_keys.append(item['__key__'])
    #     if i == 100: break
    # print (len(all_keys), len(set(all_keys)))
    # print(all_keys)
    # raise Exception

    if train:
        dataset = dataset.shuffle(100)
    dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')

    # Process the dataset
    dataset = dataset.compose(split_data)

    # Filter/clean the dataset
    SUPPRESS_KP_CONF_THRESH = cfg.DATASETS.get('SUPPRESS_KP_CONF_THRESH', 0.0)
    SUPPRESS_BETAS_THRESH = cfg.DATASETS.get('SUPPRESS_BETAS_THRESH', 0.0)
    SUPPRESS_BAD_POSES = cfg.DATASETS.get('SUPPRESS_BAD_POSES', False)
    POSES_BETAS_SIMULTANEOUS = cfg.DATASETS.get('POSES_BETAS_SIMULTANEOUS', False)
    BETAS_REG = cfg.DATASETS.get('BETAS_REG', False)
    FILTER_NO_POSES = cfg.DATASETS.get('FILTER_NO_POSES', False)
    FILTER_NUM_KP = cfg.DATASETS.get('FILTER_NUM_KP', 4)
    FILTER_NUM_KP_THRESH = cfg.DATASETS.get('FILTER_NUM_KP_THRESH', 0.0)
    FILTER_REPROJ_THRESH = cfg.DATASETS.get('FILTER_REPROJ_THRESH', 0.0)
    FILTER_MIN_BBOX_SIZE = cfg.DATASETS.get('FILTER_MIN_BBOX_SIZE', 0.0)
    if SUPPRESS_KP_CONF_THRESH > 0:
        dataset = dataset.map(lambda x: suppress_bad_kps(x, thresh=SUPPRESS_KP_CONF_THRESH))
    if SUPPRESS_BETAS_THRESH > 0:
        dataset = dataset.map(lambda x: supress_bad_betas(x, thresh=SUPPRESS_BETAS_THRESH))
    if SUPPRESS_BAD_POSES:
        dataset = dataset.map(lambda x: supress_bad_poses(x))
    if POSES_BETAS_SIMULTANEOUS:
        dataset = dataset.map(lambda x: poses_betas_simultaneous(x))
    if FILTER_NO_POSES:
        dataset = dataset.select(lambda x: filter_no_poses(x))
    if FILTER_NUM_KP > 0:
        dataset = dataset.select(lambda x: filter_numkp(x, numkp=FILTER_NUM_KP, thresh=FILTER_NUM_KP_THRESH))
    if FILTER_REPROJ_THRESH > 0:
        dataset = dataset.select(lambda x: filter_reproj_error(x, thresh=FILTER_REPROJ_THRESH))
    if FILTER_MIN_BBOX_SIZE > 0:
        dataset = dataset.select(lambda x: filter_bbox_size(x, thresh=FILTER_MIN_BBOX_SIZE))
    if BETAS_REG:
        dataset = dataset.map(lambda x: set_betas_for_reg(x))       # NOTE: Must be at the end
    # add intrinsics
    dataset = dataset.map(lambda x: set_intrx(x))

    use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
    border_mode = {
        'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE,
    }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

    # Process the dataset further
    dataset = dataset.map(lambda x: process_webdataset_tar_item(x, train,
                                                    augm_config=cfg.DATASETS.CONFIG,
                                                    MEAN=MEAN, STD=STD, IMG_SIZE=IMG_SIZE,
                                                    BBOX_SHAPE=BBOX_SHAPE,
                                                    use_skimage_antialias=use_skimage_antialias,
                                                    border_mode=border_mode,
                                                    pos_enc=cfg.MODEL.get('POS_ENC', None),
                                                    use_gt_p=cfg.MODEL.get('USE_GT_P', False),
                                                    ))
    dataset = dataset.map(lambda x: adapt_for_arctic(x))
    # process dataset to fit the arctic format
    
    if epoch_size is not None:
        dataset = dataset.with_epoch(epoch_size)

    return dataset