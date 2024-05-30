#!/usr/bin/env python3
import os
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from config.common import Config
from config.options import BaseOptions
import pickle as pkl
from utils.basic_utils import Basic_Utils
import yaml
import scipy.io as scio
import scipy.misc
from glob import glob
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
import random

try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey
import math
import time

# for get depth_filling function
config_fill = Config(ds_name='ycb')
bs_utils_fill = Basic_Utils(config_fill)


class Dataset():

    def __init__(self, dataset_name, cls_type="duck", DEBUG=False):
        self.DEBUG = DEBUG
        self.opt = BaseOptions().parse()
        self.opt.lm_no_render = False
        self.opt.lm_no_fuse = False
        self.config = Config(ds_name='linemod', cls_type=self.opt.linemod_cls)
        self.bs_utils = Basic_Utils(self.config)
        self.dataset_name = dataset_name
        self.xmap = np.array([[j for i in range(self.opt.width)] for j in range(self.opt.height)])
        self.ymap = np.array([[i for i in range(self.opt.width)] for j in range(self.opt.height)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])

        self.cls_type = cls_type

        self.cls_id = self.config.cls_id
        print("cls_id in lm_dataset.py", self.cls_id)
        self.root = self.config.lm_root
        self.cls_root = self.config.cls_root
        self.rng = np.random
        meta_file = open(os.path.join(self.cls_root, 'gt.yml'), "r")
        self.meta_lst = yaml.safe_load(meta_file)
        if dataset_name == 'train':
            self.add_noise = True
            real_img_pth = self.config.train_path
            self.real_lst = self.bs_utils.read_lines(real_img_pth)

            if not self.opt.lm_no_render:
                self.rnd_lst = self.bs_utils.read_lines(self.config.render_files)
                # rnd_img_ptn = self.config.render_path
                # self.rnd_lst = glob(rnd_img_ptn)
            # Remove render images
            else:
                self.rnd_lst = []

            print("render data length: ", len(self.rnd_lst))
            if len(self.rnd_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without rendered data will hurt model performance \n"
                warning += "Please generate rendered data from https://github.com/ethnhe/raster_triangle.\n"
                print(warning)

            if not self.opt.lm_no_fuse:
                self.fuse_lst = self.bs_utils.read_lines(self.config.fuse_files)
                # fuse_img_ptn = self.config.fuse_path
                # self.fuse_lst = glob(fuse_img_ptn)
            # Remove fuse images
            else:
                self.fuse_lst = []
            print("fused data length: ", len(self.fuse_lst))
            if len(self.fuse_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without fused data will hurt model performance \n"
                warning += "Please generate fused data from https://github.com/ethnhe/raster_triangle.\n"
                print(warning)
            # self.rnd_lst = random.sample(self.rnd_lst, 2000)
            # self.fuse_lst = random.sample(self.fuse_lst, 2000)
            self.all_lst = self.real_lst + self.rnd_lst + self.fuse_lst
            self.minibatch_per_epoch = len(self.all_lst) // self.opt.mini_batch_size
        else:
            self.add_noise = False
            tst_img_pth = self.config.test_path
            self.tst_lst = self.bs_utils.read_lines(tst_img_pth)
            self.all_lst = self.tst_lst
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))

    # def fill_missing(
    #     self, dpt, cam_scale, scale_2_80m, fill_type='multiscale',
    #         extrapolate=False, show_process=False, blur_type='bilateral'
    # ):
    #     dpt = dpt / cam_scale * scale_2_80m
    #     projected_depth = dpt.copy()
    #     if fill_type == 'fast':
    #         final_dpt = bs_utils_fill.fill_in_fast(
    #             projected_depth, extrapolate=extrapolate, blur_type=blur_type,
    #             # max_depth=2.0
    #         )
    #     elif fill_type == 'multiscale':
    #         final_dpt, process_dict = depth_map_utils.fill_in_multiscale(
    #             projected_depth, extrapolate=extrapolate, blur_type=blur_type,
    #             show_process=show_process,
    #             max_depth=3.0
    #         )
    #     else:
    #         raise ValueError('Invalid fill_type {}'.format(fill_type))
    #     dpt = final_dpt / scale_2_80m * cam_scale
    #     return dpt
    def real_syn_gen(self, real_ratio=0.3):
        if len(self.rnd_lst + self.fuse_lst) == 0:
            real_ratio = 1.0
        if self.rng.rand() < real_ratio:  # real
            n_imgs = len(self.real_lst)
            idx = self.rng.randint(0, n_imgs)
            pth = self.real_lst[idx]
            return pth
        else:
            if len(self.fuse_lst) > 0 and len(self.rnd_lst) > 0:
                fuse_ratio = 0.4
            elif len(self.fuse_lst) == 0:
                fuse_ratio = 0.
            else:
                fuse_ratio = 1.
            if self.rng.rand() < fuse_ratio:
                idx = self.rng.randint(0, len(self.fuse_lst))
                pth = self.fuse_lst[idx]
            else:
                idx = self.rng.randint(0, len(self.rnd_lst))
                pth = self.rnd_lst[idx]
            return pth

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand() * (hi - lo) + lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1 - 0.25, 1 + .25)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1 - .15, 1 + .15)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.cls_root, "depth", real_item + '.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.cls_root, "mask", real_item + '.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label < 255).astype(rgb.dtype)
        # bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        if len(bk_label.shape) > 2:
            bk_label = bk_label[:, :, 0]
        # Add pseudo-background
        with np.load(os.path.join(self.cls_root, "pseudo_nrm_angles/{}.npz".format(real_item))) as data:
            angles = data['angles']

            # convert angles and signed angles to image range (0~255)
            sed_angles = self.scale_pseudo(angles)

            sed_angles = Image.fromarray(np.uint8(sed_angles))

            back = np.array(sed_angles)[:, :, :3] * bk_label[:, :, None]

        # Add real-RGB background
        # with Image.open(os.path.join(self.cls_root, "rgb", real_item+'.png')) as ri:
        #     back_r = np.array(ri)[:, :, :3] * bk_label[:, :, None]

        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        if self.rng.rand() < 0.6:
            msk_back = (labels <= 0).astype(rgb.dtype)
            msk_back = np.repeat(msk_back[:, :, None], 3, 2)
            rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
              dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def scale_pseudo(self, pseudo):

        # Scale the pseudo angles and signed angles to image range (0 ~ 255)
        pseudo[:, :, 0][pseudo[:, :, 0] == 360] = 255
        pseudo[:, :, 0][pseudo[:, :, 0] < 255] = (pseudo[:, :, 0][pseudo[:, :, 0] < 255] - pseudo[:, :, 0][pseudo[:, :, 0] < 255].min()) * (254 / (pseudo[:, :, 0][pseudo[:, :, 0] < 255].max() - pseudo[:, :, 0][pseudo[:, :, 0] < 255].min()))
        pseudo[:, :, 1][pseudo[:, :, 1] == 360] = 255
        pseudo[:, :, 1][pseudo[:, :, 1] < 255] = (pseudo[:, :, 1][pseudo[:, :, 1] < 255] - pseudo[:, :, 1][pseudo[:, :, 1] < 255].min()) * (254 / (pseudo[:, :, 1][pseudo[:, :, 1] < 255].max() - pseudo[:, :, 1][pseudo[:, :, 1] < 255].min()))
        pseudo[:, :, 2][pseudo[:, :, 2] == 360] = 255
        pseudo[:, :, 2][pseudo[:, :, 2] < 255] = (pseudo[:, :, 2][pseudo[:, :, 2] < 255] - pseudo[:, :, 2][pseudo[:, :, 2] < 255].min()) * (254 / (pseudo[:, :, 2][pseudo[:, :, 2] < 255].max() - pseudo[:, :, 2][pseudo[:, :, 2] < 255].min()))

        # pseudo[:,:,0][pseudo[:,:,0]==360] = 255
        # pseudo[:,:,0][pseudo[:,:,0]<255] = pseudo[:,:,0][pseudo[:,:,0]<255]*254.0/180.0
        # pseudo[:,:,1][pseudo[:,:,1]==360] = 255
        # pseudo[:,:,1][pseudo[:,:,1]<255] = pseudo[:,:,1][pseudo[:,:,1]<255]*254.0/180.0
        # pseudo[:,:,2][pseudo[:,:,2]==360] = 255
        # pseudo[:,:,2][pseudo[:,:,2]<255] = pseudo[:,:,2][pseudo[:,:,2]<255]*254.0/180.0

        return pseudo

    def get_item(self, item_name):

        if ".npz" in item_name:
            # item_name_full = os.path.join(self.config.lm_root, item_name)
            data = np.load(item_name)
            dpt_mm = data['depth'] * 1000.
            angles = data['angles']  # data['rgb'] actually contains pseudo angles image with background
            # convert angles and signed angles to image range (0~255)
            sed_angles = self.scale_pseudo(angles)
            sed_angles = Image.fromarray(np.uint8(sed_angles))

            cam_scale = 1000.0
            nrm_angles = np.float32(sed_angles)
            labels = data['mask']
            K = data['K']
            RT = data['RT']
            rnd_typ = data['rnd_typ']
            if rnd_typ == "fuse":
                labels = (labels == self.cls_id).astype("uint8")
            else:
                labels = (labels > 0).astype("uint8")
        else:

            with Image.open(os.path.join(self.cls_root, "depth/{}.png".format(item_name))) as di:
                dpt_mm = np.array(di)
            with Image.open(os.path.join(self.cls_root, "mask/{}.png".format(item_name))) as li:
                labels = np.array(li)
                labels = (labels > 0).astype("uint8")
            # with Image.open(os.path.join(self.cls_root, "pseudo_angles/{}.png".format(item_name))) as ri:

            with np.load(os.path.join(self.cls_root, "pseudo_nrm_angles/{}.npz".format(item_name))) as data:
                angles = data['angles']

                # convert angles and signed angles to image range (0~255)
                sed_angles = self.scale_pseudo(angles)
                sed_angles = Image.fromarray(np.uint8(sed_angles))

                if self.add_noise:
                    sed_angles = self.trancolor(sed_angles)
                nrm_angles = np.array(sed_angles)[:, :, :3]

            meta = self.meta_lst[int(item_name)]
            if self.cls_id == 2:
                for i in range(0, len(meta)):
                    if meta[i]['obj_id'] == 2:
                        meta = meta[i]
                        break
            else:
                meta = meta[0]
            R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
            T = np.array(meta['cam_t_m2c']) / 1000.0
            RT = np.concatenate((R, T[:, None]), axis=1)
            rnd_typ = 'real'
            K = self.config.intrinsic_matrix["linemod"]
            cam_scale = 1000.0
            dpt_mm = bs_utils_fill.fill_missing(dpt_mm, cam_scale, 1)

        dpt_mm = dpt_mm.copy().astype(np.uint16)
        min_val = np.min(dpt_mm)
        max_val = np.max(dpt_mm)
        normalized_depth_image = (255 * ((dpt_mm - min_val) / (max_val - min_val))).astype(np.uint8)

        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        dpt_m = dpt_mm.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0

        if len(labels.shape) > 2:
            labels = labels[:, :, 0]
        rgb_labels = labels.copy()

        if self.add_noise and rnd_typ != 'real':
            if rnd_typ == 'render' or self.rng.rand() < 0.8:
                nrm_angles = self.rgb_add_noise(nrm_angles)
                rgb_labels = labels.copy()
                msk_dp = dpt_mm > 1e-6
                nrm_angles, dpt_mm = self.add_real_back(nrm_angles, rgb_labels, dpt_mm, msk_dp)
            if self.rng.rand() > 0.8:
                nrm_angles = self.rgb_add_noise(nrm_angles)

        msk_dp = dpt_mm > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > self.opt.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.opt.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.opt.n_sample_points - len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        nrm_angles_pt = nrm_angles.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        choose = np.array([choose])
        # cld_angle_nrm = cld.transpose(1, 0)
        cld_angle_nrm = np.concatenate((cld, nrm_angles_pt, nrm_pt), axis=1).transpose(1, 0)

        RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(
            cld, labels_pt, RT
        )

        h, w = self.opt.height, self.opt.width

        nrm_angles = np.transpose(nrm_angles, (2, 0, 1))  # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(5):
            scale = pow(2, i + 1)
            nh, nw = h // pow(2, i + 1), w // pow(2, i + 1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys * scale, xs * scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)

        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0)
            for ii, item in enumerate(xyz_lst)
        }

        rgb_ds_sr = [4, 8, 16, 32]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
            cld = sub_pts

        # UpSample stage
        n_up_layers = 3
        rgb_up_sr = [16, 8, 4]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

        # nrm_angles = nrm_angles.transpose(1,2,0).astype(np.uint8)
        dpt_map_m = np.repeat(np.expand_dims(normalized_depth_image, axis=0), 3, axis=0)

        if ".npz" in item_name:
            item_name = item_name.split('/')[-1].split('.')[0]
        item_dict = dict(
            img_id=np.uint8(item_name),
            nrm_angles=nrm_angles.astype(np.uint8),  # [c, h, w]
            cld_angle_nrm=cld_angle_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            # dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
            dpt_map_m=dpt_map_m.astype(np.uint8),  # [h, w]
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
        )
        '''
        img_id: 数据项的标识符（可能是图像文件的文件名或者其他标识符）。
        nrm_angles: 角度信息的图像表示，以uint8类型保存。
        cld_angle_nrm: 包含点云、角度信息和法向量的数组，以float32类型保存。
        choose: 选择的点的索引，以int32类型保存。
        labels: 标签信息，以int32类型保存。
        rgb_labels: 标签的RGB表示，以int32类型保存。
        dpt_map_m: 深度图的归一化坐标，以float32类型保存。
        RTs: 位姿信息，以float32类型保存。
        kp_targ_ofst: 关键点的目标偏移，以float32类型保存。
        ctr_targ_ofst: 中心点的目标偏移，以float32类型保存。
        cls_ids: 类别标识符，以int32类型保存。
        ctr_3ds: 中心点的三维坐标，以float32类型保存。
        kp_3ds: 关键点的三维坐标，以float32类型保存。
        '''
        item_dict.update(inputs)
        # if self.DEBUG:
        #     extra_d = dict(
        #         dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
        #         cam_scale=np.array([cam_scale]).astype(np.float32),
        #         K=K.astype(np.float32),
        #     )5
        #     item_dict.update(extra_d)
        #     item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

    def get_pose_gt_info(self, cld, labels, RT):
        RTs = np.zeros((self.config.n_objects, 3, 4))
        kp3ds = np.zeros((self.config.n_objects, self.opt.n_keypoints, 3))
        ctr3ds = np.zeros((self.config.n_objects, 3))
        cls_ids = np.zeros((self.config.n_objects, 1))
        kp_targ_ofst = np.zeros((self.opt.n_sample_points, self.opt.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((self.opt.n_sample_points, 3))
        for i, cls_id in enumerate([1]):
            RTs[i] = RT
            r = RT[:, :3]
            t = RT[:, 3]

            ctr = self.bs_utils.get_ctr(self.cls_type, ds_type="linemod")[:, None]
            ctr = np.dot(ctr.T, r.T) + t
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0 * ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([1])

            self.minibatch_per_epoch = len(self.all_lst) // self.opt.mini_batch_size
            if self.opt.n_keypoints == 8:
                kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(self.opt.n_keypoints)
            kps = self.bs_utils.get_kps(
                self.cls_type, kp_type=kp_type, ds_type='linemod'
            )
            kps = np.dot(kps, r.T) + t
            kp3ds[i] = kps

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0 * kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):

        if self.dataset_name == 'train':
            item_name = self.real_syn_gen()
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            return data
        else:
            item_name = self.all_lst[idx]
            return self.get_item(item_name)
