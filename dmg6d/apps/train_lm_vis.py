from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import time
import tqdm
import shutil
import resource
import numpy as np
from cv2 import imshow, waitKey
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR, CyclicLR
from utils import scheduler
import torch.backends.cudnn as cudnn
# from ignite.handlers import param_scheduler as ps

from config.options import BaseOptions
from config.common import Config, ConfigRandLA

import models.pytorch_utils as pt_utils
from models.SwinDePose import SwinDePose
from models.loss import OFLoss, FocalLoss
from utils.pvn3d_eval_utils_kpls import TorchEval
from utils.basic_utils import Basic_Utils
import datasets.linemod.linemod_dataset_vis as dataset_desc


from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model
from apex import amp


    
# get options
opt = BaseOptions().parse()

config = Config(ds_name=opt.dataset_name, cls_type=opt.linemod_cls)
bs_utils = Basic_Utils(config)

# create log folders 
if not os.path.exists(opt.log_eval_dir):
    os.makedirs(opt.log_eval_dir)
if not os.path.exists(opt.save_checkpoint):
    os.makedirs(opt.save_checkpoint)
if not os.path.exists(opt.log_traininfo_dir):
    os.makedirs(opt.log_traininfo_dir)


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

color_lst = [(0, 0, 0)]
for i in range(config.n_objects):
    col_mul = (255 * 255 * 255) // (i+1)
    color = (col_mul//(255*255), (col_mul//255) % 255, col_mul % 255)
    color_lst.append(color)




lr_clip = 1e-5
bnm_clip = 1e-2


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel) or \
                isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
        "amp": amp.state_dict(),
    }


def save_checkpoint(
        state,  filename="checkpoint"
):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)



def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint["epoch"] 
        print("epoch: ", epoch)
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        print("best_prec: ", best_prec)
        if model is not None and checkpoint["model_state"] is not None:
            ck_st = checkpoint['model_state']
            if 'module' in list(ck_st.keys())[0]:
                tmp_ck_st = {}
                for k, v in ck_st.items():
                    tmp_ck_st[k.replace("module.", "")] = v
                ck_st = tmp_ck_st
            model.load_state_dict(ck_st)
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        amp.load_state_dict(checkpoint["amp"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def view_labels(rgb_chw, img_id, obj_id, cld_cn, labels, K=config.intrinsic_matrix['linemod']):
    
    #rgb_hwc = np.transpose(rgb_chw[0].numpy(), (1, 2, 0)).astype("uint8").copy()
    import pdb;pdb.set_trace()
    rgb_hwc = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id)+'/rgb/'+img_id+'.png')
    cld_nc = np.transpose(cld_cn.numpy(), (1, 0)).copy()
    p2ds = bs_utils.project_p3d(cld_nc, 1.0, K).astype(np.int32)
    labels = labels.squeeze().contiguous().cpu().numpy()
    colors = []
    #h, w = rgb_hwc.shape[0], rgb_hwc.shape[1]
    #rgb_hwc = np.zeros((h, w, 3), "uint8")
    for lb in labels:
        if int(lb) == 0:
            c = (255, 255, 255)
        else:
            c = color_lst[int(lb)]
            #c = (0, 0, 0)
        colors.append(c)
    show = bs_utils.draw_p2ds(rgb_hwc, p2ds, 3, colors, 0.6)
    return show


def model_fn_decorator(
    criterion, criterion_of, test=False,
):
    teval = TorchEval()

    def model_fn(
        model, data, it=0, epoch=0, is_eval=False, is_test=False, finish_test=False,
        test_pose=False
    ):
        
        
        if is_eval:
            model.eval()
        with torch.set_grad_enabled(not is_eval):
            cu_dt = {}
            # device = torch.device('cuda:{}'.format(args.local_rank))
            for key in data.keys():
                
                img_id = data['img_id']
                if key!='img_id':
                    if data[key].dtype in [np.float32, np.uint8]:
                        cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
                    elif data[key].dtype in [np.int32, np.uint32]:
                        cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
                    elif data[key].dtype in [torch.uint8, torch.float32]:
                        cu_dt[key] = data[key].float().cuda()
                    elif data[key].dtype in [torch.int32, torch.int16]:
                        cu_dt[key] = data[key].long().cuda()
                    else:
                        cu_dt[key] = data[key].str().cuda()
                
            end_points = model(cu_dt)
            
            labels = cu_dt['labels']
            

            _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
            acc_rgbd = (cls_rgbd == labels).float().sum() / labels.numel()
            
            
            
            
            acc_dict = {
                'acc_rgbd': acc_rgbd.item(),
            }
            
            
            
            if is_test and test_pose:
                cld = cu_dt['cld_angle_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
                
                if not opt.test_gt:
                    # eval pose from point cloud prediction.
                    pred_RTs, gt_RTs, pred_kp, gt_kp, gt_ctr = teval.eval_pose_parallel_vis(
                        cld, img_id, cu_dt['nrm_angles'], cls_rgbd, end_points['pred_ctr_ofs'],
                        cu_dt['ctr_targ_ofst'], labels, epoch, cu_dt['cls_ids'],
                        cu_dt['RTs'], end_points['pred_kp_ofs'],
                        cu_dt['kp_3ds'], cu_dt['ctr_3ds'],
                        ds='linemod', obj_id=config.cls_id,
                        min_cnt=1, use_ctr_clus_flter=True, use_ctr=True,
                    )
                
                else:
                    
                    # test GT labels 
                    gt_ctr_ofs = cu_dt['ctr_targ_ofst'].unsqueeze(2).permute(0, 2, 1, 3)
                    gt_kp_ofs = cu_dt['kp_targ_ofst'].permute(0, 2, 1, 3)
                    add, adds, pred_kp, gt_kp, gt_ctr = teval.eval_pose_parallel_vis(
                        cld, cu_dt['img_id'], cu_dt['nrm_angles'], labels, end_points['pred_ctr_ofs'],
                        cu_dt['ctr_targ_ofst'], labels, epoch, cu_dt['cls_ids'],
                        cu_dt['RTs'], end_points['pred_kp_ofs'],
                        cu_dt['kp_3ds'], cu_dt['ctr_3ds'],
                        ds='linemod', obj_id=config.cls_id,
                        min_cnt=1, use_ctr_clus_flter=True, use_ctr=True
                    )
            if opt.eval_net:
                test_res = {
                'img_id':img_id,
                'pred_RTs':pred_RTs,
                'gt_RTs':gt_RTs,
                'cld':cu_dt['cld_angle_nrm'][:,0:3,:].cpu().numpy()
            }
   
        return (
            end_points, test_res
        )
        

    return model_fn


class Trainer(object):
    """
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    """

    def __init__(
        self,
        model,
        model_fn,
        optimizer,
        checkpoint_name="ckpt",
        lr_scheduler=None,
        bnm_scheduler=None,
        viz=None,
    ):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model,
            model_fn,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
        )

        self.checkpoint_name = checkpoint_name

        self.training_best, self.eval_best = {}, {}
        self.viz = viz

    def eval_epoch(self, d_loader, epoch,  is_test=False, test_pose=False):
        self.model.eval()

        if opt.eval_net:
            img_ids = []
            pred_RTs = []
            gt_RTs = []
            pred_kp = []
            gt_kp = []
            gt_ctr = []
            cld = []
        eval_dict = {}
        total_loss = 0.0
        count = 1
        for _, data in tqdm.tqdm(enumerate(d_loader)):
             
            count += 1
            self.optimizer.zero_grad()
            if opt.eval_net:
                
                eval_res, test_res = self.model_fn(
                self.model, data, is_eval=True, is_test=is_test, test_pose=test_pose
            )
            
            if opt.eval_net:
                img_ids.append(test_res['img_id'])
                pred_RTs.append(test_res['pred_RTs'])
                gt_RTs.append(test_res['gt_RTs'])
                cld.append(test_res['cld'])
            
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]
        
        if opt.eval_net:
            test_results={
            'img_ids':img_ids,
            'pred_RTs':pred_RTs,
            'gt_RTs':gt_RTs,
            'cld':cld
            }
        
        mean_eval_dict = {}
        acc_dict = {}
        for k, v in eval_res.items():
            per = 100 if 'acc' in k else 1
            mean_eval_dict[k] = np.array(v.cpu()).mean() * per
            if 'acc' in k:
                acc_dict[k] = v
        for k, v in mean_eval_dict.items():
            print(k, v)
        
        if is_test:
            if test_pose:
                self.model_fn(
                    self.model, data, is_eval=True, is_test=is_test, finish_test=True,
                    test_pose=test_pose
                )
            seg_res_fn = 'seg_res'
            for k, v in acc_dict.items():
                seg_res_fn += '_%s%.2f' % (k, v)
            with open(os.path.join(opt.log_eval_dir, seg_res_fn), 'w') as of:
                for k, v in acc_dict.items():
                    print(k, v, file=of)
        if opt.local_rank == 0:
            print(acc_dict)

            
        return eval_dict, test_results
        

    def train(
        self,
        start_it,
        start_epoch,
        n_epochs,
        train_loader,
        train_sampler,
        test_loader=None,
        best_loss=0.0,
        log_epoch_f=None,
        tot_iter=1
    ):
        """
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        print("Totally train %d iters per gpu." % tot_iter)

        # def is_to_eval(epoch, it):
        #     if it == 100:
        #         return True, 1
        #     wid = tot_iter // clr_div
        #     if (it // wid) % 2 == 1:
        #         eval_frequency = wid // 15
        #     else:
        #         eval_frequency = wid // 6
        #     to_eval = (it % eval_frequency) == 0
        #     return to_eval, eval_frequency

        # it = start_it
        # _, eval_frequency = is_to_eval(0, it)
        
        # Early stopping
        last_loss = 1e10
        patience = 7
        trigger_times = 0
        it = start_it
        for start_epoch in tqdm.tqdm(range(n_epochs)):
            
            if train_sampler is not None:
                train_sampler.set_epoch(start_epoch)
            # Reset numpy seed.
            # REF: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed()
            if log_epoch_f is not None:
                os.system("echo {} > {}".format(start_epoch, log_epoch_f))
            for batch in tqdm.tqdm(train_loader):
                self.model.train()

                self.optimizer.zero_grad()
                _, loss, res = self.model_fn(self.model, batch, it=it)

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(it)

                if self.bnm_scheduler is not None:
                    self.bnm_scheduler.step(it)

                it += 1
                
                if self.viz is not None:
                    self.viz.update("train", it, res)

                # eval_flag, eval_frequency = is_to_eval(start_epoch, it)
                
            
            if test_loader is not None:
                if opt.eval_net:
                    res, test_res = self.eval_epoch(test_loader, start_epoch)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    if opt.local_rank == 0:
                        save_checkpoint(
                            checkpoint_state(
                                self.model, self.optimizer, val_loss, start_epoch, it
                            ),
                            filename=self.checkpoint_name)
                # Early Stopping
                current_loss = val_loss
                if current_loss > last_loss:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print('Early Stopping!\n')
                        exit()
                else:
                    trigger_times = 0
                last_loss = current_loss
        
        return val_loss


def train():
    
    
    print("local_rank:", opt.local_rank)
    cudnn.benchmark = True
    if opt.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(opt.local_rank)
        torch.set_printoptions(precision=10)
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

    test_ds = dataset_desc.Dataset('test', cls_type=opt.linemod_cls)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=opt.test_mini_batch_size, shuffle=False,
        num_workers=opt.num_threads
    )

    rndla_cfg = ConfigRandLA
    model = SwinDePose(
        n_classes=config.n_objects, n_pts=opt.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=opt.n_keypoints
    )

    print("Number of model parameters: ", count_parameters(model))
    model = convert_syncbn_model(model)
    device = torch.device('cuda:{}'.format(opt.local_rank))
    print('local_rank:', opt.local_rank)
    model.to(device)
    # optimizer = optim.AdamW(
    #     model.parameters(), betas=(0.9, 0.999), lr=opt.lr, weight_decay=opt.weight_decay
    # )
    optimizer = optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
    )
    opt_level = opt.opt_level
    model, optimizer = amp.initialize(
        model, optimizer, opt_level=opt_level,
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    if opt.load_checkpoint is not None:
        
        checkpoint_status = load_checkpoint(
            model, optimizer, filename=opt.load_checkpoint
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
        if opt.eval_net:
            assert checkpoint_status is not None, "Failed loadding model."


    lr_scheduler = None

    bnm_lmbd = lambda it: max(
        opt.bn_momentum * opt.bn_decay ** (int(it * opt.mini_batch_size / opt.decay_step)),
        bnm_clip,
    )
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bnm_lmbd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    if opt.eval_net:
        model_fn = model_fn_decorator(
            FocalLoss(gamma=2), OFLoss(),
            opt.test, 
        )
    
    checkpoint_fd = opt.save_checkpoint

    trainer = Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name=os.path.join(checkpoint_fd, "%s" % opt.linemod_cls),
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
    )

    if opt.eval_net:
        start = time.time()
        eval_dict, test_results= trainer.eval_epoch(
            test_loader, opt.n_total_epoch, is_test=True, test_pose=opt.test_pose
        )
        end = time.time()
        print("\nUse time: ", end - start, 's')
        
        # save test results
        # img_ids = test_results['img_ids']
        # pred_RTs = test_results['pred_RTs']
        # gt_RTs = test_results['gt_RTs']
        
        # cld = test_results['cld']
        
        # np.save(os.path.join(opt.log_eval_dir, 'cld.npy'), cld) 
        # np.save(os.path.join(opt.log_eval_dir, 'pred_RTs.npy'), pred_RTs) 
        # np.save(os.path.join(opt.log_eval_dir, 'gt_RTs.npy'), gt_RTs) 
        # np.savetxt(os.path.join(opt.log_eval_dir, 'img_ids.txt'),img_ids)

        
        
    

if __name__ == "__main__":
    opt.world_size = opt.gpus * opt.nodes
    
    train()
