import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.utils.dist_util import get_dist_info
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from importlib import import_module
import basicsr.loss as loss
import numpy as np
import matplotlib.pyplot as plt

import json

def create_video_model(opt):
    module = import_module('basicsr.models.archs.' + opt['model'].lower())
    model = module.make_model(opt)
    return model

metric_module = importlib.import_module('basicsr.metrics')

class VideoRestorationModel(BaseModel):
    def __init__(self, opt):
        super(VideoRestorationModel, self).__init__(opt)
        self.net_g = create_video_model(opt)
        self.net_g = self.model_to_device(self.net_g)
        self.n_sequence = opt['n_sequence']
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
            print("load_model", load_path)
        if self.is_train:
            self.init_training_settings()
        self.loss = loss.L1BaseLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def init_training_settings(self):
        self.net_g.train()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def model_to_device(self, net):
        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        train_opt['optim_g'].pop('type')
        self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                            **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    
    # method to feed the data to the model.
    def feed_data(self, data):
        lq, gt, _, _ = data
        self.lq = lq.to(self.device).half()
        self.gt = gt.to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        with torch.cuda.amp.autocast():
            loss_dict = OrderedDict()
            loss_dict['l_pix'] = 0

            frame_num = self.lq.shape[1]
            k_cache, v_cache = None, None
            for j in range(frame_num):
                target_g_images = self.gt[:, j, :, :, :]
                current_input = self.lq[:, j,:, :, :].unsqueeze(1)
                pre_input = self.lq[:, j if j == 0 else j-1, :, :, :].unsqueeze(1)
                
                input = torch.concat([pre_input, current_input], dim=1)                
                (out_g, k_cache, v_cache) = self.net_g(input, k_cache, v_cache)

                l_pix = self.loss(out_g, target_g_images)
                loss_dict['l_pix'] += l_pix

        # normalize w.r.t. total frames seen.
        loss_dict['l_pix'] /= frame_num
        l_total = loss_dict['l_pix'] + 0 * sum(p.sum() for p in self.net_g.parameters())
        loss_dict['l_pix'] = loss_dict['l_pix']

        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        # do gradient clipping to avoid larger updates.
        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.outputs_list = []
            self.gt_lists = []
            self.lq_lists = []
            frame_num = self.lq.shape[1]
            k_cache, v_cache = None, None
            for j in range(frame_num):
                target_g_images = self.gt[:, j, :, :, :]    
                current_input = self.lq[:, j,:, :, :].unsqueeze(1)
                pre_input = self.lq[:, j if j == 0 else j-1, :, :, :].unsqueeze(1)
                input = torch.concat([pre_input, current_input], dim=1)
                out_g, k_cache, v_cache = self.net_g(input.float(), 
                                                     k_cache, 
                                                     v_cache)
                self.outputs_list.append(out_g)
                self.gt_lists.append(target_g_images)
                self.lq_lists.append(self.lq[:, j,:, :, :])
        self.net_g.train()
    
    def non_cached_test(self):
        # proxy to the actual scores to save time.
        self.net_g.eval()
        with torch.no_grad():
            k_cache, v_cache = None, None
            pred, _, _, _ = self.net_g(self.lq.float(), k_cache, v_cache)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        import os
        return self.nondist_validation(dataloader, current_iter, 
                                       tb_logger, save_img, 
                                       rgb2bgr, use_image)
    
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')
        cnt = 0
        
        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            folder_name, img_name = val_data[len(val_data)-1][0][0].split('.')
            self.feed_data(val_data)
            self.test()

            for temp_i  in range(len(self.outputs_list)):
                sr_img = tensor2img(self.outputs_list[temp_i], rgb2bgr=rgb2bgr)
                gt_img = tensor2img(self.gt_lists[temp_i], rgb2bgr=rgb2bgr)
                lq_img = tensor2img(self.lq_lists[temp_i], rgb2bgr=rgb2bgr)

                if save_img:
                    # if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                            folder_name,
                                            f'{img_name}_frame{temp_i}_res.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                            folder_name,
                                            f'{img_name}_frame{temp_i}_gt.png')
                    
                    save_lq_img_path = osp.join(self.opt['path']['visualization'],
                    folder_name,
                    f'{img_name}_frame{temp_i}_lq.png')
                        
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    imwrite(lq_img, save_lq_img_path)

                if with_metrics:
                    # calculate metrics
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    if use_image:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(sr_img, gt_img, **opt_)
                    else:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(self.outputs_list[temp_i], self.gt_lists[temp_i], **opt_)

                cnt += 1
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Test {img_name}')
        
        if rank == 0:
            pbar.close()
            
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, tb_logger):
        log_str = f'Validation,\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        # pick the current frame.
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:,1,:,:,:].detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[:,1,:,:,:].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)