from __future__ import division
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
selected_gpu_ids = [0,1]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in selected_gpu_ids)

import sys
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from config import config
from datasets import get_train_loader, OccScanNet
from nets import Network
from utils.init_func import init_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp

def sequence_loss(predictions,label,label_weight,local_rank):

    n_predictions=len(predictions)

    cri_weights = torch.FloatTensor([config.empty_loss_weight, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none',
                                    weight=cri_weights).cuda(local_rank)
    seq_loss=0
    for i in range(n_predictions):
        iweight=0.8**(n_predictions-i-1)

        selectindex = torch.nonzero(label_weight.view(-1)).view(-1).cuda(local_rank)
        filterLabel = torch.index_select(label.reshape(-1), 0, selectindex).cuda(local_rank)
        filterOutput = torch.index_select(predictions[i].permute(
            0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
        loss_semantic = criterion(filterOutput, filterLabel)
        loss_semantic = torch.mean(loss_semantic)

        seq_loss+=loss_semantic*iweight


    return seq_loss

parser = argparse.ArgumentParser()

port = str(int(float(time.time())) % 20)
os.environ['MASTER_PORT'] = str(10097 + int(port))



def main():

    args = parser.parse_args()
    args.nprocs = len(selected_gpu_ids)

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    
def main_worker(local_rank, nprocs, args):


        engine = Engine(custom_parser=parser)
        cudnn.benchmark = True
        seed = config.seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:21266',
                                world_size=args.nprocs,
                                rank=local_rank)

        traindataset = get_train_loader(engine, OccScanNet)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            traindataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            traindataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            sampler=train_sampler,
        )
        # Per-rank steps per epoch (DistributedSampler splits data across GPUs)
        niters_per_epoch = len(train_loader)

        BatchNorm3d=torch.nn.BatchNorm3d

        model = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                        norm_layer=BatchNorm3d)
        init_weight(model.business_layer, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, config.bn_eps, config.bn_momentum,
                    mode='fan_in')  # , nonlinearity='relu')

        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr  # * engine.world_size

        for param in model.backbone.encoder.parameters():
            param.requires_grad = False


        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
       
        # lr schedule: total steps = epochs * steps per rank per epoch
        total_iteration = config.nepochs * niters_per_epoch
        lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

        torch.cuda.set_device(local_rank)
        model.cuda(local_rank)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        

        engine.register_state(dataloader=train_loader, model=model,
                              optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()
        tb_writer = SummaryWriter(config.tb_dir) if local_rank == 0 else None

        model.train()
        print('begin train')

        for epoch in range(engine.state.epoch, config.nepochs):
            train_loader.sampler.set_epoch(epoch)
           
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)

            sum_loss = 0
            sum_sem = 0
            sum_com = 0
            sum_rest = 0
            sum_sketch = 0
            sum_sketch_gsnn = 0
            sum_kld = 0
            sum_sketch_raw = 0

            torch.cuda.empty_cache()
            for idx in pbar:

                optimizer.zero_grad()
                engine.update_iteration(epoch, idx)

                minibatch = next(iter(dataloader))
                img = minibatch['data']
                label = minibatch['label']
                label_sc = minibatch['label_sc']
                label_weight = minibatch['label_weight']
                tsdf = minibatch['tsdf']
                depth_mapping_3d = minibatch['depth_mapping_3d']

                img = img.cuda(non_blocking=True).cuda(local_rank)
                label = label.cuda(non_blocking=True).cuda(local_rank)
                label_sc = label_sc.cuda(non_blocking=True).cuda(local_rank)
                tsdf = tsdf.cuda(non_blocking=True).cuda(local_rank)
                label_weight = label_weight.cuda(non_blocking=True).cuda(local_rank)
                depth_mapping_3d = depth_mapping_3d.cuda(non_blocking=True).cuda(local_rank)

                # # 1. 有效 mask
                # gt_mask = (label != 0) & (label != 255)
                # mapping_mask = (depth_mapping_3d != 307200)
                #
                # # 2. 获取有效体素坐标
                # gt_coords = gt_mask.nonzero(as_tuple=False)  # [N_gt, 3]
                # mapping_coords = mapping_mask.nonzero(as_tuple=False)  # [N_map, 3]
                #
                # # 3. 转为一维索引（等价于 np.ravel_multi_index）
                # _, D, H, W = label.shape
                # gt_flat_idx = gt_coords[:, 1] * (H * W) + gt_coords[:, 2] * W + gt_coords[:, 3]
                # mapping_flat_idx = mapping_coords
                #
                # # 4. 求交集（torch 没有直接的 intersect1d，需要这样做）
                # # 方法1：torch.isin（1.10+ 版本支持）
                # common_mask = torch.isin(gt_flat_idx, mapping_flat_idx)
                # common_flat_idx = gt_flat_idx[common_mask]

                # 5. 统计数量


                #print(f"相同体素数量: {len(common_flat_idx)}")
                    
                
                
                output, output_sc = model(img, depth_mapping_3d, tsdf, epoch)

                #label = label.transpose(1,2)

                seq_loss = sequence_loss(output, label, label_weight, local_rank)


                loss = seq_loss


                current_idx = epoch * niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)

                optimizer.param_groups[0]['lr'] = lr
               
                # optimizer.param_groups[1]['lr'] = lr
                for i in range(1, len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr


                loss=loss.cuda()
                loss.backward()

                sum_loss += loss.item()
                sum_sem += 0
               
                optimizer.step()
                print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss=%.5f' % (sum_loss / (idx + 1)) \
                            + ' kldloss=%.5f' % (sum_kld / (idx + 1))

                pbar.set_description(print_str, refresh=False)

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss/tot', sum_loss / len(pbar), epoch)
                tb_writer.add_scalar('train_loss/semantic', sum_sem / len(pbar), epoch)
                tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            if (epoch >=0):
                if dist.get_rank() == 0:
                    if engine.distributed and (engine.local_rank == 0):
                        engine.save_and_link_checkpoint(config.snapshot_dir,
                                                        config.log_dir,
                                                        config.log_dir_link)
                    elif not engine.distributed:
                        engine.save_and_link_checkpoint(config.snapshot_dir,
                                                        config.log_dir,
                                                        config.log_dir_link)

if __name__ == '__main__':

    main()