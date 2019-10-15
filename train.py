import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

import os, argparse
from tqdm import tqdm

from model import PixelwiseRegression
import datasets
from utils import setup_seed, step_loader, save_model, draw_skeleton_torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default="default",
        help="the suffix of model file and log file"
    )

    parser.add_argument('--dataset', type=str, default='MSRA', 
        help="choose from MSRA, ICVL, NYU, HAND17"    
    )

    parser.add_argument('--seed', type=int, default=0,
        help="the random seed used in the training, 0 means do not use fix seed"    
    )

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--label_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--sigmoid', type=float, default=1.5)
    parser.add_argument('--norm_method', type=str, default='batch', help='choose from batch and instance')
    parser.add_argument('--heatmap_method', type=str, default='softmax', help='choose from softmax and origin')
    parser.add_argument('--using_rotation', action='store_true')
    parser.add_argument('--using_scale', action='store_true')
    parser.add_argument('--using_flip', action='store_true')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument("--num_workers", type=int, default=9999)
    parser.add_argument('--stages', type=int, default=2)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--save_step', type=int, default=10000)

    parser.add_argument('--opt', type=str, default='adam', help='choose from adam and sgd')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument("--lambda_h", type=float, default=100.0)
    parser.add_argument('--lambda_d', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)

    args = parser.parse_args()

    if not os.path.exists('Model'):
        os.makedirs('Model')

    if not args.seed == 0:
        setup_seed(args.seed)

    dataset_parameters = {
        "image_size" : args.label_size * 2,
        "label_size" : args.label_size,
        "kernel_size" : args.kernel_size,
        "sigmoid" : args.sigmoid,
        "dataset" : "train",
        "using_rotation" : args.using_rotation,
        "using_scale" : args.using_scale,
        "using_flip" : args.using_flip,
    }

    train_loader_parameters = {
        "batch_size" : args.batch_size,
        "shuffle" : True,
        "pin_memory" : True, 
        "drop_last" : True,
        "num_workers" : min(args.num_workers, os.cpu_count()),
    }

    val_loader_parameters = {
        "batch_size" : args.batch_size,
        "shuffle" : False,
        "pin_memory" : True, 
        "drop_last" : False,
        "num_workers" : min(args.num_workers, os.cpu_count()),
    }

    model_parameters = {
        "stage" : args.stages, 
        "label_size" : args.label_size, 
        "features" : args.features, 
        "level" : args.level,
        "norm_method" : args.norm_method,
        "heatmap_method" : args.heatmap_method,
    }

    log_name = "{}_{}".format(args.dataset, args.suffix)
    model_name = log_name + "_{}.pt" 

    Dataset = getattr(datasets, "{}Dataset".format(args.dataset))
    trainset = Dataset(**dataset_parameters)

    joints = trainset.joint_number
    config = trainset.config
    
    valset, trainset = torch.utils.data.random_split(trainset, (1024, len(trainset) - 1024))

    train_loader = torch.utils.data.DataLoader(trainset, **train_loader_parameters)
    val_loader = torch.utils.data.DataLoader(valset, **val_loader_parameters)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    model = PixelwiseRegression(joints, **model_parameters)
    model = model.to(device)

    if args.opt == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr)

    writer = SummaryWriter('logs/{}'.format(log_name))

    step = 0
    with tqdm(total=args.steps) as pbar:
        for batch in step_loader(train_loader):
            img, label_img, mask, box_size, com, uvd, heatmaps, depthmaps = batch
            
            img = img.to(device, non_blocking=True)
            label_img = label_img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            uvd = uvd.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            depthmaps = depthmaps.to(device, non_blocking=True)

            results = model(img, label_img, mask)

            every_loss = []
            for i, result in enumerate(results):
                _heatmaps, _depthmaps, _uvd = result
                heatmap_loss = args.lambda_h * torch.mean((_heatmaps - heatmaps) ** 2)
                depthmap_loss = args.lambda_d * torch.mean((_depthmaps - depthmaps) ** 2)
                uvd_loss = torch.mean((_uvd - uvd) ** 2)
                every_loss.append((heatmap_loss, depthmap_loss, uvd_loss))

            loss = 0
            for losses in every_loss:
                heatmap_loss, depthmap_loss, uvd_loss = losses
                loss = loss + args.alpha * uvd_loss + (1 - args.alpha) * (heatmap_loss + depthmap_loss) 

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % args.log_step == 0:
                # log image results in tensorboard
                writer.add_images('input_image', img, global_step=step)
                writer.add_images('input_heatmap', heatmaps[0].unsqueeze(1) / heatmaps[0].max(), global_step=step)
                writer.add_images('input_depthmap', depthmaps[0].unsqueeze(1), global_step=step)
                skeleton = draw_skeleton_torch(img[0].cpu(), uvd[0].cpu(), config)
                writer.add_image('input_skeleton', skeleton, global_step=step)
                for i, result in enumerate(results):
                    _heatmaps, _depthmaps, _uvd = result
                    writer.add_images('stage{}_heatmap'.format(i), _heatmaps[0].unsqueeze(1) / _heatmaps[0].max(), global_step=step)
                    writer.add_images('stage{}_depthmap'.format(i), _depthmaps[0].unsqueeze(1), global_step=step)
                    _skeleton = draw_skeleton_torch(img[0].cpu(), _uvd[0].detach().cpu(), config)
                    writer.add_image('stage{}_skeleton'.format(i), _skeleton, global_step=step)
                         
                with torch.no_grad():
                    # compute val losses
                    num = 0

                    val_every_loss = []
                    for i in range(len(results)):
                        val_every_loss.append((0, 0, 0))

                    for val_batch in iter(val_loader):
                        num += 1
                        img, label_img, mask, box_size, com, uvd, heatmaps, depthmaps = val_batch
                        
                        img = img.to(device, non_blocking=True)
                        label_img = label_img.to(device, non_blocking=True)
                        mask = mask.to(device, non_blocking=True)
                        uvd = uvd.to(device, non_blocking=True)
                        heatmaps = heatmaps.to(device, non_blocking=True)
                        depthmaps = depthmaps.to(device, non_blocking=True)

                        results = model(img, label_img, mask)

                        for i, result in enumerate(results):
                            _heatmaps, _depthmaps, _uvd = result
                            heatmap_loss = args.lambda_h * torch.mean((_heatmaps - heatmaps) ** 2)
                            depthmap_loss = args.lambda_d * torch.mean((_depthmaps - depthmaps) ** 2)
                            uvd_loss = torch.mean((_uvd - uvd) ** 2)
                            _heatmaps_loss, _depthmap_loss, _uvd_loss = val_every_loss[i]
                            val_every_loss[i] = ((
                                _heatmaps_loss + heatmap_loss, 
                                _depthmap_loss + depthmap_loss, 
                                _uvd_loss + uvd_loss))
                    
                    for i in range(len(results)):
                        _heatmaps_loss, _depthmap_loss, _uvd_loss = val_every_loss[i]
                        val_every_loss[i] = ((
                            _heatmaps_loss / num, 
                            _depthmap_loss / num, 
                            _uvd_loss / num))

                    val_loss = 0
                    for losses in val_every_loss:
                        heatmap_loss, depthmap_loss, uvd_loss = losses
                        val_loss = val_loss + args.alpha * uvd_loss + (1 - args.alpha) * (heatmap_loss + depthmap_loss) 
                
                # log scalas in tensorboard
                writer.add_scalars('loss', {'train' : loss.item(), 'val' : val_loss.item()}, global_step=step)
                for i in range(len(every_loss)):
                    train_heatmap_loss, train_depthmap_loss, train_uvd_loss = every_loss[i]
                    val_heatmap_loss, val_depthmap_loss, val_uvd_loss = val_every_loss[i]

                    writer.add_scalars('stage{}_heatmap_loss'.format(i), 
                        {'train' : train_heatmap_loss, 'val' : val_heatmap_loss},
                        global_step=step
                    )
                    writer.add_scalars('stage{}_depthmap_loss'.format(i), 
                        {'train' : train_depthmap_loss, 'val' : val_depthmap_loss},
                        global_step=step
                    )
                    writer.add_scalars('stage{}_uvd_loss'.format(i), 
                        {'train' : train_uvd_loss, 'val' : val_uvd_loss},
                        global_step=step
                    )

            if step % args.save_step == 0:
                save_model(model, os.path.join('Model', model_name.format(step)))

            step = step + 1
            pbar.update(1)

            if step >= args.steps:
                break

    save_model(model, os.path.join('Model', model_name.format('final')))
