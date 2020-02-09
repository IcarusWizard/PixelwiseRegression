import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

import os, argparse
from tqdm import tqdm

from model import FullRegression
import datasets
from utils import setup_seed, save_model, draw_skeleton_torch, select_gpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default="full_regression",
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
    parser.add_argument('--norm_method', type=str, default='instance', help='choose from batch and instance')
    
    parser.add_argument('--using_rotation', action='store_true')
    parser.add_argument('--using_scale', action='store_true')
    parser.add_argument('--using_flip', action='store_true')

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=9999)
    parser.add_argument('--stages', type=int, default=1)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--level', type=int, default=4)

    parser.add_argument('--lr', type=float, default=2e-4)

    args = parser.parse_args()

    if not os.path.exists('Model'):
        os.makedirs('Model')

    seed = args.seed if args.seed else np.random.randint(0, 100000)
    setup_seed(seed) 

    trainset_parameters = {
        "dataset" : "train",
        "image_size" : args.label_size * 2,
        "label_size" : args.label_size,
        "kernel_size" : args.kernel_size,
        "sigmoid" : args.sigmoid,
        "using_rotation" : args.using_rotation,
        "using_scale" : args.using_scale,
        "using_flip" : args.using_flip,
    }

    valset_parameters = {
        "dataset" : "val",
        "image_size" : args.label_size * 2,
        "label_size" : args.label_size,
        "kernel_size" : args.kernel_size,
        "sigmoid" : args.sigmoid,
        "using_rotation" : False,
        "using_scale" : False,
        "using_flip" : False,
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
    }

    log_name = "{}_{}".format(args.dataset, args.suffix)
    model_name = log_name + "_{}.pt" 

    Dataset = getattr(datasets, "{}Dataset".format(args.dataset))
    trainset = Dataset(**trainset_parameters)
    valset = Dataset(**valset_parameters)

    joints = trainset.joint_number
    config = trainset.config

    train_loader = torch.utils.data.DataLoader(trainset, **train_loader_parameters)
    val_loader = torch.utils.data.DataLoader(valset, **val_loader_parameters)

    select_gpus(args.gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = FullRegression(joints, **model_parameters)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter('logs/{}'.format(log_name))

    steps_per_epoch = len(trainset) // args.batch_size
    print("there are {} steps per epoch!".format(steps_per_epoch))
    total_steps = steps_per_epoch * args.epoch

    best_epoch = 0
    best_loss = 9999999

    with tqdm(total=total_steps) as pbar:
        for epoch in range(args.epoch):
            for batch in iter(train_loader):
                img, label_img, mask, box_size, com, uvd, heatmaps, depthmaps = batch
                
                img = img.to(device, non_blocking=True)
                label_img = label_img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                uvd = uvd.to(device, non_blocking=True)

                results = model(img, label_img, mask)

                every_loss = []
                for i, result in enumerate(results):
                    _uvd = result
                    uvd_loss = torch.mean(torch.sum((_uvd - uvd) ** 2, dim=2))
                    every_loss.append(uvd_loss)

                loss = 0
                for losses in every_loss:
                    uvd_loss = losses
                    loss = loss + uvd_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.update(1)

            # log image results in tensorboard
            writer.add_images('input_image', img, global_step=epoch)
            skeleton = draw_skeleton_torch(img[0].cpu(), uvd[0].cpu(), config)
            writer.add_image('input_skeleton', skeleton, global_step=epoch)
            for i, result in enumerate(results):
                _uvd = result
                _skeleton = draw_skeleton_torch(img[0].cpu(), _uvd[0].detach().cpu(), config)
                writer.add_image('stage{}_skeleton'.format(i), _skeleton, global_step=epoch)
                        
            with torch.no_grad():
                # compute val losses
                num = 0

                val_every_loss = []
                for i in range(len(results)):
                    val_every_loss.append(0)

                for val_batch in iter(val_loader):
                    num += 1
                    img, label_img, mask, box_size, com, uvd, heatmaps, depthmaps = val_batch
                    
                    img = img.to(device, non_blocking=True)
                    label_img = label_img.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    uvd = uvd.to(device, non_blocking=True)

                    results = model(img, label_img, mask)

                    for i, result in enumerate(results):
                        _uvd = result
                        uvd_loss = torch.mean(torch.sum((_uvd - uvd) ** 2, dim=2))
                        _uvd_loss = val_every_loss[i]
                        val_every_loss[i] = _uvd_loss + uvd_loss
                
                for i in range(len(results)):
                    _uvd_loss = val_every_loss[i]
                    val_every_loss[i] = _uvd_loss / num

                val_loss = 0
                for losses in val_every_loss:
                    uvd_loss = losses
                    val_loss = val_loss + uvd_loss 
            
            # log scalas in tensorboard
            writer.add_scalars('loss', {'train' : loss.item(), 'val' : val_loss.item()}, global_step=epoch)

            save_model(model, os.path.join('Model', model_name.format(epoch)), seed=seed, model_param=model_parameters)

            if val_every_loss[-1] < best_loss:
                best_epoch = epoch
                best_loss = val_every_loss[-1]

    print("best epoch is {}".format(best_epoch))
    os.system('cp {} {}'.format(os.path.join('Model', model_name.format(best_epoch)), os.path.join('Model', model_name.format('final'))))