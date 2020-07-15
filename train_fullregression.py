import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

import os, argparse
from tqdm import tqdm

from model import FullRegression
import datasets
from utils import setup_seed, save_model, draw_skeleton_torch, select_gpus, recover_uvd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default="full_regression",
        help="the suffix of model file and log file"
    )

    parser.add_argument('--dataset', type=str, default='NYU', 
        help="choose from ICVL, NYU, HAND17"    
    )

    parser.add_argument('--seed', type=int, default=0,
        help="the random seed used in the training, 0 means do not use fix seed"    
    )

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--label_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--sigmoid', type=float, default=1.5)
    parser.add_argument('--norm_method', type=str, default='instance', help='choose from batch and instance')
    
    parser.add_argument('--using_rotation', type=lambda x: [False, True][int(x)], default=True)
    parser.add_argument('--using_scale', type=lambda x: [False, True][int(x)], default=True)
    parser.add_argument('--using_shift', type=lambda x: [False, True][int(x)], default=True)
    parser.add_argument('--using_flip', type=lambda x: [False, True][int(x)], default=False)

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=9999)
    parser.add_argument('--stages', type=int, default=1)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--level', type=int, default=4)

    parser.add_argument('--opt', type=str, default='adam', help='choose from adam and sgd')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mixed_precision', action='store_true', help='enbale mixed precision training')
    parser.add_argument('--lr_decay', type=float, default=0.2)
    parser.add_argument('--decay_epoch', type=float, default=15)

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
        "using_shift" : args.using_shift,
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

    if args.opt == 'adam':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_epoch, gamma=args.lr_decay)
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter('logs/{}'.format(log_name))

    steps_per_epoch = len(trainset) // args.batch_size
    print("there are {} steps per epoch!".format(steps_per_epoch))
    total_steps = steps_per_epoch * args.epoch

    best_epoch = 0
    best_error = 9999999

    with tqdm(total=total_steps) as pbar:
        for epoch in range(args.epoch):
            for batch in iter(train_loader):
                img, label_img, mask, box_size, cube_size, com, uvd, heatmaps, depthmaps = batch
                
                img = img.to(device, non_blocking=True)
                label_img = label_img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                uvd = uvd.to(device, non_blocking=True)

                optim.zero_grad()

                if args.mixed_precision: # mixed precision training code
                    with torch.cuda.amp.autocast():
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

                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()

                else: # normal training code
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

            scheduler.step()

            # log image results in tensorboard
            writer.add_images('input_image', img, global_step=epoch)
            skeleton = draw_skeleton_torch(img[0].cpu(), uvd[0].cpu(), config)
            writer.add_image('input_skeleton', skeleton, global_step=epoch)
            for i, result in enumerate(results):
                _uvd = result
                _skeleton = draw_skeleton_torch(img[0].cpu(), _uvd[0].detach().cpu(), config)
                writer.add_image('stage{}_skeleton'.format(i), _skeleton, global_step=epoch)
            
            model.eval()
            with torch.no_grad():
                # compute val losses
                num = 0

                val_every_loss = []
                dataset_results = []
                for i in range(len(results)):
                    val_every_loss.append(0)
                    dataset_results.append([])

                for val_batch in iter(val_loader):
                    num += 1
                    img, label_img, mask, box_size, cube_size, com, uvd, heatmaps, depthmaps = val_batch
                    
                    img = img.to(device, non_blocking=True)
                    label_img = label_img.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    uvd = uvd.to(device, non_blocking=True)

                    results = model(img, label_img, mask)

                    true_uvd = uvd.cpu()
                    true_uvd = recover_uvd(true_uvd, box_size, com, cube_size)
                    true_uvd = true_uvd.numpy()

                    true_xyz = valset.uvd2xyz(true_uvd)

                    for i, result in enumerate(results):
                        _uvd = result
                        uvd_loss = torch.mean(torch.sum((_uvd - uvd) ** 2, dim=2))
                        _uvd_loss = val_every_loss[i]
                        val_every_loss[i] = _uvd_loss + uvd_loss

                        _uvd = _uvd.cpu()
                        _uvd = recover_uvd(_uvd, box_size, com, cube_size)
                        _uvd = _uvd.numpy()

                        _xyz = valset.uvd2xyz(_uvd)
                        dataset_results[i].append(np.mean(np.sqrt(np.sum((_xyz - true_xyz) ** 2, axis=2)), axis=1))
                
                for i in range(len(results)):
                    _uvd_loss = val_every_loss[i]
                    val_every_loss[i] = _uvd_loss / num

                    dataset_results[i] = np.mean(np.concatenate(dataset_results[i], axis=0))

                val_loss = 0
                for losses in val_every_loss:
                    uvd_loss = losses
                    val_loss = val_loss + uvd_loss 
            
            model.train()
            # log scalas in tensorboard
            writer.add_scalars('loss', {'train' : loss.item(), 'val' : val_loss.item()}, global_step=epoch)

            save_model(model, os.path.join('Model', model_name.format(epoch)), seed=seed, model_param=model_parameters)

            if dataset_results[-1] < best_error:
                best_epoch = epoch
                best_error = dataset_results[-1]

    print("best epoch is {}".format(best_epoch))
    os.system('cp {} {}'.format(os.path.join('Model', model_name.format(best_epoch)), os.path.join('Model', model_name.format('final'))))