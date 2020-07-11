import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

import os, argparse
from tqdm import tqdm

from model import PixelwiseRegression
import datasets
from utils import setup_seed, step_loader, save_model, draw_skeleton_torch, select_gpus, draw_features_torch, recover_uvd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default="default",
        help="the suffix of model file and log file"
    )

    parser.add_argument('--dataset', type=str, default='NYU', 
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
    parser.add_argument('--heatmap_method', type=str, default='softmax', help='choose from softmax and sum')

    # need more time to train if using any of these augmentation
    parser.add_argument('--using_rotation', action='store_true')
    parser.add_argument('--using_scale', action='store_true')
    parser.add_argument('--using_flip', action='store_true')
    parser.add_argument('--small', action='store_true')

    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=9999)
    parser.add_argument('--stages', type=int, default=2)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--filter_size', type=int, default=3)

    parser.add_argument('--opt', type=str, default='adam', help='choose from adam and sgd')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--lambda_h", type=float, default=1.0)
    parser.add_argument('--lambda_d', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)

    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--decay_epoch', type=float, default=30)

    args = parser.parse_args()

    if not os.path.exists('Model'):
        os.makedirs('Model')

    seed = args.seed if args.seed else np.random.randint(0, 100000)
    setup_seed(seed) 

    trainset_parameters = {
        "dataset" : "train" if not args.small else "small_train",
        "image_size" : args.label_size * 2,
        "label_size" : args.label_size,
        "kernel_size" : args.kernel_size,
        "sigmoid" : args.sigmoid,
        "using_rotation" : args.using_rotation,
        "using_scale" : args.using_scale,
        "using_flip" : args.using_flip,
    }

    valset_parameters = {
        "dataset" : "val" if not args.small else "small_val",
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
        "heatmap_method" : args.heatmap_method,
        "kernel_size" : args.filter_size,
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
    
    model = PixelwiseRegression(joints, **model_parameters)
    model = model.to(device)

    if args.opt == 'adam':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_epoch, gamma=args.lr_decay)
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
                heatmaps = heatmaps.to(device, non_blocking=True)
                depthmaps = depthmaps.to(device, non_blocking=True)

                results = model(img, label_img, mask)

                every_loss = []
                for i, result in enumerate(results):
                    _heatmaps, _depthmaps, _uvd = result
                    heatmap_loss = args.lambda_h * torch.mean(torch.sum((_heatmaps - heatmaps) ** 2, dim=(2, 3)))
                    depthmap_loss = args.lambda_d * torch.mean(torch.sum((_depthmaps - depthmaps) ** 2, dim=(2, 3)))
                    uvd_loss = torch.mean(torch.sum((_uvd - uvd) ** 2, dim=2))
                    every_loss.append((heatmap_loss, depthmap_loss, uvd_loss))

                loss = 0
                for losses in every_loss:
                    heatmap_loss, depthmap_loss, uvd_loss = losses
                    loss = loss + args.alpha * uvd_loss + (1 - args.alpha) * (heatmap_loss + depthmap_loss) 

                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.update(1)
            scheduler.step()

            # log image results in tensorboard
            writer.add_images('input_image', img, global_step=epoch)
            writer.add_figure('input_heatmap', draw_features_torch(heatmaps[0]), global_step=epoch)
            writer.add_figure('input_depthmap', draw_features_torch(depthmaps[0]), global_step=epoch)
            skeleton = draw_skeleton_torch(img[0].cpu(), uvd[0].cpu(), config)
            writer.add_image('input_skeleton', skeleton, global_step=epoch)
            for i, result in enumerate(results):
                _heatmaps, _depthmaps, _uvd = result
                writer.add_figure('stage{}_heatmap'.format(i), draw_features_torch(_heatmaps[0]), global_step=epoch)
                writer.add_figure('stage{}_depthmap'.format(i), draw_features_torch(_depthmaps[0]), global_step=epoch)
                _skeleton = draw_skeleton_torch(img[0].cpu(), _uvd[0].detach().cpu(), config)
                writer.add_image('stage{}_skeleton'.format(i), _skeleton, global_step=epoch)
            
            model.eval()
            with torch.no_grad():
                # compute val losses
                num = 0

                val_every_loss = []
                dataset_results = []
                for i in range(len(results)):
                    val_every_loss.append((0, 0, 0))
                    dataset_results.append([])

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

                    true_uvd = uvd.cpu()
                    true_uvd = recover_uvd(true_uvd, box_size, com, valset.cube_size)
                    true_uvd = true_uvd.numpy()

                    true_xyz = valset.uvd2xyz(true_uvd)

                    for i, result in enumerate(results):
                        _heatmaps, _depthmaps, _uvd = result
                        heatmap_loss = args.lambda_h * torch.mean(torch.sum((_heatmaps - heatmaps) ** 2, dim=(2, 3)))
                        depthmap_loss = args.lambda_d * torch.mean(torch.sum((_depthmaps - depthmaps) ** 2, dim=(2, 3)))
                        uvd_loss = torch.mean(torch.sum((_uvd - uvd) ** 2, dim=2))
                        _heatmaps_loss, _depthmap_loss, _uvd_loss = val_every_loss[i]
                        val_every_loss[i] = ((
                            _heatmaps_loss + heatmap_loss, 
                            _depthmap_loss + depthmap_loss, 
                            _uvd_loss + uvd_loss))

                        _uvd = _uvd.cpu()
                        _uvd = recover_uvd(_uvd, box_size, com, valset.cube_size)
                        _uvd = _uvd.numpy()

                        _xyz = valset.uvd2xyz(_uvd)
                        dataset_results[i].append(np.mean(np.sqrt(np.sum((_xyz - true_xyz) ** 2, axis=2)), axis=1))
                
                for i in range(len(results)):
                    _heatmaps_loss, _depthmap_loss, _uvd_loss = val_every_loss[i]
                    val_every_loss[i] = ((
                        _heatmaps_loss / num, 
                        _depthmap_loss / num, 
                        _uvd_loss / num))

                    dataset_results[i] = np.mean(np.concatenate(dataset_results[i], axis=0))

                val_loss = 0
                for losses in val_every_loss:
                    heatmap_loss, depthmap_loss, uvd_loss = losses
                    val_loss = val_loss + args.alpha * uvd_loss + (1 - args.alpha) * (heatmap_loss + depthmap_loss) 
            
            model.train()
            # log scalas in tensorboard
            writer.add_scalars('loss', {'train' : loss.item(), 'val' : val_loss.item()}, global_step=epoch)
            for i in range(len(every_loss)):
                train_heatmap_loss, train_depthmap_loss, train_uvd_loss = every_loss[i]
                val_heatmap_loss, val_depthmap_loss, val_uvd_loss = val_every_loss[i]

                writer.add_scalars('stage{}_heatmap_loss'.format(i), 
                    {'train' : train_heatmap_loss, 'val' : val_heatmap_loss},
                    global_step=epoch
                )
                writer.add_scalars('stage{}_depthmap_loss'.format(i), 
                    {'train' : train_depthmap_loss, 'val' : val_depthmap_loss},
                    global_step=epoch
                )
                writer.add_scalars('stage{}_uvd_loss'.format(i), 
                    {'train' : train_uvd_loss, 'val' : val_uvd_loss},
                    global_step=epoch
                )
                writer.add_scalar('stage{}_result'.format(i), dataset_results[i], global_step=epoch)

            save_model(model, os.path.join('Model', model_name.format(epoch)), seed=seed, model_param=model_parameters)

            if val_every_loss[-1][-1] < best_loss:
                best_epoch = epoch
                best_loss = val_every_loss[-1][-1]

    print("best epoch is {}".format(best_epoch))
    os.system('cp {} {}'.format(os.path.join('Model', model_name.format(best_epoch)), os.path.join('Model', model_name.format('final'))))
