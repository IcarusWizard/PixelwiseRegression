import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

import os, argparse, tempfile
from tqdm import tqdm
import cv2

from model import PixelwiseRegression
import datasets
from utils import load_model, recover_uvd, select_gpus

def del_white(img):
    image = img.copy()
    white = np.all(image == 1, axis=2)
    real = white == False
    index = np.argwhere(real)
    top, left = np.min(index, axis=0)
    buttom, right = np.max(index, axis=0)
    return image[top:buttom+1,left:right+1]

def get_image(image, cmap=plt.cm.jet):
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, 'temp.png')
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.savefig(temp_file)
    output_image = plt.imread(temp_file)
    output_image = del_white(output_image)
    return output_image
    
def overlap_images(image1, image2):
    output_image = 0.5 * image1 + 0.5 * image2
    return output_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--label_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--sigmoid', type=float, default=1.5)
    parser.add_argument('--norm_method', type=str, default='instance', help='choose from batch and instance')
    parser.add_argument('--heatmap_method', type=str, default='softmax', help='choose from softmax and sumz')

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--stages', type=int, default=2)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--seed', type=str, default='final')

    args = parser.parse_args()

    output_dir = 'sfr'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    suffixes = {
        'detection' : 'detection',
        'mix' : 'mix_h1.0_d0.01',
        'regression' : 'new_center_regression',
    }

    joint_indexes = {
        "Palm" : 13,
        "MCP" : 10,
        "IP" : 9,
        "TIP" : 8,
    }

    assert os.path.exists('Model'), "Please put the models in ./Model folder"

    dataset_parameters = {
        "image_size" : args.label_size * 2,
        "label_size" : args.label_size,
        "kernel_size" : args.kernel_size,
        "sigmoid" : args.sigmoid,
        "dataset" : "val",
        "test_only" : False,
    }

    test_loader_parameters = {
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

    Dataset = datasets.NYUDataset
    testset = Dataset(**dataset_parameters)

    joints = testset.joint_number
    config = testset.config

    test_loader = torch.utils.data.DataLoader(testset, **test_loader_parameters)
    
    models = {}
    for model_name, suffix in suffixes.items():
        model_file_name = "NYU_{}_{}.pt".format(suffix, args.seed)
        model = PixelwiseRegression(joints, **model_parameters)
        load_model(model, os.path.join('Model', model_file_name), eval_mode=True)
        model.requires_grad_(False)
        models[model_name] = model

    window_created = False
    for batch in iter(test_loader):
        img, label_img, mask, box_size, cube_size, com, uvd, heatmaps, depthmaps = batch

        output_images = {}

        output_depth = img.numpy()[0, 0]
        output_depth = get_image(output_depth, plt.cm.gray)

        output_images['depth'] = output_depth

        for model_name, model in models.items():
            results = model(img, label_img, mask)

            _heatmaps, _depthmaps, _uvd = results[-1]

            for joint_name, index in joint_indexes.items():
                output_heatmap = _heatmaps.numpy()[0, index]
                output_heatmap = get_image(output_heatmap)
                output_heatmap = overlap_images(output_heatmap, output_depth)
                output_images[model_name + '_' + joint_name + '_heatmap'] = output_heatmap

                output_depthmap = _depthmaps.numpy()[0, index]
                output_depthmap = get_image(output_depthmap)
                output_depthmap = overlap_images(output_depthmap, output_depth)
                output_images[model_name + '_' + joint_name + '_depthmap'] = output_depthmap

        for name, image in output_images.items():
            if not window_created:
                cv2.namedWindow(name, 0)
                window_created = True
            cv2.imshow(name, image)

        ch = cv2.waitKey(0)
        if ch == ord('s'):
            for name, image in output_images.items():
                plt.imsave(os.path.join(output_dir, name + '.png'), image)
        elif ch == ord('q'):
            break
