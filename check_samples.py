import argparse
import datasets
import torch, torchvision
import matplotlib.pyplot as plt

from utils import draw_skeleton_torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MSRA', 
        help="choose from MSRA, ICVL, NYU, HAND17"    
    )
    parser.add_argument('--set', type=str, default='train',
        help='choose from train and test'
    )
    parser.add_argument('--using_rotation', action='store_true')
    parser.add_argument('--using_scale', action='store_true')
    parser.add_argument('--using_flip', action='store_true')
    args = parser.parse_args()

    Dataset = getattr(datasets, "{}Dataset".format(args.dataset))
    dataset = Dataset(dataset=args.set, test_only=(args.set == 'test'), 
        using_rotation=args.using_rotation, using_scale=args.using_scale, using_flip=args.using_flip)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in iter(loader):
        if args.set == 'test':
            img, label_img, mask, box_size, com = batch
        else:
            img, label_img, mask, box_size, com, uvd, heatmaps, depthmaps = batch

        if not args.set == 'test':  
            skeleton = draw_skeleton_torch(img[0], uvd[0], dataset.config)
            skeleton = skeleton.permute(1, 2, 0).numpy()
            fig, ax = plt.subplots()
            ax.imshow(skeleton)

        img = img.numpy()[0][0]

        fig, ax = plt.subplots()
        ax.imshow(img)

        plt.show() 

    