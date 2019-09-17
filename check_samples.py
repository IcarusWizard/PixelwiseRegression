import argparse
import datasets
import torch, torchvision
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MSRA', 
        help="choose from MSRA, ICVL, NYU, HAND17"    
    )
    parser.add_argument('--set', type=str, default='train',
        help='choose from train and test'
    )
    args = parser.parse_args()

    Dataset = getattr(datasets, "{}Dataset".format(args.dataset))
    dataset = Dataset(dataset=args.set, test_only=(args.set == 'test'))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in iter(loader):
        img, *_ = batch
        img = img.numpy()[0][0]

        plt.imshow(img)
        plt.show() 

    