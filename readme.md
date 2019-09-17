# Pixel-wise Regression for 3D hand pose estimation 
**PyTroch release of our paper:**   
[Pixel-wise Regression: 3D Hand Pose Estimation via Spatial-form Representation and Differentiable Decoder](https://arxiv.org/abs/1905.02085)  
*Xingyuan Zhang, Fuhai Zhang*

If you find this open source release useful, please reference in your paper:
```
@misc{zhang2019pixelwise,
    title={Pixel-wise Regression: 3D Hand Pose Estimation via Spatial-form Representation and Differentiable Decoder},
    author={Xingyuan Zhang and Fuhai Zhang},
    year={2019},
    eprint={1905.02085},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Requirement  
PyTorch >= 1.0  
tensorboard (install by `pip install tb-nightly`)

## Dataset  
All datasets should be placed in `./Data` folder.  
### MSRA  
1. Download the dataset from [dropbox](https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0).
2. Unzip the files to `./Data` and rename the folder as `MSRA`.
3. Download the default [train.txt](https://drive.google.com/open?id=1RESPwhnlbQ1Rg7qNPCJ0zGBy7aOjdGmB) and [test.txt](https://drive.google.com/open?id=1QZTl5X5IX4GPZ429l_EOkf0yBnzYGQHO) from Google Drive. In our default setting, 76318 out of 76500 frames are valid for training. If you don't want to use our default setting, feel free to change the parameters in `datasets.py`, and run `python check_dataset.py --dataset MSRA` to build the data files.

### ICVL  
1. Download the dataset from [here](https://labicvl.github.io/hand.html).  
2. Extract `Training.tar.gz` and `Testing.tar.gz` to `./Data/ICVL/Training` and `./Data/ICVL/Testing` respectively.
3. Download the default [train.txt](https://drive.google.com/open?id=1JaBPrhTwPYeaasz1-LsseRBFPm-jWFmU) and [test.txt](https://drive.google.com/open?id=15VqwOfUc0vGX1ivi6GygnQ0pb3JB43L8) from Google Drive. In our default setting, 330885 out of 331006 frames are valid for training. If you don't want to use our default setting, feel free to change the parameters in `datasets.py`, and run `python check_dataset.py --dataset ICVL` to build the data files.

### HAND17  
1. Ask for the permission from the author of the dataset and download.  
http://icvl.ee.ic.ac.uk/hands17/challenge/
2. Extract `frame.zip` and `images.zip` to `./Data/HAND17/`. Your should end with a folder look like below:
```
HAND17/
  |
  |-- training/
  |     |
  |     |-- images/
  |     |
  |     |-- Training_Annotation.txt
  |
  |-- frame/
  |     |
  |     |-- images/
  |     |
  |     |-- BoundingBox.txt
```
3. Download the default [train.txt](https://drive.google.com/open?id=1JaBPrhTwPYeaasz1-LsseRBFPm-jWFmU) and [test.txt](https://drive.google.com/open?id=15VqwOfUc0vGX1ivi6GygnQ0pb3JB43L8) from Google Drive. In our default setting, only 76318 frames are valid for training. If you don't want to use our default setting, feel free to change the parameters in `datasets.py`, and run `python check_dataset.py --dataset HAND17` to build the data files.

## Train  
Run `python train.py --dataset <DatasetName>`, `DatasetName` can choose from `MSRA`, `ICVL` and `HAND17`.  
Run `python train.py -h` to find parameters that you can tune.

## Test  
Run `python test.py --dataset <DatasetName>`.  
Run `python train.py -h` to find parameters that you can tune.

## Result  
You can download our results from Google Drive
- MSRA: [test result]() [model]()
- ICVL: [test result]() [model]()
- HAND17: [test result]() [model]()