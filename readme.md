# Pixel-wise Regression for 3D hand pose estimation 
PyTroch release of our paper:   
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
2. Unzip the files to `./data` and rename the folder as `MSRA`.
3. Download the default [train.txt](https://drive.google.com/open?id=1JaBPrhTwPYeaasz1-LsseRBFPm-jWFmU) and [test.txt](https://drive.google.com/open?id=15VqwOfUc0vGX1ivi6GygnQ0pb3JB43L8) from Google Drive. In our default setting, only 76318 frames are valid for training. If you don't want to use our default setting, feel free to change the parameters in `datasets.py`, and run `python check_dataset.py` to build the data files.

### ICVL  

### HAND17  

## Train  

## Test  

## Result  
You can download our results from Google Drive
- MSRA: [test result]() [model]()
- ICVL: [test result]() [model]()
- HAND17: [test result]() [model]()