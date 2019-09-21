import torch, torchvision

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass # use to compute center of mass (row, col)
from scipy.io import loadmat
import cv2

import os, random, time, struct, re
from tqdm import tqdm
import multiprocessing as mp

from utils import load_bin, draw_skeleton, center_crop, \
    generate_com_filter, floodFillDepth, generate_heatmap, random_rotated, generate_kernel

class HandDataset(torch.utils.data.Dataset):
    def __init__(self, fx, fy, halfu, halfv, path, sigmoid, image_size, kernel_size,
                label_size, test_only, using_rotation, using_scale, using_flip, scale_factor, 
                threshold, joint_number, process_mode='uvd', dataset='train'):
        super(HandDataset, self).__init__()
        self.fx = fx                        # focal length in x axis
        self.fy = fy                        # focal length in y axis
        self.halfu = halfu                  # shift in u axis
        self.halfv = halfv                  # shift in v axis
        self.path = path                    # path contains the dataset
        self.sigmoid = sigmoid              # the sigmoid used in the gauss filter
        self.image_size = image_size        # the output image size
        self.kernel_size = kernel_size      # kernel size of the gauss filter 
        self.label_size = label_size        # the size of label images
        self.test_only = test_only          # set to True, the dataset will not return heatmap and depthmap
        self.using_rotation = using_rotation# whether to perform random rotation for images
        self.using_scale = using_scale      # whether to perform a random scale for images
        self.using_flip = using_flip        # whether to filp the image to solve the chirality problem
        self.scale_factor = scale_factor    # the factor used in random scale
        self.threshold = threshold          # half of the boxsize in z axis 
        self.joint_number = joint_number    # number of joints
        self.config = None                  # configuration of fingers (index in bottom-up order) 
        self.process_mode = process_mode    # the processing mode used in processing single data (uvd or bb)
        self.dataset = dataset

        if self.test_only:
            assert not (self.using_flip or self.using_scale or self.using_rotation), \
                "you can not transform the test data"

        self.build_data()

        dataname = os.path.join(self.path, self.dataset + ".txt")
        with open(dataname, 'r') as f:
            datatexts = f.readlines()

        self.text_list = datatexts

    def __getitem__(self, index):
        """
            Overwrite function in Dataset
        """
        return self.process_single_data(self.text_list[index])

    def __len__(self):
        """
            Overwrite function in Dataset
        """
        return len(self.text_list)
    
    def xyz2uvd(self, data):
        """
            Convert data from xyz to uvd
        """
        x = data.copy()
        if len(x.shape) == 3:
            x[:, :, 0] = x[:, :, 0] * self.fx / x[:, :, 2] + self.halfu
            x[:, :, 1] = x[:, :, 1] * self.fy / x[:, :, 2] + self.halfv
        elif len(x.shape) == 2:
            x[:, 0] = x[:, 0] * self.fx / x[:, 2] + self.halfu
            x[:, 1] = x[:, 1] * self.fy / x[:, 2] + self.halfv
        return x

    def uvd2xyz(self, data):
        """
            Convert data from uvd to xyz
        """
        x = data.copy()
        if len(x.shape) == 3:
            x[:, :, 0] = (x[:, :, 0] - self.halfu) / self.fx * x[:, :, 2]
            x[:, :, 1] = (x[:, :, 1] - self.halfv) / self.fy * x[:, :, 2]
        elif len(x.shape) == 2:
            x[:, 0] = (x[:, 0] - self.halfu) / self.fx * x[:, 2]
            x[:, 1] = (x[:, 1] - self.halfv) / self.fy * x[:, 2]
        return x

    def write_data_txt(self, filename, paths, joints):
        """
            Write paths and joints to txt file
            Inputs:
                filename: output path
                paths: bin paths, a list of path
                joints: joints label, a list of ndarray  (xyz)
        """
        with open(filename, 'w') as f:
            for i in range(len(paths)):
                path = paths[i]
                joint = joints[i]
                joint = list(joint)
                joint = list(map(str, joint))
                f.write(path + " " + " ".join(joint) + "\n")
    
    def decode_line_txt(self, string):
        """
            Decode a single line of txt file
            Input:
                string: text to decode
            Output:
                image filename
                joints xyz: ndarray(J, 3)
        """
        l = string.strip().split()
        path = l[0]
        data = l[1:]
        data = np.array(list(map(float, data)))
        return path, data.reshape((data.shape[0] // 3, 3))

    @property
    def data_ready(self):
        """
            Check if the dataset is already created
        """
        return os.path.exists(os.path.join(self.path, "train.txt")) and os.path.exists(os.path.join(self.path, "test.txt"))

    def build_data(self):
        """
            This function should build the train and test set 
        """
        raise NotImplementedError()

    def check_text(self, text):
        """
            A helper function to check the dataset
        """
        try:
            self.process_single_data(text)
            return text
        except:
            return False

    def load_from_text(self, text):
        """
        This function decode the text and load the image with only hand and the uvd coordinate of joints
        OUTPUT:
            image, uvd
        """
        raise NotImplementedError()

    def load_from_text_bb(self, text):
        """
        This function decode the text and load the image with with boundary box
        OUTPUT:
            image
        """
        raise NotImplementedError()

    def process_single_data(self, text):
        """
            Process a single line of text in the data file
            INPUT:
                text: string    a line of text in the data file
                img_size: int   the output image size
                using_rotation bool if to use random rotation to make data enhancement
                kernel_size int the size of kernel in gauss kernel
                label_size: int the output label size
            OUTPUT:
                Normalized Image, Normalized Label Image, mask, Box Size, COM
                Normalized uvd groundtruth, Heatmap, Normalized Dmap
        """

        if self.process_mode == 'uvd':
            # decode the text and load the image with only hand and the uvd coordinate of joints
            image, joint_uvd = self.load_from_text(text)
        else: # process_mode == 'bb'
            assert self.test_only
            image = self.load_from_text_bb(text)
            
        # crop the image
        mean = np.mean(image[image > 0])
        com = center_of_mass(image)

        try:
            if self.using_scale:
                box_size = int((self.scale_factor + (6000 * random.random() - 3000)) / mean)
            else:
                box_size = int(self.scale_factor / mean) # empirical number of boundar box fitting
        except:
            image_name, _ = self.decode_line_txt(text)
            print("error for {}, nothing in the image".format(image_name))
            raise ValueError("error for {}, nothing in the image".format(image_name))
        
        box_size = max(box_size, 2) # constrain the min value of the box_size is 2, if we only cut the background the box_size may be 1 or 0
        crop_img = center_crop(image, com, box_size)
        
        # TODO : Do we really need this filter
        # filter the pixel that have value out of the box
        crop_img[crop_img - mean > self.threshold] = 0
        crop_img[crop_img - mean < -self.threshold] = 0 
        
        # norm the image and uvd to COM
        crop_img[crop_img > 0] -= mean # center the depth image to COM
        com = [int(com[1]), int(com[0]), mean]
        com = np.array(com) # rebuild COM the uvd format
        
        box_size = crop_img.shape[0] # update box_size

        if self.using_flip:
            if random.random() < 0.5: # probality to flip
                for j in range(crop_img.shape[1] // 2):
                    tem = crop_img[:, j].copy()
                    crop_img[:, j] = crop_img[:, crop_img.shape[1] - j - 1].copy()
                    crop_img[:, crop_img.shape[1] - j - 1] = tem
                joint_uvd_centered[:, 0] = - joint_uvd_centered[:, 0]

        # resize the image and uvd
        try:
            img_resize = cv2.resize(crop_img, (self.image_size, self.image_size))
        except:
            # probably because size is zero
            print("resize error")
            raise ValueError("Resize error")

        # Generate label_image and mask
        label_image = cv2.resize(img_resize, (self.label_size, self.label_size))
        is_hand = label_image != 0
        mask = is_hand.astype(float)

        if self.test_only:
            # Just return the basic elements we need to run the network
            # normalize the image first before return
            normalized_img = img_resize / self.threshold
            normalized_label_img = label_image / self.threshold

            # Convert to torch format
            normalized_img = torch.from_numpy(normalized_img).float().unsqueeze(0)
            normalized_label_img = torch.from_numpy(normalized_label_img).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            box_size = torch.tensor(box_size).float()
            com = torch.from_numpy(com).float()
            
            return normalized_img, normalized_label_img, mask, box_size, com

        # ---------------------------------------------------------------------- #
        #                    Below code is only for training                     #
        # ---------------------------------------------------------------------- #

        joint_uvd_centered = joint_uvd - com # center the uvd to COM
        joint_uvd_centered_resize = joint_uvd_centered.copy()
        joint_uvd_centered_resize[:,:2] = joint_uvd_centered_resize[:,:2] / (box_size - 1) * (self.image_size - 1)

        if self.using_rotation:
            # random rotated the image and the label
            img_resize, joint_uvd_centered_resize = random_rotated(img_resize, joint_uvd_centered_resize)

        # Generate Heatmap
        joint_uvd_kernel = joint_uvd_centered_resize.copy()
        joint_uvd_kernel[:,:2] = joint_uvd_kernel[:,:2] / (self.image_size - 1) * (self.label_size - 1) + np.array([self.label_size // 2, self.label_size // 2])
        try:
            heatmaps = [generate_kernel(generate_heatmap(self.label_size, joint_uvd_kernel[i, 0], joint_uvd_kernel[i, 1]), kernel_size=self.kernel_size, sigmoid=self.sigmoid)[:, :, np.newaxis] for i in range(self.joint_number)]
        except:
            path, _ = self.decode_line_txt(text)
            print("{} heatmap error".format(path))
            raise ValueError("{} heatmap error".format(path))

        heatmaps = np.concatenate(heatmaps, axis=2)

        # Generate Dmap
        Dmap = []
        for i in range(self.joint_number):
            heatmask = heatmaps[:, :, i] > 0
            heatmask = heatmask.astype(float) * mask
            # # Only use below code when facing NAN problem
            # if np.sum(heatmask) == 0:
            #     # Heatmap don't on hand may creat an error
            #     print("heatmap don't on hand")
            #     return None, None, None, None, None, None, None, None
            Dmap.append(((joint_uvd_centered_resize[i, 2] - label_image.copy()) * heatmask)[:, :, np.newaxis])
        Dmap = np.concatenate(Dmap, axis=2)   

        # Normalize data
        normalized_img = img_resize / self.threshold
        normalized_label_img = label_image / self.threshold
        normalized_Dmap = Dmap / self.threshold
        normalized_uvd = joint_uvd_centered_resize.copy()
        normalized_uvd[:, :2] = normalized_uvd[:, :2] / (self.image_size - 1)
        normalized_uvd[:, 2] = normalized_uvd[:, 2] / self.threshold
        
        if np.any(np.isnan(normalized_img)) or np.any(np.isnan(normalized_uvd)) or np.any(np.isnan(heatmaps)) or np.any(np.isnan(normalized_label_img)) or np.any(np.isnan(normalized_Dmap)) or np.any(np.isnan(mask)) or np.sum(mask) < 10:
            path, data = self.decode_line_txt(text)
            print("Wired things happen, image contain Nan {}, {}".format(path, np.sum(mask)))
            raise ValueError("Wired things happen, image contain Nan {}, {}".format(path, np.sum(mask)))

        # Convert to torch format
        normalized_img = torch.from_numpy(normalized_img).float().unsqueeze(0)
        normalized_label_img = torch.from_numpy(normalized_label_img).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        box_size = torch.tensor(box_size).float()
        com = torch.from_numpy(com).float()
        normalized_uvd = torch.from_numpy(normalized_uvd).float()
        heatmaps = torch.from_numpy(heatmaps).float().permute(2, 0, 1).contiguous()
        normalized_Dmap = torch.from_numpy(normalized_Dmap).float().permute(2, 0, 1).contiguous()

        return normalized_img, normalized_label_img, mask, box_size, com, normalized_uvd, heatmaps, normalized_Dmap        

class MSRADataset(HandDataset):
    def __init__(self, fx = 241.42, fy = 241.42, halfu = 160, halfv = 120, path="Data/MSRA", 
                sigmoid=1.5, image_size=128, kernel_size=7,
                label_size=64, test_only=False, using_rotation=False, using_scale=False, using_flip=False, 
                scale_factor=50000, threshold=200, joint_number=21, dataset='train'):
        super(MSRADataset, self).__init__(fx, fy, halfu, halfv, path, sigmoid, image_size, kernel_size,
                label_size, test_only, using_rotation, using_scale, using_flip, scale_factor, 
                threshold, joint_number, process_mode='uvd', dataset=dataset)

        Index = [0, 1, 2, 3, 4]
        Mid = [0, 5, 6, 7, 8]
        Ring = [0, 9, 10, 11, 12]
        Small = [0, 13, 14, 15, 16]
        Thumb = [0, 17, 18, 19, 20]
        self.config = [Thumb, Index, Mid, Ring, Small]

    def build_data(self):
        if self.data_ready:
            print("Data is Already build~")
            return
        
        persons = ["P%d" % i for i in range(9)]
        gestures = os.listdir(os.path.join(self.path, persons[0]))
        gestures.sort()
        paths = [os.path.join(self.path, person, gesture) for person in persons for gesture in gestures]
        bin_paths = []
        joints = []

        print('loading file list ......')
        with tqdm(total=len(paths)) as pbar:
            for i, path in enumerate(paths):
                _joints = np.loadtxt(os.path.join(path, 'joint.txt'), skiprows=1)
                with open(os.path.join(path, 'joint.txt')) as f:
                    samples = int(f.readline())
                _joints = _joints.reshape((samples, 21, 3))
                _joints[:, :, 1] = - _joints[:, :, 1]
                _joints[:, :, 2] = - _joints[:, :, 2]
                joints.append(_joints.reshape((samples, 63)))
                for j in range(samples):
                    bin_paths.append(os.path.join(path, "%06d_depth.bin" % j))
                pbar.update(1)
        joints = np.concatenate(joints, axis=0)

        print('saving test.txt ......')
        super().write_data_txt(os.path.join(self.path, "test.txt"), list(bin_paths), list(joints))

        print('checking data ......')
        dataname = os.path.join(self.path, "test.txt")
        with open(dataname, 'r') as f:
            datatexts = f.readlines()

        pool = mp.Pool(processes=os.cpu_count())
        processing = []
        for text in datatexts:
            r = pool.apply_async(self.check_text, (text, ))
            processing.append(r)

        traintxt = []
        with tqdm(total=len(datatexts)) as pbar:
            for r in processing:
                text = r.get()
                if text:
                    traintxt.append(text)
                pbar.update(1)
        pool.close()

        # traintxt = []
        # with tqdm(total=len(datatexts)) as pbar:
        #     for text in datatexts:
        #         try:
        #             self.process_single_data(text)
        #             traintxt.append(text)
        #         except:
        #             pass
        #         pbar.update(1)
        
        print('{} / {} data can use to train'.format(len(traintxt), len(datatexts)))
        with open(os.path.join(self.path, "train.txt"), 'w') as f:
            f.writelines(traintxt)
        
    def load_from_text(self, text):
        """
        This function decode the text and load the image with only hand and the uvd coordinate of joints
        OUTPUT:
            image, uvd
        """
        path, joint_xyz = super().decode_line_txt(text)
        joint_uvd = super().xyz2uvd(joint_xyz)
        img, left, top, right, bottom = load_bin(path)

        image = np.zeros((self.halfv * 2, self.halfu * 2))
        image[top:bottom, left:right] = img.copy()
        return image, joint_uvd

class ICVLDataset(HandDataset):
    def __init__(self, fx = 241.42, fy = 241.42, halfu = 160, halfv = 120, path="Data/ICVL/", 
                sigmoid=1.5, image_size=128, kernel_size=7, label_size=64, 
                test_only=False, using_rotation=False, using_scale=False, using_flip=False, 
                scale_factor=60000, threshold=200, joint_number=16, dataset='train'):

        super(ICVLDataset, self).__init__(fx, fy, halfu, halfv, path, sigmoid, image_size, kernel_size,
                label_size, test_only, using_rotation, using_scale, using_flip, scale_factor, 
                threshold, joint_number, process_mode='uvd', dataset=dataset)

        Index = [0, 4, 5, 6]
        Mid = [0, 7, 8, 9]
        Ring = [0, 10, 11, 12]
        Small = [0, 13, 14, 15]
        Thumb = [0, 1, 2, 3]
        self.config = [Thumb, Index, Mid, Ring, Small]

    def build_data(self):
        if self.data_ready:
            print("Data is Already build~")
            return

        if not os.path.exists(os.path.join(self.path, "test.txt")):
            print("building text.txt ...")
            test_set = []
            with open(os.path.join(self.path, "Testing", "test_seq_1.txt"), "r") as f:
                lines = f.readlines()
            test_line = [line.strip() for line in lines if not line == "\n"]
            for line in test_line:
                words = line.split()
                path = words[0]
                words = words[1:]
                name = os.path.join(self.path, "Testing", "Depth", path)
                words = [name] + words
                test_set.append(" ".join(words))

            with open(os.path.join(self.path, "Testing", "test_seq_2.txt"), "r") as f:
                lines = f.readlines()
            test_line = [line.strip() for line in lines if not line == "\n"]
            for line in test_line:
                words = line.split()
                path = words[0]
                words = words[1:]
                name = os.path.join(self.path, "Testing", "Depth", path)
                words = [name] + words
                test_set.append(" ".join(words))

            print("saving text.txt ...")
            with open(os.path.join(self.path, "test.txt"), 'w') as f:
                f.write("\n".join(test_set))

        if not os.path.exists(os.path.join(self.path, "train.txt")):
            print("building text.txt ...")

            datatexts = []
            with open(os.path.join(self.path, "Training", "labels.txt"), 'r') as f:
                train_line = f.readlines()

            for line in train_line:
                words = line.split()
                path = words[0]
                words = words[1:]
                name = os.path.join(self.path, "Training", "Depth", path)
                words = [name] + words
                datatexts.append(" ".join(words))

            print('checking data ......')

            pool = mp.Pool(processes=os.cpu_count())
            processing = []
            for text in datatexts:
                r = pool.apply_async(self.check_text, (text, ))
                processing.append(r)

            traintxt = []
            with tqdm(total=len(datatexts)) as pbar:
                for r in processing:
                    text = r.get()
                    if text:
                        traintxt.append(text)
                    pbar.update(1)
            pool.close()
            
            print('{} / {} data can use to train'.format(len(traintxt), len(datatexts)))
            with open(os.path.join(self.path, "train.txt"), 'w') as f:
                f.write("\n".join(traintxt))
        
    def load_from_text(self, text):
        """
            This function decode the text and load the image with only hand and the uvd coordinate of joints
            OUTPUT:
                image, uvd
        """
        path, joint_uvd = super().decode_line_txt(text)

        try:
            image = plt.imread(path) * 65535
        except:
            print("{} do not exist!".format(path))
            raise ValueError("file do not exist")

        # crop the image by boundary box
        uv = joint_uvd[:, :2]
        left, top = np.min(uv, axis=0) - 20
        right, buttom = np.max(uv, axis=0) + 20
        left = max(int(left), 0)
        top = max(int(top), 0)
        right = min(int(right), 320)
        buttom = min(int(buttom), 240)
        MM = np.zeros(image.shape)
        MM[top:buttom, left:right] = 1
        image = image * MM

        # remove the background in the boundary box
        depth = joint_uvd[:, 2]
        depth_max = np.max(depth)
        depth_min = np.min(depth)
        image[image > depth_max + 50] = 0
        image[image < depth_min - 50] = 0

        return image, joint_uvd

class NYUDataset(HandDataset):
    def __init__(self, fx = 241.42, fy = 241.42, halfu = 320, halfv = 240, path="Data/NYU/", 
                sigmoid=1.5, image_size=128, kernel_size=7, label_size=64, 
                test_only=False, using_rotation=False, using_scale=False, using_flip=False, 
                scale_factor=180000, threshold=300, joint_number=14, dataset='train'):
        # TODO scale_factor and threshold need to be further tuned
        
        super(NYUDataset, self).__init__(fx, fy, halfu, halfv, path, sigmoid, image_size, kernel_size,
                label_size, test_only, using_rotation, using_scale, using_flip, scale_factor, 
                threshold, joint_number, process_mode='uvd', dataset=dataset)

        Index = [13, 1, 0]
        Mid = [13, 3, 2]
        Ring = [13, 5, 4]
        Small = [13, 7, 6]
        Thumb = [13, 10, 9, 8]
        PALM = [11, 13, 12]
        self.config = [Thumb, Index, Mid, Ring, Small, PALM]
        self.index = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]

    def _build_bin(self, input_name, output_name, center, threshold, uvd):
        try:
            _image = plt.imread(input_name)
        except:
            return False
        image = (_image[:,:,1] * 256 + _image[:,:,2]) * 255
        uv = uvd[:, :2]
        left, top = np.min(uv, axis=0) - 20
        right, buttom = np.max(uv, axis=0) + 20
        left = max(int(left), 0)
        top = max(int(top), 0)
        right = min(int(right), 2 * self.halfu)
        buttom = min(int(buttom), 2 * self.halfv)
        MM = np.zeros(image.shape)
        MM[top:buttom, left:right] = 1
        image = image * MM
        Mask = floodFillDepth(image, center, threshold)
        if np.sum(Mask) > 200:
            image = image * Mask
        index = np.argwhere(image > 0)
        top, left = np.min(index, axis=0)
        bottum, right = np.max(index, axis=0) + 1
        # write the bin file
        with open(output_name, 'wb') as f:
            data = struct.pack('i', 640)
            f.write(data)
            data = struct.pack('i', 480)
            f.write(data)
            data = struct.pack('i', left)
            f.write(data)
            data = struct.pack('i', top)
            f.write(data)
            data = struct.pack('i', right)
            f.write(data)
            data = struct.pack('i', bottum)
            f.write(data)
            for i in range(top, bottum):
                for j in range(left, right):
                    data = struct.pack('f', image[i, j])
                    f.write(data)
        return True

    def build_data(self):
        if self.data_ready:
            print("Data is Already build~")
            return
        pool = mp.Pool(processes=os.cpu_count())
        if not os.path.exists(os.path.join(self.path, "train.txt")):
            mat = loadmat(os.path.join(self.path, "train", "joint_data.mat"))
            uvds = mat['joint_uvd'][0]
            training_set = []
            os.mkdir(os.path.join(self.path, "bin_train"))
            processing = []
            for i in range(uvds.shape[0]):
                uvd = uvds[i]
                png = "depth_1_%07d.png" % (i+1)
                name = os.path.join(self.path, "bin_train", "%07d.bin" % (i+1))
                center = (int(float(uvd[32, 1])), int(float(uvd[32, 0])))
                r = pool.apply_async(self._build_bin, (os.path.join(self.path, "train", png), name, center, 100, uvd.copy()))
                # self._build_bin(os.path.join(self.path, "Training", "Depth", path), name, center)
                uvd = uvd[self.index]
                uvd = uvd.reshape((42,))
                words = [str(uvd[j]) for j in range(uvd.shape[0])]
                words = [name] + words
                processing.append((r, words))

            for r, words in processing:
                if r.get():
                    training_set.append(" ".join(words))

            with open(os.path.join(self.path, "train.txt"), 'w') as f:
                f.write("\n".join(training_set))

        if not os.path.exists(os.path.join(self.path, "test.txt")):
            os.mkdir(os.path.join(self.path, "bin_test"))
            test_set = []
            processing = []
            mat = loadmat(os.path.join(self.path, "test", "joint_data.mat"))
            uvds = mat['joint_uvd'][0]
            for i in range(uvds.shape[0]):
                uvd = uvds[i]
                png = "depth_1_%07d.png" % (i+1)
                name = os.path.join(self.path, "bin_test", "%07d.bin" % (i+1))
                center = (int(float(uvd[32, 1])), int(float(uvd[32, 0])))
                r = pool.apply_async(self._build_bin, (os.path.join(self.path, "test", png), name, center, 100, uvd.copy()))
                # self._build_bin(os.path.join(self.path, "Training", "Depth", path), name, center)
                uvd = uvd[self.index]
                uvd = uvd.reshape((42,))
                words = [str(uvd[j]) for j in range(uvd.shape[0])]
                words = [name] + words
                processing.append((r, words))

            for r, words in processing:
                if r.get():
                    test_set.append(" ".join(words)) 
                    
            with open(os.path.join(self.path, "test.txt"), 'w') as f:
                f.write("\n".join(test_set))

        pool.close()
        pool.join()
        
    def load_from_text(self, text):
        """
        This function decode the text and load the image with only hand and the uvd coordinate of joints
        OUTPUT:
            image, uvd
        """
        path, joint_uvd = super().decode_line_txt(text)
        img, left, top, right, bottom = load_bin(path)

        image = np.zeros((self.halfv * 2, self.halfu * 2))
        image[top:bottom, left:right] = img.copy()
        return image, joint_uvd

class HAND17Dataset(HandDataset):
    def __init__(self, fx = 475.065948, fy = 475.065857, halfu = 315.944855, halfv = 245.287079, path="Data/HAND17/", 
                sigmoid=1.5, image_size=128, kernel_size=7, label_size=64, 
                test_only=False, using_rotation=False, using_scale=False, using_flip=False, 
                scale_factor=100000, threshold=150, joint_number=21, dataset='train'):

        super(HAND17Dataset, self).__init__(fx, fy, halfu, halfv, path, sigmoid, image_size, kernel_size,
                label_size, test_only, using_rotation, using_scale, using_flip, scale_factor, 
                threshold, joint_number, process_mode='uvd' if dataset == 'train' else 'bb', dataset=dataset)

        Index = [0, 2, 9, 10, 11]
        Mid = [0, 3, 12, 13, 14]
        Ring = [0, 4, 15, 16, 17]
        Small = [0, 5, 18, 19, 20]
        Thumb = [0, 1, 6, 7, 8]
        self.config = [Thumb, Index, Mid, Ring, Small]

    def build_data(self):
        if self.data_ready:
            print("Data is Already build~")
            return

        print('building test data ......')
        with open(os.path.join(self.path, 'frame', 'BoundingBox.txt'), 'r') as f:
            test_text = f.read()

        with open(os.path.join(self.path, 'test.txt'), 'w') as f:
            f.write(test_text)

        print('checking train data ......')
        with open(os.path.join(self.path, 'training', 'Training_Annotation.txt'), 'r') as f:
            datatexts = f.readlines()

        pool = mp.Pool(processes=os.cpu_count())
        processing = []
        for text in datatexts:
            r = pool.apply_async(self.check_text, (text, ))
            processing.append(r)

        traintxt = []
        with tqdm(total=len(datatexts)) as pbar:
            for r in processing:
                text = r.get()
                if text:
                    traintxt.append(text)
                pbar.update(1)
        pool.close()
        
        print('{} / {} data can use to train'.format(len(traintxt), len(datatexts)))
        with open(os.path.join(self.path, "train.txt"), 'w') as f:
            f.writelines(traintxt)
        
    def load_from_text(self, text):
        """
            This function decode the text and load the image with only hand and the uvd coordinate of joints
            OUTPUT:
                image, uvd
        """
        path, joint_xyz = super().decode_line_txt(text)
        joint_uvd = self.xyz2uvd(joint_xyz)

        image = plt.imread(os.path.join(self.path, 'training', 'images', path)) * 65535

        # crop the image by boundary box
        uv = joint_uvd[:, :2]
        left, top = np.min(uv, axis=0) - 20
        right, buttom = np.max(uv, axis=0) + 20
        left = max(int(left), 0)
        top = max(int(top), 0)
        right = min(int(right), 640)
        buttom = min(int(buttom), 480)
        MM = np.zeros(image.shape)
        MM[top:buttom, left:right] = 1
        image = image * MM

        # remove the background in the boundary box
        depth = joint_uvd[:, 2]
        depth_max = np.max(depth)
        depth_min = np.min(depth)
        image[image > depth_max + 50] = 0
        image[image < depth_min - 50] = 0

        return image, joint_uvd

    def load_from_text_bb(self, text):
        l = text.strip().split()
        # print(l)
        path = l[0]
        ustart, vstart, du, dv = map(float, l[1:])

        image = plt.imread(os.path.join(self.path, 'frame', 'images', path)) * 65535

        # crop the image by boundary box
        MM = np.zeros(image.shape)
        MM[int(vstart):int(vstart+dv), int(ustart):int(ustart+du)] = 1
        image = image * MM

        # remove the background in the boundary box
        mean = np.mean(image[image > 0])
        _image = image.copy()
        _image[_image > mean + 100] = 0
        mean = np.mean(_image[_image > 0])
        image[image > mean + 100] = 0
        # image[image < mean - 200] = 0        

        return image
