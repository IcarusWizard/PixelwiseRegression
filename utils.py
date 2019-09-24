import matplotlib.pyplot as plt
import numpy as np
from struct import unpack
import os, re, cv2
from random import shuffle
from matplotlib.font_manager import FontProperties
from matplotlib import rc
from queue import Queue
import torch
import random

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')
font.set_size(20)

font_title = FontProperties()
font_title.set_family('serif')
font_title.set_name('Times New Roman')
font_title.set_style('italic')
font_title.set_size(25)

def generate_com_filter(size_u, size_v):
    """
    generate com base conv filter
    """
    center_u = size_u // 2
    center_v = size_v // 2
    _filter = np.zeros((size_v, size_u, 2)) # 0 channel is for u, 1 channel is for v
    for i in range(size_v):
        for j in range(size_u):
            _filter[i, j, 0] = (j - center_u) / (size_u - 1)
            _filter[i, j, 1] = (i - center_v) / (size_v - 1)
    return _filter

def generate_heatmap(img_size, u, v):
    """
        Return heatmap base on the location
    """
    try:
        heatmap = np.zeros((img_size, img_size))
        low_u = int(np.floor(u))
        low_v = int(np.floor(v))
        du = u - low_u
        dv = v - low_v

        min_d = max(du + dv - 1, 0)
        max_d = min(du, dv)
        d = (max_d + min_d) / 2
        b = du - d
        c = dv - d
        a = 1 + d - du - dv

        heatmap[low_v, low_u] = a
        heatmap[low_v, low_u + 1] = b
        heatmap[low_v + 1, low_u] = c
        heatmap[low_v + 1, low_u + 1] = d

        return heatmap
    except:
        raise Exception("Out of range")

def generate_kernel(heatmap, kernel_size = 3, sigmoid = 1.5):
    return cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigmoid)

def random_rotated(_img, _uvd):
    # random rotated the img within -30 ~ 30
    img = _img.copy()
    uvd = _uvd.copy()

    angle = random.random() * 60 - 30
    size = img.shape[0]
    M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
    rot_img = cv2.warpAffine(img, M, (size, size))
    
    angle = angle / 180.0 * np.pi
    Rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    uvd[:, :2] = uvd[:, :2] @ Rot.T

    return rot_img, uvd

def draw_skeleton(img, joints, config, *, rP = 3, linewidth = 1):
    if joints.shape[0] == 14:
        img3D = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            img3D[:, :, i] = img
        img3D = img3D / np.max(img3D)
        img3D = img3D * 0.5 + 0.25
        img3D = 1 - img3D
        _joint = [(int(joints[i][0]), int(joints[i][1])) for i in range(joints.shape[0])]
        colors = [(1, 0, 0), (0.5, 0.5, 0), (0, 1, 0), (0, 0.5, 0.5), (0, 0, 1), (0.5, 0.5, 0.5)]
        for i in range(6):
            for index in config[i]:
                cv2.circle(img3D, _joint[index], rP, colors[i], -1)
            for j in range(len(config[i]) - 1):
                cv2.line(img3D, _joint[config[i][j]], _joint[config[i][j+1]], colors[i], linewidth)
        return img3D
    else:
        img3D = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            img3D[:, :, i] = img
        img3D = img3D / np.max(img3D)
        img3D = img3D * 0.5 + 0.25
        img3D = 1 - img3D
        _joint = [(int(joints[i][0]), int(joints[i][1])) for i in range(joints.shape[0])]
        colors = [(1, 0, 0), (0.5, 0.5, 0), (0, 1, 0), (0, 0.5, 0.5), (0, 0, 1)]
        for i in range(5):
            for index in config[i]:
                cv2.circle(img3D, _joint[index], rP, colors[i], -1)
            for j in range(len(config[i]) - 1):
                cv2.line(img3D, _joint[config[i][j]], _joint[config[i][j+1]], colors[i], linewidth)
        return img3D

def draw_skeleton_torch(img, joints, config, *, rP = 3, linewidth = 1):
    img = img.numpy()[0]
    size = img.shape[0]
    joints = joints.numpy() * (size - 1) + np.array([size // 2, size // 2, 0])
    img3D = draw_skeleton(img, joints, config, rP=rP, linewidth=linewidth)
    img3D = torch.from_numpy(img3D).float().permute(2, 0, 1).contiguous()
    return img3D

def findmax_batch(img):
    imgsize = img.shape[1]
    temimg = img.reshape((img.shape[0], imgsize ** 2, img.shape[3]))
    index = np.argmax(temimg, axis=1)
    return index // imgsize, index % imgsize

def find_max(img):
    length = img.shape[1]
    index = np.argmax(img)
    return index // length, index % length

def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = - (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x

def center_crop(img, center, window):
    u, v = center
    u = int(u)
    v = int(v)
    shift = window // 2
    dimg = np.pad(img, ((shift, shift), (shift, shift)), "constant", constant_values=0)
    return dimg[u:u+2*shift, v:v+2*shift]

def cropimg(imgs, features, predict_us, predict_vs, output_channel, img_window=24):
    img_tensors = []
    feature_tensors = []
    img_size = imgs.shape[1]
    feature_size = features.shape[1]
    scale = img_size // feature_size
    feature_window = img_window // scale 
    for i in range(imgs.shape[0]):
        img = np.pad(imgs[i], ((img_window // 2, img_window // 2), (img_window // 2, img_window // 2)), mode='constant', constant_values=0)
        feature = np.pad(features[i], ((feature_window // 2, feature_window // 2), (feature_window // 2, feature_window // 2), (0, 0)), mode='constant', constant_values=0)
        predict_u = predict_us[i]
        predict_v = predict_vs[i]
        sub_tensor_img = []
        sub_tensor_feature = []
        for j in range(output_channel):
            u = predict_u[j]
            v = predict_v[j]
            sub_tensor_img.append(center_crop(img, (u + img_window // 2,v + img_window // 2), img_window)[:,:,np.newaxis])
            sub_tensor_feature.append(center_crop(feature, (u // scale + feature_window // 2, v // scale + feature_window // 2), feature_window))
        img_tensors.append(np.concatenate(sub_tensor_img, axis=2)[np.newaxis])
        feature_tensors.append(np.concatenate(sub_tensor_feature, axis=2)[np.newaxis])
    return np.concatenate(img_tensors, axis=0), np.concatenate(feature_tensors, axis=0)

def compute_error(img, label, norm_d, para, focal_x, focal_y, ux, uy, ground_truth):
    ushift, vshift, left, top, origin, min_value, scale = para
    x = np.zeros((21, 3))
    x[:, 2] = (norm_d + 1) * scale / 2 + min_value
    for i in range(21):
        x[i, 1], x[i, 0] = find_max(label[:, :, i])
    x[:, 0] = (x[:, 0] / (img.shape[0] - 1) * (origin - 1)) - ushift + left
    x[:, 1] = (x[:, 1] / (img.shape[0] - 1) * (origin - 1)) - vshift + top
    x = pixel2world(x[np.newaxis], focal_x, focal_y, ux, uy)
    return np.sqrt(np.sum((x - ground_truth[np.newaxis]) ** 2, axis=2))

def batch_compute_error(imgs, labels, norm_d, paras, focal_x, focal_y, ux, uy, ground_truths):
    return np.concatenate([compute_error(imgs[i], labels[i], norm_d[i], paras[i], focal_x, focal_y, ux, uy, ground_truths[i]) for i in range(imgs.shape[0])], axis=0)

def fill_to_square(img):
    m, n = img.shape
    ushift, vshift = 0, 0
    if m > n:
        img = np.concatenate([np.zeros((m, (m - n) // 2)), img, np.zeros((m, (m - n) // 2 + (m - n) % 2))], axis=1)
        ushift = (m - n) // 2
    elif m < n:
        img = np.concatenate([np.zeros(((n - m) // 2, n)), img, np.zeros(((n - m) // 2 + (n - m) % 2, n))], axis=0)
        vshift = (n - m) // 2
    return img, ushift, vshift

def filte_img(img):
    value = img[img > 0]
    mean_val = np.mean(value)
    std_val = np.std(value)
    img[np.abs(img - mean_val) / std_val > 3] = 0
    return img

def norm_img(img):
    value = img[img > 0]
    mean = np.mean(value)
    max_value = np.max(value)
    min_value = np.min(value)
    scale = max_value - min_value
    img[img > 0] = 2 * (value - min_value) / scale - 1
    return img, min_value, scale

def findStep(modelpath):
    if not os.path.exists(modelpath) or len(os.listdir(modelpath)) == 0:
        print('model path not exist, train with initial point')
        return 0
    else:
        l = os.listdir(modelpath)
        step = 0
        match = re.compile(r'ckpt-(\d*)')
        for name in l:
            result = match.findall(name)
            if not len(result) == 0:
                step = max(step, int(result[0]))
        return step

def load_bin(filename):
    with open(filename, 'rb') as f:
        img_width, img_height, left, top, right, bottom = map(lambda s: int.from_bytes(s, 'little'), [f.read(4) for i in range(6)])
        img = np.zeros((bottom - top, right - left))
        for i in range(bottom - top):
            for j in range(right - left):
                img[i][j] = unpack('f', f.read(4))[0]
        return img, left, top, right, bottom

def norm(A):
    max_val = np.max(A)
    min_val = np.min(A)
    return (A - min_val) / (max_val - min_val)

def build_gauss(img_size, m, n, kernel_size):
    gauss = np.zeros(img_size)
    gauss[m, n] = 1
    gauss = norm(cv2.GaussianBlur(gauss, kernel_size, 0))
    return gauss[:,:,np.newaxis]

def floodFillDepth(img, startPoint, threshold):
    mask = np.zeros(img.shape, dtype=np.uint8)
    h, w = img.shape
    q = Queue()
    q.put((startPoint, startPoint))
    while not q.empty():
        prepoint, point = q.get()
        x, y = point
        if(x < h and x >= 0 and y < w and y >= 0 and (mask[point] == 0) and np.abs(img[prepoint] - img[point]) <= threshold):
            mask[point] = 1
            q.put((point, (x + 1, y)))
            q.put((point, (x - 1, y)))
            q.put((point, (x, y + 1)))
            q.put((point, (x, y - 1)))
    return mask

def flip(img):
    flip_img = np.zeros(img.shape)
    for j in range(img.shape[1]):
        flip_img[:, j] = img[:, img.shape[1] - j - 1].copy()
    return flip_img

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename, eval_mode=False):
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    if eval_mode:
        model.eval()

def step_loader(dataloder):
    data = iter(dataloder)
    while True:
        try:
            x = next(data)
        except:
            data = iter(dataloder)
            x = next(data)
        yield x

def recover_uvd(uvd, box_size, com, threshold):
    uvd[:, :, :2] = uvd[:, :, :2] * (box_size - 1).view(-1, 1, 1)
    uvd[:, :, 2] = uvd[:, :, 2] * threshold
    uvd = uvd + com.unsqueeze(1)

    return uvd