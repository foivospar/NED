import torch
import numpy as np
import numpy as np
import os
import cv2
from PIL import Image
from scipy.spatial import distance

def reshape(tensors):
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]
    if tensors is None:
        return None
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:,:,0]
    elif image_numpy.shape[2] == 2: # uv image case
        zeros = np.zeros((image_numpy.shape[0], image_numpy.shape[1], 1)).astype(int)
        image_numpy = np.concatenate([image_numpy, zeros], 2)
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fit_ROI_in_frame(center, opt):
    center_w, center_h = center[0], center[1]
    center_h = torch.tensor(opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_h < opt.ROI_size // 2 else center_h
    center_w = torch.tensor(opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_w < opt.ROI_size // 2 else center_w
    center_h = torch.tensor(opt.loadSize - opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_h > opt.loadSize - opt.ROI_size // 2 else center_h
    center_w = torch.tensor(opt.loadSize - opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_w > opt.loadSize - opt.ROI_size // 2 else center_w
    return (center_w, center_h)

def crop_ROI(img, center, ROI_size):
    return img[..., center[1] - ROI_size // 2:center[1] + ROI_size // 2,
                    center[0] - ROI_size // 2:center[0] + ROI_size // 2]

def get_ROI(tensors, centers, opt):
    real_A, real_B, fake_B = tensors
    # Extract region of interest around the center.
    real_A_ROI = []
    real_B_ROI = []
    fake_B_ROI = []
    for t in range(centers.shape[0]):
        center = fit_ROI_in_frame(centers[t], opt)
        real_A_ROI.append(crop_ROI(real_A[t], center, opt.ROI_size))
        real_B_ROI.append(crop_ROI(real_B[t], center, opt.ROI_size))
        fake_B_ROI.append(crop_ROI(fake_B[t], center, opt.ROI_size))
    real_A_ROI = torch.stack(real_A_ROI, dim=0)
    real_B_ROI = torch.stack(real_B_ROI, dim=0)
    fake_B_ROI = torch.stack(fake_B_ROI, dim=0)
    return real_A_ROI, real_B_ROI, fake_B_ROI
