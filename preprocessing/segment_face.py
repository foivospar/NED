import os
from skimage import io, img_as_float32, img_as_ubyte
from skimage.measure import label
import torch
import numpy as np
import argparse
from tqdm import tqdm
import cv2
from preprocessing.segmentation.simple_unet import UNet
from postprocessing.image_blending.image_blender import SoftErosion

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dir):
    # Returns list: [path1, path2, ...]
    image_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_files.append(path)
    return image_files

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image_pths, masks):
    # Make dirs
    mask_pths = [p.replace('/images/', '/masks/') for p in image_pths]
    out_paths = set(os.path.dirname(mask_pth) for mask_pth in mask_pths)
    for out_path in out_paths:
        mkdir(out_path)
        mkdir(out_path.replace('/masks', '/faces'))
    print('Saving results')
    for mask, image_pth in tqdm(zip(masks, image_pths), total=len(image_pths)):
        image = cv2.imread(image_pth)
        cv2.imwrite(image_pth.replace('/images/', '/masks/'), img_as_ubyte(mask))
        cv2.imwrite(image_pth.replace('/images/', '/faces/'), image*mask)

def dirs_exist(image_pths):
    mask_pths = [p.replace('/images/', '/masks/') for p in image_pths]
    out_paths = set(os.path.dirname(mask_pth) for mask_pth in mask_pths)
    return all([os.path.exists(out_path) for out_path in out_paths])

def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest)
    return labels_max

def get_face_masks(img_paths, predictor, smoother, device):
    print('Extracting face masks')
    masks = []
    prev_mask = None
    for i in tqdm(range(len(img_paths))):
        img = img_as_float32(io.imread(img_paths[i]))

        # convert to torch.tensor, change position of channel dim, and add batch dim
        im_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
        # predict face mask
        pred = predictor(im_tensor)  # 3-channel image for 3-wise segmentation (background, face, hair)
        mask = (pred.argmax(1, keepdim=True) == 1)
        _,mask = smoother(mask)   # soft erosion

        # convert to single-channel image
        mask = mask.squeeze(0).permute(1,2,0).cpu().numpy()

        if True in mask:
            # keep only the largest connected component if more than one found
            mask = getLargestCC(mask)
            prev_mask = mask
            masks.append(mask)
        else:
            print('No face mask detected, using previous mask')
            masks.append(prev_mask)

    return masks

def print_args(parser, args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------------------'
    print(message)

def main():
    print('---------- Face segmentation --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--celeb', type=str, default='JackNicholson', help='Path to celebrity folder.')
    args = parser.parse_args()

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Print Arguments
    print_args(parser, args)

    # Load pretrained face segmenter
    segmenter_path = 'preprocessing/segmentation/lfw_figaro_unet_256_2_0_segmentation_v1.pth'
    checkpoint = torch.load(segmenter_path)
    predictor = UNet(n_classes=3,feature_scale=1).to(device)
    predictor.load_state_dict(checkpoint['state_dict'])
    smooth_mask = SoftErosion(kernel_size=21, threshold=0.6).to(device)

    # Get the path of each image.
    images_dir = os.path.join(args.celeb, 'images')
    image_paths = get_image_paths(images_dir)

    if not dirs_exist(image_paths):
        masks = get_face_masks(image_paths, predictor, smooth_mask, device)
        save_results(image_paths, masks)
        print('DONE!')
    else:
        print('Face segmentation already done!')

if __name__=='__main__':
    main()
