import os
import argparse
import cv2
import numpy as np
import torch
from skimage import img_as_float32, img_as_ubyte
from tqdm import tqdm
from postprocessing.image_blending.image_blender import Blend

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dir):
    # Returns list: [path1, path2, ...]
    image_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in sorted(os.listdir(dir)):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            image_files.append(path)
    return image_files

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_bboxes(dir):
    # Returns list with bounding boxes
    boxes = []
    txt_files = [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if '.txt' in f]
    for t in txt_files:
        boxes.extend(list(np.loadtxt(t, skiprows=1).astype(int)))
    return boxes

def blend_and_save_images(face_pths, blender, args):

    mkdir(os.path.join(args.celeb, args.exp_name, 'full_frames'))
    if args.save_images:
        mkdir(os.path.join(args.celeb, args.exp_name, 'images'))

    # Load original bounding boxes
    videos_folder = os.path.join(args.celeb, 'videos')
    boxes = load_bboxes(videos_folder)

    print('Bleding and saving images')
    for face_pth in tqdm(face_pths):
        full_frame_pth = face_pth.replace(f'/{args.exp_name}/faces', '/full_frames')
        mask_pth = face_pth.replace(f'/{args.exp_name}/faces', '/masks')
        img_pth = face_pth.replace(f'/{args.exp_name}/faces', '/images')

        full_frame = img_as_float32(cv2.imread(full_frame_pth))

        ind = int(os.path.splitext(os.path.basename(face_pth))[0])
        box = boxes[ind]

        if args.resize_first:
            imgA = full_frame[box[1]:box[3], box[0]:box[2]]
        else:
            imgA = img_as_float32(cv2.imread(img_pth))
        imgB = img_as_float32(cv2.imread(face_pth))
        mask = img_as_float32(cv2.imread(mask_pth))
        shape = imgB.shape

        new = blender(imgA, imgB, mask)
        if args.resize_first:
            full_frame[box[1]:box[3], box[0]:box[2]] = new
        else:
            full_frame[box[1]:box[3], box[0]:box[2]] = cv2.resize(new, (box[2]-box[0], box[3]-box[1]), interpolation = cv2.INTER_LANCZOS4)
        cv2.imwrite(face_pth.replace('/faces/', '/full_frames/'), img_as_ubyte(np.clip(full_frame,0,1)))

        if args.save_images:
            if new.shape!=shape:
                new = cv2.resize(new, (shape[1], shape[0]), interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(face_pth.replace('/faces/', '/images/'), img_as_ubyte(np.clip(new,0,1)))

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
    print('---------- Image blending --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--celeb', type=str, default='JackNicholson', help='Path to celebrity folder.')
    parser.add_argument('--exp_name', type=str, default='Pacino', help='Experiment sub-folder')
    parser.add_argument('--resize_first', action='store_true', help='If specified, first resize image, then blend, else reversely')
    parser.add_argument('--save_images', action='store_true', help='If specified, save the cropped blended images, apart from the full frames')
    parser.add_argument('--method', type=str, default='pyramid', choices = ['copy_paste', 'pyramid', 'poisson'], help='Blending method')
    parser.add_argument('--n_levels', type=int, default=4, help='Number of levels of the laplacian pyramid, if pyramid blending is used')
    parser.add_argument('--n_levels_copy', type=int, default=0, help='Number of levels at the top of the laplacian pyramid to copy from image A')
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

    faces_path = os.path.join(args.celeb, args.exp_name, 'faces')

    face_paths = get_image_paths(faces_path)

    if not os.path.exists(faces_path.replace('/faces', '/full_frames')):
        blender = Blend(method=args.method, n_levels=args.n_levels, n_levels_copy=args.n_levels_copy, device = device)
        blend_and_save_images(face_paths, blender, args)
        print('DONE!')
    else:
        print('Image blending already done!')

if __name__=='__main__':
    main()
