import os
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import torch
import argparse
from facenet_pytorch import MTCNN, extract_face
from tqdm import tqdm

VID_EXTENSIONS = ['.mp4', '.avi']

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)

def tensor2npimage(image_tensor, imtype=np.uint8):
    # Tesnor in range [0,255]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2npimage(image_tensor[i], imtype))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, transpose = True):
    if transpose:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def get_video_paths(dir):
    # Returns list of paths to video files
    video_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_video_file(fname):
                path = os.path.join(root, fname)
                video_files.append(path)
    return video_files

def detect_and_save_faces(detector, mp4_path, args):

    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    ann_path = os.path.join(args.annotations_path, os.path.basename(mp4_path)[:-4]+'.txt')
    if os.path.exists(ann_path):
        with open(ann_path, 'r') as f:
            labels = [int(line.rstrip('\n')) for line in f.readlines()[1:]]
        if len(labels)<n_frames:
            print('Skipping this video!')
            return
    else:
        print('Skipping this video!')
        return

    print('Reading %s, extracting faces, and saving images' % mp4_path)
    for i in tqdm(range(n_frames)):
        _, image = reader.read()
        if labels[i]>=0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, _ = detector.detect([Image.fromarray(image)])
            box = boxes[0]
            if box is not None:
                box = box[0] # if more than one faces detected, keep one

                offset_w = box[2] - box[0]
                offset_h = box[3] - box[1]
                offset_dif = (offset_h - offset_w) / 2
                # width
                box[0] = box[2] - offset_w - offset_dif
                box[2] = box[2] + offset_dif
                # height - center a bit lower
                box[3] = box[3] + args.height_recentre * offset_h
                box[1] = box[3] - offset_h

                face = extract_face(Image.fromarray(image), box, args.cropped_image_size, args.margin)
                n_frame = "{:06d}".format(i)
                save_dir = os.path.join(args.save_dir, 'images', os.path.basename(mp4_path)[:-4])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_image(tensor2npimage(face), os.path.join(save_dir, n_frame + '.png'))


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
    print('-------------- Face detection in aff-wild2 videos -------------- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--videos_path', type=str, default='.', help='Path to aff-wild2 videos (train set).')
    parser.add_argument('--annotations_path', type=str, default='.', help='Path to aff-wild2 annotations (train set).')
    parser.add_argument('--save_dir', type=str, default='./aff-wild2_frames', help='Path to save the extracted frames.')
    parser.add_argument('--cropped_image_size', default=256, type=int, help='The size of frames after cropping the face.')
    parser.add_argument('--margin', default=70, type=int, help='.')
    parser.add_argument('--height_recentre', default=0.0, type=float, help='The amount of re-centring bounding boxes lower on the face.')

    args = parser.parse_args()
    print_args(parser, args)

    # check if face detection has already been done
    images_dir = os.path.join(args.save_dir, 'images')
    if os.path.isdir(images_dir):
        print('Face detection already done!')

    else:
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


        # Store video paths in list.
        mp4_paths = get_video_paths(args.videos_path)
        n_mp4s = len(mp4_paths)
        print('Number of videos to process: %d \n' % n_mp4s)

        # Initialize the MTCNN face  detector.
        detector = MTCNN(image_size=args.cropped_image_size, select_largest = True, margin=args.margin, post_process=False, device=device)

        # Run detection
        n_completed = 0
        for path in mp4_paths:
            n_completed += 1
            detect_and_save_faces(detector, path, args)
            print('(%d/%d) %s [SUCCESS]' % (n_completed, n_mp4s, path))
        print('DONE!')

if __name__ == "__main__":
    main()
