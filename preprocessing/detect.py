import os
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import torch
import argparse
from facenet_pytorch import MTCNN, extract_face
from tqdm import tqdm
from shutil import rmtree

VID_EXTENSIONS = ['.mp4']

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

def save_images(images, folder, split, start_i, args):
    for i in range(len(images)):
        n_frame = "{:06d}".format(i + start_i)
        part = "{:06d}".format((i + start_i) // args.seq_length) if split else ""
        save_dir = os.path.join(args.celeb, folder, part)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(images[i], os.path.join(save_dir, n_frame + '.png'), transpose = folder =='images')

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

def smooth_boxes(boxes, previous_box, args):
    # Check if there are None boxes.
    if boxes[0] is None:
        boxes[0] = previous_box
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = next((item for item in boxes[i+1:] if item is not None), boxes[i-1])
    boxes = [box[0] for box in boxes]   # if more than one faces detected, keep the one with the heighest probability
    # Smoothen boxes
    old_boxes = np.array(boxes)
    window_length = min(args.window_length, old_boxes.shape[0])
    if window_length % 2 == 0:
        window_length -= 1
    smooth_boxes = np.concatenate([ndimage.median_filter(old_boxes[:,i], size=window_length, mode='reflect').reshape((-1,1)) for i in range(4)], 1)
    # Make boxes square.
    for i in range(len(smooth_boxes)):
        offset_w = smooth_boxes[i][2] - smooth_boxes[i][0]
        offset_h = smooth_boxes[i][3] - smooth_boxes[i][1]
        offset_dif = (offset_h - offset_w) / 2
        # width
        smooth_boxes[i][0] = smooth_boxes[i][2] - offset_w - offset_dif
        smooth_boxes[i][2] = smooth_boxes[i][2] + offset_dif
        # height - center a bit lower
        smooth_boxes[i][3] = smooth_boxes[i][3] + args.height_recentre * offset_h
        smooth_boxes[i][1] = smooth_boxes[i][3] - offset_h

    return smooth_boxes

def get_faces(detector, images, previous_box, args):
    ret_faces = []
    ret_boxes = []

    all_boxes = []
    all_imgs = []

    # Get bounding boxes
    for lb in np.arange(0, len(images), args.mtcnn_batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+args.mtcnn_batch_size]]
        boxes, _ = detector.detect(imgs_pil)
        all_boxes.extend(boxes)
        all_imgs.extend(imgs_pil)
    # Temporal smoothing
    boxes = smooth_boxes(all_boxes, previous_box, args)
    # Crop face regions.
    for img, box in zip(all_imgs, boxes):
        face = extract_face(img, box, args.cropped_image_size, args.margin)
        ret_faces.append(face)
        # Find real bbox   (taken from https://github.com/timesler/facenet-pytorch/blob/54c869c51e0e3e12f7f92f551cdd2ecd164e2443/models/utils/detect_face.py#L358)
        margin = [
            args.margin * (box[2] - box[0]) / (args.cropped_image_size - args.margin),
            args.margin * (box[3] - box[1]) / (args.cropped_image_size - args.margin),
        ]
        raw_image_size = img.size
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]
        ret_boxes.append(box)

    return ret_faces, ret_boxes, boxes[-1]

def detect_and_save_faces(detector, mp4_path, split, start_i, args):

    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    images = []
    previous_box = None

    print('Reading %s, extracting faces, and saving images' % mp4_path)
    for i in tqdm(range(n_frames)):
        _, image = reader.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(images) < args.filter_length:
            images.append(image)
        # else, detect faces in sequence and create new list
        else:
            face_images, boxes, previous_box = get_faces(detector, images, previous_box, args)
            save_images(tensor2npimage(face_images), 'images', split, start_i, args)

            if args.save_full_frames:
                save_images(images, 'full_frames', split, start_i, args)

            if args.save_videos_info:
                videos_file = os.path.splitext(mp4_path)[0] + '.txt'
                if not os.path.exists(videos_file):
                    vfile = open(videos_file, "a")
                    vfile.write('{} {} fps {} frames\n'.format(mp4_path, fps, n_frames))
                    vfile.close()
                for box in boxes:
                    vfile = open(videos_file, "a")
                    np.savetxt(vfile, np.expand_dims(box,0))
                    vfile.close()

            start_i += len(images)
            images = [image]
    # last sequence
    face_images, boxes, _ = get_faces(detector, images, previous_box, args)
    save_images(tensor2npimage(face_images), 'images', split, start_i, args)

    if args.save_full_frames:
        save_images(images, 'full_frames', split, start_i, args)

    if args.save_videos_info:
        videos_file = os.path.splitext(mp4_path)[0] + '.txt'
        if not os.path.exists(videos_file):
            vfile = open(videos_file, "a")
            vfile.write('{} {} fps {} frames\n'.format(mp4_path, fps, n_frames))
            vfile.close()
        for box in boxes:
            vfile = open(videos_file, "a")
            np.savetxt(vfile, np.expand_dims(box,0))
            vfile.close()

    start_i += len(images)

    reader.release()
    return start_i


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
    print('-------------- Face detection -------------- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--celeb', type=str, default='JackNicholson', help='Path to celebrity folder.')
    parser.add_argument('--save_videos_info', action='store_true', help='Whether to save videos meta-data (fps, #frames, bounding boxes) in .txt file')
    parser.add_argument('--save_full_frames', action='store_true', help='Whether to save full video frames (for reproducing the original clip)')
    parser.add_argument('--mtcnn_batch_size', default=8, type=int, help='The number of frames for face detection.')
    parser.add_argument('--select_largest', action='store_true', help='In case of multiple detected faces, keep the largest (if specified), or the one with the highest probability')
    parser.add_argument('--cropped_image_size', default=256, type=int, help='The size of frames after cropping the face.')
    parser.add_argument('--margin', default=70, type=int, help='.')
    parser.add_argument('--filter_length', default=500, type=int, help='Number of consecutive bounding boxes to be filtered')
    parser.add_argument('--window_length', default=49, type=int, help='savgol filter window length.')
    parser.add_argument('--height_recentre', default=0.0, type=float, help='The amount of re-centring bounding boxes lower on the face.')
    parser.add_argument('--split', action='store_true', help='Whether to split video sequence to sub-sequences (for training)')
    parser.add_argument('--seq_length', default=50, type=int, help='The number of frames for each training sub-sequence.')

    args = parser.parse_args()
    print_args(parser, args)

    # check if face detection has already been done
    images_dir = os.path.join(args.celeb, 'images')
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

        # subfolder containing videos
        videos_path = os.path.join(args.celeb, 'videos')

        # Store video paths in list.
        mp4_paths = get_video_paths(videos_path)
        n_mp4s = len(mp4_paths)
        print('Number of videos to process: %d \n' % n_mp4s)

        # Initialize the MTCNN face  detector.
        detector = MTCNN(image_size=args.cropped_image_size, select_largest = args.select_largest, margin=args.margin, post_process=False, device=device)

        # Run detection
        n_completed = 0
        start_i = 0
        for path in mp4_paths:
            n_completed += 1
            start_i = detect_and_save_faces(detector, path, args.split, start_i, args)
            print('(%d/%d) %s [SUCCESS]' % (n_completed, n_mp4s, path))
        if args.split:
            # delete last sub-sequence if has less than "seq_length" frames
            last_folder = os.path.join(images_dir, sorted(os.listdir(images_dir))[-1])
            if len(os.listdir(last_folder))!=args.seq_length:
                rmtree(last_folder)
        print('DONE!')

if __name__ == "__main__":
    main()
