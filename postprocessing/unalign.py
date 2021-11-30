import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

IMAGE_EXTENSIONS = ['.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMAGE_EXTENSIONS)

def get_faces_a_paths(dir):
    # Returns list: [path1, path2, ...]
    faces_a_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in sorted(os.listdir(dir)):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            faces_a_files.append(path)
    return faces_a_files

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_faces(face_a_pths, faces, args):
    mkdir(os.path.join(args.celeb, args.exp_name, 'faces'))
    print('Saving results')
    for face_a_pth, face in tqdm(zip(face_a_pths, faces), total=len(faces)):
        cv2.imwrite(face_a_pth.replace('/faces_aligned/', '/faces/'), face)

def unalign(face_a_paths, args):
    faces = []
    print('Removing alignment from face images')
    for face_a_pth in tqdm(face_a_paths):
        mat_file = os.path.splitext(face_a_pth.replace(f'/{args.exp_name}/faces_aligned', '/align_transforms'))[0]+'.txt'
        mask_file = face_a_pth.replace(f'/{args.exp_name}/faces_aligned', '/masks')
        face_a = cv2.imread(face_a_pth)
        mat = np.loadtxt(mat_file)
        mask = cv2.imread(mask_file)
        face = cv2.warpAffine(face_a, mat, (face_a.shape[1], face_a.shape[0]), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LANCZOS4)
        face[np.where(mask==0)] = 0

        faces.append(face)

    return faces

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
    print('---------- Undo face alignment --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, default='JackNicholson', help='Path to celebrity folder.')
    parser.add_argument('--exp_name', type=str, default='Pacino', help='Experiment sub-folder')
    args = parser.parse_args()

    # Print Arguments
    print_args(parser, args)

    # Get the path of each aligned face image.
    faces_a_dir = os.path.join(args.celeb, args.exp_name, 'faces_aligned')
    face_a_paths = get_faces_a_paths(faces_a_dir)

    if not os.path.exists(faces_a_dir.replace('/faces_aligned', '/faces')):
        faces = unalign(face_a_paths, args)
        save_faces(face_a_paths, faces, args)
        print('DONE!')
    else:
        print('Face un-alignment already done!')

if __name__=='__main__':
    main()
