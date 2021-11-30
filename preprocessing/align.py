import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

FILE_EXTENSIONS = ['.txt']

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in FILE_EXTENSIONS)

def get_mats_paths(dir):
    # Returns list: [path1, path2, ...]
    mats_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                mats_files.append(path)
    return mats_files

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_aligned(mat_pths, aligned, args):
    # Make dirs
    out_paths = []
    if args.faces_and_masks:
        out_paths += [p.replace('/align_transforms/', '/faces_aligned/') for p in mat_pths]
        out_paths += [p.replace('/align_transforms/', '/masks_aligned/') for p in mat_pths]
    if args.renderings:
        out_paths += [p.replace('/align_transforms/', '/renderings_aligned/') for p in mat_pths]
    if args.shapes:
        out_paths += [p.replace('/align_transforms/', '/shapes_aligned/') for p in mat_pths]
    if args.nmfcs:
        out_paths += [p.replace('/align_transforms/', '/nmfcs_aligned/') for p in mat_pths]
    if args.landmarks:
        out_paths += [p.replace('/align_transforms/', '/eye_landmarks_aligned/') for p in mat_pths]
    out_paths = set(os.path.dirname(out_pth) for out_pth in out_paths)
    for out_path in out_paths:
        mkdir(out_path)

    print('Saving results')
    for al, mat_pth in tqdm(zip(aligned, mat_pths), total=len(mat_pths)):
        al_iter = iter(al)
        if args.faces_and_masks:
            face_a, mask_a = next(al_iter)
            face_file = os.path.splitext(mat_pth.replace('/align_transforms/', '/faces_aligned/'))[0] + '.png'
            mask_file = os.path.splitext(mat_pth.replace('/align_transforms/', '/masks_aligned/'))[0] + '.png'
            cv2.imwrite(face_file, face_a)
            cv2.imwrite(mask_file, mask_a)
        if args.renderings:
            rendering_a = next(al_iter)
            rendering_file = os.path.splitext(mat_pth.replace('/align_transforms/', '/renderings_aligned/'))[0] + '.png'
            cv2.imwrite(rendering_file, rendering_a)
        if args.shapes:
            shape_a = next(al_iter)
            shape_file = os.path.splitext(mat_pth.replace('/align_transforms/', '/shapes_aligned/'))[0] + '.png'
            cv2.imwrite(shape_file, shape_a)
        if args.nmfcs:
            nmfc_a = next(al_iter)
            nmfc_file = os.path.splitext(mat_pth.replace('/align_transforms/', '/nmfcs_aligned/'))[0] + '.png'
            cv2.imwrite(nmfc_file, nmfc_a)
        if args.landmarks:
            lands_a = next(al_iter)
            lands_file = mat_pth.replace('/align_transforms/', '/eye_landmarks_aligned/')
            np.savetxt(lands_file, lands_a)

def dirs_exist(mat_pths, args):
    out_paths = []
    if args.faces_and_masks:
        out_paths += [p.replace('/align_transforms/', '/faces_aligned/') for p in mat_pths]
        out_paths += [p.replace('/align_transforms/', '/masks_aligned/') for p in mat_pths]
    if args.renderings:
        out_paths += [p.replace('/align_transforms/', '/renderings_aligned/') for p in mat_pths]
    if args.shapes:
        out_paths += [p.replace('/align_transforms/', '/shapes_aligned/') for p in mat_pths]
    if args.nmfcs:
        out_paths += [p.replace('/align_transforms/', '/nmfcs_aligned/') for p in mat_pths]
    if args.landmarks:
        out_paths += [p.replace('/align_transforms/', '/eye_landmarks_aligned/') for p in mat_pths]
    out_paths = set(os.path.dirname(out_pth) for out_pth in out_paths)
    return all([os.path.exists(out_path) for out_path in out_paths])

def transform_points(points, mat):
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points

def align(mat_paths, args):
    rets = []
    print('Aligning images')
    for mat_pth in tqdm(mat_paths):
        ret = []
        mat = np.loadtxt(mat_pth)
        if args.faces_and_masks:
            face_pth = os.path.splitext(mat_pth.replace('/align_transforms/', '/faces/'))[0]+'.png'
            face = cv2.imread(face_pth)
            mask_pth = os.path.splitext(mat_pth.replace('/align_transforms/', '/masks/'))[0]+'.png'
            mask = cv2.imread(mask_pth)
            face_a = cv2.warpAffine(face, mat, (face.shape[1], face.shape[0]), flags=cv2.INTER_LANCZOS4)
            mask_a = cv2.warpAffine(mask, mat, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
            face_a[np.where(mask_a==0)] = 0

            ret.append((face_a, mask_a))
        if args.renderings:
            r_pth = os.path.splitext(mat_pth.replace('/align_transforms/', '/renderings/'))[0]+'.png'
            rendering = cv2.imread(r_pth)
            rendering_a = cv2.warpAffine(rendering, mat, (rendering.shape[1], rendering.shape[0]), flags=cv2.INTER_LANCZOS4)

            ret.append(rendering_a)
        if args.shapes:
            s_pth = os.path.splitext(mat_pth.replace('/align_transforms/', '/shapes/'))[0]+'.png'
            shape = cv2.imread(s_pth)
            shape_a = cv2.warpAffine(shape, mat, (shape.shape[1], shape.shape[0]), flags=cv2.INTER_LANCZOS4)

            ret.append(shape_a)
        if args.nmfcs:
            n_pth = os.path.splitext(mat_pth.replace('/align_transforms/', '/nmfcs/'))[0]+'.png'
            nmfc = cv2.imread(n_pth)
            nmfc_a = cv2.warpAffine(nmfc, mat, (nmfc.shape[1], nmfc.shape[0]), flags=cv2.INTER_LANCZOS4)

            ret.append(nmfc_a)
        if args.landmarks:
            land_file = mat_pth.replace('/align_transforms', '/eye_landmarks/')
            lands = np.loadtxt(land_file)
            lands_a = transform_points(lands, mat)

            ret.append(lands_a)
        rets.append(ret)

    return rets

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
    print('---------- Face alignment --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, default='JackNicholson', help='Path to celebrity folder.')
    parser.add_argument('--faces_and_masks', action='store_true', help='Whether to aling face and mask images')
    parser.add_argument('--renderings', action='store_true', help='Whether to aling renderings')
    parser.add_argument('--shapes', action='store_true', help='Whether to aling shape images')
    parser.add_argument('--nmfcs', action='store_true', help='Whether to aling nmfc images')
    parser.add_argument('--landmarks', action='store_true', help='Whether to aling eye-landmarks')
    args = parser.parse_args()

    # Print Arguments
    print_args(parser, args)

    # Get the path of each transformation file.
    mats_dir = os.path.join(args.celeb, 'align_transforms')
    mat_paths = get_mats_paths(mats_dir)

    if not dirs_exist(mat_paths, args):
        aligned = align(mat_paths, args)
        save_aligned(mat_paths, aligned, args)
        print('DONE!')
    else:
        print('Face alignment already done!')

if __name__=='__main__':
    main()
