import cv2
import os
import torch
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import argparse
from tqdm import tqdm
import face_alignment

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

######## ported from "https://github.com/iperov/DeepFaceLab/blob/master/core/mathlib/umeyama.py"
def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T

def save_results(image_pths, landmarks, align_transforms):
    # Make dirs
    out_pths = [p.replace('/images/', '/eye_landmarks/') for p in image_pths]
    if len(align_transforms)>0:
        out_pths += [p.replace('/images/', '/align_transforms/') for p in image_pths]
    out_paths = set(os.path.dirname(out_pth) for out_pth in out_pths)
    for out_path in out_paths:
        mkdir(out_path)
    print('Saving results')
    for i in tqdm(range(len(image_pths))):
        image_pth = image_pths[i]
        landmark = landmarks[i]
        landmark_file = os.path.splitext(image_pth.replace('/images/', '/eye_landmarks/'))[0] + '.txt'
        np.savetxt(landmark_file, landmark)
        if len(align_transforms)>0:
            align_file = os.path.splitext(image_pth.replace('/images/', '/align_transforms/'))[0] + '.txt'
            tr = align_transforms[i]
            np.savetxt(align_file, tr)

def dirs_exist(image_pths):
    lnd_pths = [p.replace('/images/', '/eye_landmarks/') for p in image_pths]
    out_paths = set(os.path.dirname(lnd_pth) for lnd_pth in lnd_pths)
    return all([os.path.exists(out_path) for out_path in out_paths])

def get_mass_center(points, gray):
    im = np.zeros_like(gray)
    cv2.fillPoly(im, [points], 1)
    eyes_image = np.multiply(gray, im)
    inverse_intensity = np.divide(np.ones_like(eyes_image), eyes_image, out=np.zeros_like(eyes_image), where=eyes_image!=0)
    max = np.max(inverse_intensity)
    inverse_intensity = inverse_intensity / max
    coordinates_grid = np.indices((gray.shape[0], gray.shape[1]))
    nom = np.sum(np.multiply(coordinates_grid, np.expand_dims(inverse_intensity, axis=0)), axis=(1,2))
    denom = np.sum(inverse_intensity)
    mass_center = np.flip(nom / denom)
    return mass_center

def add_eye_pupils_landmarks(points, image):
    I = rgb2gray(image)
    left_eye_points = points[:6,:]
    right_eye_points = points[6:12,:]
    left_pupil = get_mass_center(left_eye_points, I).astype(np.int32)
    right_pupil = get_mass_center(right_eye_points, I).astype(np.int32)
    points[12, :] = left_pupil
    points[13, :] = right_pupil
    return points

def detect_landmarks(img_paths, predictor, device, mouth, template = None, multisample=True):
    landmarks = []
    prev_points = None
    align_transforms = []
    prev_aligns = None
    for i in tqdm(range(len(img_paths))):
        img = io.imread(img_paths[i])
        preds = predictor.get_landmarks_from_image(img)
        if preds is not None:
            if len(preds)>2:
                print('More than one faces were found in %s' % img_paths[i])

            # face alignment
            if template is not None:
                if multisample:
                    lnds = [preds[0]]
                    tx = [-5,+5]
                    ty = [-5,+5]
                    for x in tx:
                        for y in ty:
                            mat = np.array([[1,0,x],
                                            [0,1,y]]).astype(float)
                            img_t = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4)
                            preds_t = predictor.get_landmarks_from_image(img_t)
                            if preds_t is not None:
                                lnds.append(preds_t[0])
                    lnds = np.mean(np.stack(lnds),0)
                else:
                    lnds = preds[0]
                align = umeyama(lnds, template, True)[0:2]
                prev_aligns = align
                align_transforms.append(align)

            points = np.empty([14, 2], dtype=int)
            points[:6] = preds[0][36:42,:]   # left-eye landmarks
            points[6:12] = preds[0][42:48,:]   # right-eye landmarks
            points = add_eye_pupils_landmarks(points, img)
            if mouth:
                points = np.concatenate([points, preds[0][48:,:]], 0)    # mouth landmarks
            prev_points = points
            landmarks.append(points)
        else:
            print('No face detected, using previous landmarks')
            landmarks.append(prev_points)
            if template is not None:
                align_transforms.append(prev_aligns)
    return landmarks, align_transforms

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
    print('---------- Eye-landmarks detection --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--celeb', type=str, default='JackNicholson', help='Path to celebrity folder.')
    parser.add_argument('--mouth', action='store_true', help='Whether to save mouth landmarks too (used for training with mouth discriminator).')
    parser.add_argument('--align', action='store_true', help='Whether to calculate and save alignment transformations.')
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

    predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector='sfd')
    template = np.loadtxt('preprocessing/files/template.gz') if args.align else None    # template 68 landmarks (for face alignment)

    # Get the path of each image.
    images_dir = os.path.join(args.celeb, 'images')
    image_paths = get_image_paths(images_dir)

    if not dirs_exist(image_paths):
        landmarks, align_transforms = detect_landmarks(image_paths, predictor, device, args.mouth, template = template)
        save_results(image_paths, landmarks, align_transforms)
        print('DONE!')
    else:
        print('Eye-landmarks detection already done!')

if __name__=='__main__':
    main()
