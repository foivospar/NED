import os, sys
import cv2
from skimage.transform import warp
from skimage import img_as_ubyte
import numpy as np
import argparse
from tqdm import tqdm
import torch
import pickle
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DECA.decalib.deca import DECA
from DECA.decalib.utils import util
from DECA.decalib.utils.config import cfg as deca_cfg

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_DECA_params(path, device = 'cuda'):
    pkl_files = [os.path.join(path, pkl) for pkl in sorted(os.listdir(path))]
    params = []
    for p in pkl_files:
        with open(p, 'rb') as f:
            param = pickle.load(f)
        for key in param.keys():
            if key!='tform' and key!='original_size':
                param[key] = torch.from_numpy(param[key]).to(device)
        params.append(param)

    return params, pkl_files

def read_eye_landmarks(path):
    txt_files = [os.path.join(path, txt) for txt in sorted(os.listdir(path))]
    eye_landmarks_left = []
    eye_landmarks_right = []
    for f in txt_files:
        if os.path.exists(f):
            left = np.concatenate([np.loadtxt(f)[0:6], np.loadtxt(f)[12:13]], axis=0)
            right = np.concatenate([np.loadtxt(f)[6:12], np.loadtxt(f)[13:14]], axis=0)
            eye_landmarks_left.append(left)  # Left eye
            eye_landmarks_right.append(right) # Right eye
    return [eye_landmarks_left, eye_landmarks_right]

def transform_points(points, mat):
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points

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
    print('--------- Create modified NMFCs and eye landmarks --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default='0', help='Negative value to use CPU, or greater equal than zero for GPU id.')
    parser.add_argument('--celeb', type=str, default='JackNicholson', help='Path to celebrity folder')
    parser.add_argument('--exp_name', type=str, default='happy', help='Subfolder for specific experiment')
    parser.add_argument('--no_eye_gaze', action='store_true', help='If specified, do not use eye-landmarks')
    parser.add_argument('--no_align', action='store_true', help='If specfied, no alignment is performed')
    parser.add_argument('--save_renderings', action='store_true', help='Whether to save renderings')
    parser.add_argument('--save_shapes', action='store_true', help='Whether to save shapes')
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

    # Check if conditional input files already exist.
    save_nmfcs_dir = os.path.join(args.celeb, args.exp_name, 'nmfcs')
    if os.path.isdir(save_nmfcs_dir):
        print('Conditional input files already exist!')
        exit(0)

    # Read parameters from the DECA sub-folders.
    src_codedicts, _ = read_DECA_params(os.path.join(args.celeb, 'DECA'), device=device)
    trg_codedicts, paths = read_DECA_params(os.path.join(args.celeb, args.exp_name, 'DECA'), device=device)

    # Read src eye landmarks.
    if not args.no_eye_gaze:
        src_eye_landmarks = read_eye_landmarks(os.path.join(args.celeb, 'eye_landmarks'))

    # Create save dirs
    mkdir(os.path.join(args.celeb, args.exp_name, 'nmfcs'))
    if args.save_renderings:
        mkdir(os.path.join(args.celeb, args.exp_name, 'renderings'))
    if args.save_shapes:
        mkdir(os.path.join(args.celeb, args.exp_name, 'shapes'))
    if not args.no_eye_gaze:
        mkdir(os.path.join(args.celeb, args.exp_name, 'eye_landmarks'))
    if not args.no_align:
        mkdir(os.path.join(args.celeb, args.exp_name, 'nmfcs_aligned'))
        if args.save_renderings:
            mkdir(os.path.join(args.celeb, args.exp_name, 'renderings_aligned'))
        if args.save_shapes:
            mkdir(os.path.join(args.celeb, args.exp_name, 'shapes_aligned'))
        if not args.no_eye_gaze:
            mkdir(os.path.join(args.celeb, args.exp_name, 'eye_landmarks_aligned'))

    # run DECA decoding
    deca_cfg.model.use_tex = True
    deca = DECA(config = deca_cfg, device=device)

    for i, (src_codedict, trg_codedict, pth) in tqdm(enumerate(zip(src_codedicts, trg_codedicts, paths)), total=len(src_codedicts)):
        src_codedict['exp'] = trg_codedict['exp']
        src_codedict['pose'][0,3] = trg_codedict['pose'][0,3]

        opdict, visdict = deca.decode(src_codedict)

        nmfc_pth = os.path.splitext(pth.replace('/DECA', '/nmfcs'))[0] + '.png'
        nmfc_image = warp(util.tensor2image(visdict['nmfcs'][0])/255, src_codedict['tform'], output_shape=(src_codedict['original_size'][1], src_codedict['original_size'][0]))
        nmfc_image = img_as_ubyte(nmfc_image)
        cv2.imwrite(nmfc_pth, nmfc_image)
        if not args.no_align:
            mat_pth = os.path.splitext(pth.replace(f'/{args.exp_name}/DECA', '/align_transforms'))[0] + '.txt'
            mat = np.loadtxt(mat_pth)

            nmfc_image_a = cv2.warpAffine(nmfc_image, mat, (nmfc_image.shape[1], nmfc_image.shape[0]), flags=cv2.INTER_LANCZOS4)
            cv2.imwrite(nmfc_pth.replace('/nmfcs', '/nmfcs_aligned'), nmfc_image_a)

        if args.save_renderings:
            rendering_pth = os.path.splitext(pth.replace('/DECA', '/renderings'))[0] + '.png'
            detail_image = F.grid_sample(opdict['uv_texture'], opdict['grid'].detach(), align_corners=False)
            detail_image = warp(util.tensor2image(detail_image[0])/255, src_codedict['tform'], output_shape=(src_codedict['original_size'][1], src_codedict['original_size'][0]))
            detail_image = img_as_ubyte(detail_image)
            cv2.imwrite(rendering_pth, detail_image)
            if not args.no_align:
                detail_image_a = cv2.warpAffine(detail_image, mat, (detail_image.shape[1], detail_image.shape[0]), flags=cv2.INTER_LANCZOS4)
                cv2.imwrite(rendering_pth.replace('/renderings', '/renderings_aligned'), detail_image_a)

        if args.save_shapes:
            shape_pth = os.path.splitext(pth.replace('/DECA', '/shapes'))[0] + '.png'
            shape_image = warp(util.tensor2image(visdict['shape_detail_images'][0])/255, src_codedict['tform'], output_shape=(src_codedict['original_size'][1], src_codedict['original_size'][0]))
            shape_image = img_as_ubyte(shape_image)
            cv2.imwrite(shape_pth, shape_image)
            if not args.no_align:
                shape_image_a = cv2.warpAffine(shape_image, mat, (shape_image.shape[1], shape_image.shape[0]), flags=cv2.INTER_LANCZOS4)
                cv2.imwrite(shape_pth.replace('/shapes', '/shapes_aligned'), shape_image_a)

        # Adapt eye pupil and save eye landmarks
        if not args.no_eye_gaze:
            trg_lnds = src_codedict['tform'].inverse(112 + 112*opdict['landmarks2d'][0].cpu().numpy())
            trg_left_eye = trg_lnds[36:42]
            trg_right_eye = trg_lnds[42:48]

            src_left_eye = src_eye_landmarks[0][i]
            src_right_eye = src_eye_landmarks[1][i]

            src_left_center = np.mean(src_left_eye[0:6], axis=0, keepdims=True)
            src_right_center = np.mean(src_right_eye[0:6], axis=0, keepdims=True)

            trg_left_center = np.mean(trg_left_eye[0:6], axis=0, keepdims=True)
            trg_right_center = np.mean(trg_right_eye[0:6], axis=0, keepdims=True)

            trg_left_pupil = src_left_eye[6:7] + (trg_left_center - src_left_center)
            trg_right_pupil = src_right_eye[6:7] + (trg_right_center - src_right_center)

            eye_lnds = np.concatenate([trg_left_eye, trg_right_eye, trg_left_pupil, trg_right_pupil], axis=0).astype(np.int32)
            eye_lnds_pth = os.path.splitext(pth.replace('/DECA', '/eye_landmarks'))[0] + '.txt'
            np.savetxt(eye_lnds_pth, eye_lnds)

            if not args.no_align:
                eye_lnds_a = transform_points(eye_lnds, mat)
                np.savetxt(eye_lnds_pth.replace('/eye_landmarks', '/eye_landmarks_aligned'), eye_lnds_a)
    print('DONE!')

if __name__=='__main__':
    main()
