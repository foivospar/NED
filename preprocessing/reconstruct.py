import os, sys
import cv2
from skimage.transform import warp
from skimage import img_as_ubyte
import numpy as np
from time import time
import argparse
from tqdm import tqdm
import torch
import pickle
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DECA.decalib.deca import DECA
from DECA.decalib.datasets import datasets
from DECA.decalib.utils import util
from DECA.decalib.utils.config import cfg as deca_cfg

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dirs_exist(args):
    out_pths = [os.path.join(args.celeb, 'DECA')]
    if args.save_renderings:
        out_pths.append(os.path.join(args.celeb, 'renderings'))
    if args.save_shapes:
        out_pths.append(os.path.join(args.celeb, 'shapes'))
    if args.save_nmfcs:
        out_pths.append(os.path.join(args.celeb, 'nmfcs'))
    return all([os.path.exists(out_path) for out_path in out_pths])

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
    print('---------- 3D face reconstruction (DECA) --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, default='JackNicholson', help='Path to celebrity folder.')
    parser.add_argument('--gpu_id', type=int, default='0', help='Negative value to use CPU, or greater equal than zero for GPU id.')
    parser.add_argument('--save_renderings', action='store_true', help='Save the rendering images.')
    parser.add_argument('--save_shapes', action='store_true', help='Save the detailed shape produced by DECA.')
    parser.add_argument('--save_nmfcs', action='store_true', help='Save the nmfc produced by DECA.')
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

    # load test images
    images_folder = os.path.join(args.celeb, 'images')
    dataset = datasets.TestData(images_folder, iscrop=True, face_detector='fan', scale=1.25, device=device)
    template = np.loadtxt('preprocessing/files/template.gz')

    if not dirs_exist(args):

        # run DECA
        deca_cfg.model.use_tex = True
        deca = DECA(config = deca_cfg, device=device)

        print('Reconstructing faces and saving results')
        for i in tqdm(range(len(dataset))):
            img_pth = dataset[i]['imagepath']
            images = dataset[i]['image'].to(device)[None,...]
            with torch.no_grad():
                codedict = deca.encode(images)
                mkdir(os.path.dirname(img_pth.replace('/images', '/DECA')))
                new_codedict = {}
                for key in codedict:
                    if key!='images':
                        new_codedict[key] = codedict[key].cpu().numpy()
                new_codedict['tform'] = dataset[i]['tform']
                new_codedict['original_size'] = dataset[i]['original_size']
                codedict_pth = os.path.splitext(img_pth.replace('/images', '/DECA'))[0] + '.pkl'
                with open(codedict_pth, "wb") as f:
                    pickle.dump(new_codedict, f)

                if args.save_renderings or args.save_shapes or args.save_nmfcs:
                    opdict, visdict = deca.decode(codedict)

            if args.save_renderings :
                mkdir(os.path.dirname(img_pth.replace('/images', '/renderings')))
                rendering_pth = img_pth.replace('/images', '/renderings')
                detail_image = F.grid_sample(opdict['uv_texture'], opdict['grid'].detach(), align_corners=False)
                detail_image = warp(util.tensor2image(detail_image[0])/255, dataset[i]['tform'], output_shape=(dataset[i]['original_size'][1], dataset[i]['original_size'][0]))
                cv2.imwrite(rendering_pth, (detail_image*255).astype(int))

            if args.save_shapes:
                mkdir(os.path.dirname(img_pth.replace('/images', '/shapes')))
                shape_pth = img_pth.replace('/images', '/shapes')
                shape_image = warp(util.tensor2image(visdict['shape_detail_images'][0])/255, dataset[i]['tform'], output_shape=(dataset[i]['original_size'][1], dataset[i]['original_size'][0]))
                cv2.imwrite(shape_pth, (shape_image*255).astype(int))

            if args.save_nmfcs:
                mkdir(os.path.dirname(img_pth.replace('/images', '/nmfcs')))
                nmfc_pth = img_pth.replace('/images', '/nmfcs')
                nmfc_image = warp(util.tensor2image(visdict['nmfcs'][0])/255, dataset[i]['tform'], output_shape=(dataset[i]['original_size'][1], dataset[i]['original_size'][0]))
                cv2.imwrite(nmfc_pth, (nmfc_image*255).astype(int))
        print('DONE!')
    else:
        print('3D face reconstruction already done!')

if __name__=='__main__':
    main()
