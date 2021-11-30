import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm
from moviepy.editor import *

def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', type=str, nargs='+', default='.',
                        help="path to saved images")
    parser.add_argument('--out_path', type=str, default='.',
                        help="path to save video")
    parser.add_argument('--fps', type=float, default=30,
                        help=".")
    parser.add_argument('--audio', type=str, default=None,
                        help="Path to original .mp4 file that contains audio")
    
    args = parser.parse_args()

    for root, _, fnames in sorted(os.walk(args.imgs_path[0])):
        if len(fnames)==0:
            continue
        for name in sorted(fnames):
            im = cv2.imread(os.path.join(root, name))
            w,h = im.shape[1], im.shape[0]
            break
        break

    video = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    print('Converting images to video ...')

    for root, _, fnames in sorted(os.walk(args.imgs_path[0])):
        for name in tqdm(sorted(fnames)):
            im = cv2.imread(os.path.join(root, name))
            video.write(im)

    cv2.destroyAllWindows()
    video.release()
        
    if args.audio is not None:
        print('Adding audio with MoviePy ...')
        video = VideoFileClip(args.out_path)
        video_audio = VideoFileClip(args.audio)
        video = video.set_audio(video_audio.audio)
        os.remove(args.out_path)
        video.write_videofile(args.out_path)

    print('DONE')

if __name__ == "__main__":
    main()
