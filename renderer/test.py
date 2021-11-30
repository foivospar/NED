import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from renderer.options.test_options import TestOptions
from renderer.data.custom_dataset_data_loader import CreateDataLoader
from renderer.models.head2head_model import create_model
import renderer.util.util as util

opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batch_size = 1
opt.serial_batches = True


modelG = create_model(opt)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

print('Generating %d frames' % dataset_size)

postfix = '_aligned' if not opt.no_align else ''
self = 'SELF' if not opt.self_name else opt.self_name
save_dir = os.path.join(opt.celeb, opt.exp_name if opt.exp_name else self, 'faces' + postfix)
util.mkdir(save_dir)

for i, data in enumerate(dataset):
    if opt.time_fwd_pass:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    if data['change_seq']:
        modelG.fake_B_prev = None

    _, _, height, width = data['nmfc_video'].size()
    nmfc_video = Variable(data['nmfc_video']).view(1, -1, 3, height, width)
    input_A = nmfc_video
    rgb_video = Variable(data['rgb_video']).view(1, -1, 3, height, width)
    if not opt.no_eye_gaze:
        eye_gaze_video = Variable(data['eye_video']).view(1, -1, 3, height, width)
        input_A = torch.cat([nmfc_video, eye_gaze_video], dim=2)
    img_path = data['A_paths']

    if opt.use_shapes:
        shape_video = Variable(data['shape_video']).view(1, -1, 3, height, width)
        input_A = torch.cat([input_A, shape_video], dim=2)

    mask_video = Variable(data['mask_video']).view(1, -1, 3, height, width)
    mask_video = mask_video[:,:,0,:,:].unsqueeze(2)
    input_A = torch.cat([input_A, mask_video], dim=2)

    print('Processing NMFC image %s' % img_path[-1])

    generated = modelG.inference(input_A, rgb_video)

    if opt.time_fwd_pass:
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print('Forward pass time: %.6f' % start.elapsed_time(end))

    fake_frame = util.tensor2im(generated.data[0])
    util.save_image(fake_frame, os.path.join(save_dir, os.path.basename(img_path[-1][0])))
