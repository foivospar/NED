from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F

from DECA.decalib.deca import DECA
from DECA.decalib.utils import util
from DECA.decalib.utils.config import cfg as deca_cfg


@torch.no_grad()
def generate_mesh(exp_coeffs, opt):

	device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

	codedict_p = pickle.load(open('test_examples/Pacino/DECA/000001.pkl', 'rb'))

	deca_cfg.model.use_tex = True
	deca = DECA(config = deca_cfg, device=device)

	exp_coeffs = torch.FloatTensor(exp_coeffs).to(device)
	shape_images, shape_detail_images, tex_images, tex_detail_images = [], [], [], []
	for i in tqdm(range(exp_coeffs[:100].shape[0])):

		pose = torch.zeros(6).to(device)
		pose[3] = exp_coeffs[i,0]
		shape = torch.FloatTensor(codedict_p['shape']).to(device)
		exp = exp_coeffs[i,1:]

		codedict = {}

		codedict['detail'] = torch.FloatTensor(codedict_p['detail']).to(device)

		codedict['pose'] = pose.unsqueeze(0).float()
		codedict['exp'] = exp.unsqueeze(0).float()
		codedict['shape'] = shape
		codedict['cam'] = torch.FloatTensor([[6, 0, 0]]).to(device)

		codedict['light'] = torch.FloatTensor(codedict_p['light']).to(device)

		codedict['tex'] = torch.FloatTensor(codedict_p['tex']).to(device)

		opdict, visdict = deca.decode(codedict)

		shape_images.append(visdict['shape_images'][0].cpu())
		shape_detail_images.append(visdict['shape_detail_images'][0].cpu())
		tex_images.append(visdict['rendered_images'][0].cpu())
		tex_detail_images.append(F.grid_sample(opdict['uv_texture'], opdict['grid'].detach(), align_corners=False)[0].cpu())

	shape_images = torch.stack(shape_images, dim=0)
	shape_detail_images = torch.stack(shape_detail_images, dim=0)
	tex_images = torch.stack(tex_images, dim=0)
	tex_detail_images = torch.stack(tex_detail_images, dim=0)
	return shape_images, shape_detail_images, tex_images, tex_detail_images
