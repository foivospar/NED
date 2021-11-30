import torch.nn as nn
from munch import Munch
from .networks import *

def create_model(opt):
    num_exp_coeffs = 51
    generator = Generator(opt.style_dim, num_exp_coeffs)
    mapping_network = MappingNetwork(opt.latent_dim, opt.hidden_dim, opt.style_dim, len(opt.selected_emotions))
    style_encoder = StyleEncoder(opt.style_dim, num_exp_coeffs)

    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        generator.cuda(opt.gpu_ids[0])
        mapping_network.cuda(opt.gpu_ids[0])
        style_encoder.cuda(opt.gpu_ids[0])

        generator = nn.DataParallel(generator, device_ids=opt.gpu_ids)
        mapping_network = nn.DataParallel(mapping_network, device_ids=opt.gpu_ids)
        style_encoder = nn.DataParallel(style_encoder, device_ids=opt.gpu_ids)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder)

    if opt.isTrain:
        discriminator = Discriminator(len(opt.selected_emotions), num_exp_coeffs)
        if len(opt.gpu_ids):
            assert(torch.cuda.is_available())
            discriminator.cuda(opt.gpu_ids[0])
            discriminator = nn.DataParallel(discriminator, device_ids=opt.gpu_ids)
        nets.discriminator = discriminator

    return nets
