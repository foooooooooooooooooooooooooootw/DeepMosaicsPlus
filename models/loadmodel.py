import torch
import os
import json
from . import model_util
from .pix2pix_model import define_G as pix2pix_G
from .pix2pixHD_model import define_G as pix2pixHD_G
# from .video_model import MosaicNet
# from .videoHD_model import MosaicNet as MosaicNet_HD
from .BiSeNet_model import BiSeNet
from .BVDNet import define_G as video_G

def show_paramsnumber(net,netname='net'):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    print(netname+' parameters: '+str(parameters)+'M')

def pix2pix(opt):
    # print(opt.model_path,opt.netG)
    if opt.netG == 'HD':
        netG = pix2pixHD_G(3, 3, 64, 'global' ,4)
    else:
        netG = pix2pix_G(3, 3, 64, opt.netG, norm='batch',use_dropout=True, init_type='normal', gpu_ids=[])
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path))
    netG = model_util.todevice(netG,opt.gpu_id)
    netG = model_util.to_inference_device(netG, opt)
    netG.eval()
    return netG


def style(opt):
    if opt.edges:
        netG = pix2pix_G(1, 3, 64, 'resnet_9blocks', norm='instance',use_dropout=True, init_type='normal', gpu_ids=[])
    else:
        netG = pix2pix_G(3, 3, 64, 'resnet_9blocks', norm='instance',use_dropout=False, init_type='normal', gpu_ids=[])

    #in other to load old pretrain model
    #https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/models/base_model.py
    if isinstance(netG, torch.nn.DataParallel):
        netG = netG.module
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state_dict = torch.load(opt.model_path, map_location='cpu')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        model_util.patch_instance_norm_state_dict(state_dict, netG, key.split('.'))
    netG.load_state_dict(state_dict)

    netG = model_util.todevice(netG,opt.gpu_id)
    netG = model_util.to_inference_device(netG, opt)
    netG.eval()
    return netG

def video(opt):
    netG = video_G(N=2,n_blocks=4,gpu_id=opt.gpu_id)
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path))
    netG = model_util.todevice(netG,opt.gpu_id)
    netG = model_util.to_inference_device(netG, opt)
    netG.eval()
    return netG

def bisenet(opt,type='roi'):
    '''
    type: roi or mosaic
    '''
    net = BiSeNet(num_classes=1, context_path='resnet18',train_flag=False)
    show_paramsnumber(net,'segment')
    if type == 'roi':
        net.load_state_dict(torch.load(opt.model_path))
    elif type == 'mosaic':
        net.load_state_dict(torch.load(opt.mosaic_position_model_path))
    net = model_util.todevice(net,opt.gpu_id)
    net = model_util.to_inference_device(net, opt)
    net.eval()
    return net


# ── Size-based model auto-selection (--auto_model) ──────────────────────────
# Maps a detected mosaic block size (see util.mosaic.estimate_mosaic_block_pct,
# expressed as % of the frame's shorter side) to a specific pretrained clean
# model checkpoint. Manifest format is a flat JSON object of
# {"<bucket_pct>": "<path to .pth, relative to the manifest file's folder>"}.
# Selection is nearest-bucket, clamped to the manifest's own min/max buckets.

def load_size_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        raw = json.load(f)
    if not raw:
        raise ValueError(f"Model manifest '{manifest_path}' is empty")
    # keys are stored as strings in JSON; normalise to float for comparison
    return {float(k): v for k, v in raw.items()}

def select_model_by_pct(pct, manifest_path):
    """Return (absolute_model_path, matched_bucket_pct) for the manifest
    bucket nearest to `pct`, clamped to the manifest's own range so a
    detected size outside [min_bucket, max_bucket] still resolves to the
    closest trained model rather than erroring out.
    """
    manifest = load_size_manifest(manifest_path)
    buckets = sorted(manifest.keys())
    nearest = min(buckets, key=lambda b: abs(b - pct))
    rel_path = manifest[nearest]
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    abs_path = rel_path if os.path.isabs(rel_path) else os.path.normpath(os.path.join(base_dir, rel_path))
    return abs_path, nearest
