import functools
from math import exp
import os

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from torchvision import models
import torch.utils.model_zoo as model_zoo

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

def setup_device(opt):
    """Pick the best available device: CUDA > DirectML > CPU. Used by
    train/clean/train.py so training isn't CUDA-only. NOTE: DirectML
    training support in PyTorch is meaningfully less mature than DirectML
    inference or CUDA training of any kind — backward-pass/autograd op
    coverage has known gaps, especially for the kind of perceptual (VGG)
    and adversarial losses used here. This is provided as a best-effort
    option for users without any CUDA-capable GPU, not a fully-verified
    alternative to CUDA training — expect to possibly hit an unsupported
    op partway through a run, and validate with a short trial run before
    committing to a full training schedule on DirectML.
    """
    if getattr(opt, 'gpu_id', '0') == '-1':
        print("Using CPU device")
        return torch.device('cpu'), 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device, 'cuda'
    if DIRECTML_AVAILABLE:
        try:
            device = torch_directml.device()
            print(f"Using DirectML device: {device} "
                  f"(experimental for training — see setup_device() docstring)")
            return device, 'directml'
        except Exception as e:
            print(f"DirectML initialization failed: {e}, falling back to CPU")
    print("No CUDA or DirectML device available, using CPU (training will be very slow)")
    return torch.device('cpu'), 'cpu'

def resolve_inference_device(opt):
    """Decide the inference device ONCE (call right after option parsing) and
    record it on opt: opt.device_type ('cuda' | 'directml' | 'cpu'),
    opt.infer_device, opt.device_label. Prints one line the GUI reads to show
    its device badge:  [device] Inference: <TYPE> | <details>

    Previously DirectML always used its default device (index 0) and
    --gpu_id was discarded for non-CUDA systems, and only the video path's
    mosaic detector was ever moved to DirectML -- the cleaning model and the
    image path stayed on CPU. Models are now placed via to_inference_device().
    """
    requested = str(getattr(opt, 'requested_gpu_id', opt.gpu_id))
    opt.device_type, opt.infer_device = 'cpu', torch.device('cpu')
    detail = 'requested with --gpu_id -1' if requested == '-1' else 'no CUDA or DirectML device available'
    if requested != '-1':
        if opt.gpu_id != '-1' and torch.cuda.is_available():
            # CUDA_VISIBLE_DEVICES already narrowed to the chosen card(s)
            opt.device_type, opt.infer_device = 'cuda', torch.device('cuda')
            detail = f"{torch.cuda.get_device_name(0)} (gpu_id {requested})"
        elif DIRECTML_AVAILABLE:
            try:
                count = torch_directml.device_count()
                idx = int(requested) if requested.isdigit() else 0
                if idx >= count:
                    print(f"[device] gpu_id {requested} not found ({count} DirectML device(s)); using device 0")
                    idx = 0
                opt.device_type, opt.infer_device = 'directml', torch_directml.device(idx)
                detail = f"{torch_directml.device_name(idx)} (device {idx} of {count})"
            except Exception as e:
                detail = f"DirectML failed to initialise ({e})"
    if opt.device_type == 'cpu' and requested != '-1':
        detail += _cpu_fallback_reason()
    opt.device_label = {'cuda': 'CUDA', 'directml': 'DirectML', 'cpu': 'CPU'}[opt.device_type]
    print(f"[device] Inference: {opt.device_label} | {detail}", flush=True)
    return opt.infer_device

def _cpu_fallback_reason():
    """Explain an unrequested CPU fallback. The usual cause is the Python
    environment, not the hardware: the PyTorch build installed in the Python
    running this (e.g. system Python vs a venv) may be CPU-only. Naming the
    interpreter and build makes that visible in the log."""
    import sys
    build = torch.__version__
    if torch.version.cuda is None:
        why = "this PyTorch is a CPU-only build"
    else:
        why = f"this PyTorch supports CUDA {torch.version.cuda} but sees no CUDA GPU (check the NVIDIA driver)"
    return f" -- {why}; Python: {sys.executable} (torch {build})"

def to_inference_device(net, opt):
    """Move a freshly loaded model to the DirectML device if that's the
    resolved inference device. CUDA/CPU placement is already handled by
    todevice(); no-op if resolve_inference_device() was never called
    (e.g. dataset-building scripts)."""
    if getattr(opt, 'device_type', None) != 'directml':
        return net
    if hasattr(net, 'encoder3d'):
        # video model: DirectML can't run its Conv3d layers -> exact 2-D form
        from .BVDNet import make_directml_compatible
        net = make_directml_compatible(net)
    try:
        return net.to(opt.infer_device)
    except Exception as e:
        print(f"[device] Could not move {type(net).__name__} to DirectML ({e}); running it on CPU")
        return net.cpu()

################################## IO ##################################
def save(net,path,gpu_id,device=None):
    if isinstance(net, nn.DataParallel):
        torch.save(net.module.cpu().state_dict(),path)
    else:
        torch.save(net.cpu().state_dict(),path) 
    if device is not None:
        net.to(device)
    elif gpu_id != '-1':
        net.cuda()

def todevice(net,gpu_id,device=None):
    if device is not None:
        net = net.to(device)
        return net
    if gpu_id != '-1' and len(gpu_id) == 1:
        net.cuda()
    elif gpu_id != '-1' and len(gpu_id) > 1:
        net = nn.DataParallel(net)
        net.cuda()
    return net

# patch InstanceNorm checkpoints prior to 0.4
def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
           (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

################################## initialization ##################################
def get_norm_layer(norm_type='instance',mod = '2d'):
    if norm_type == 'batch':
        if mod == '2d':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif mod == '3d':
            norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        if mod == '2d':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        elif mod =='3d':
            norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)

################################## Network structure ##################################
################################## ResnetBlock ##################################
class ResnetBlockSpectralNorm(nn.Module):
    def __init__(self, dim, padding_type, activation=nn.LeakyReLU(0.2), use_dropout=False):
        super(ResnetBlockSpectralNorm, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=p)),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=p))]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

################################## Resnet ##################################
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

################################## Loss function ##################################
class HingeLossD(nn.Module):
    def __init__(self):
        super(HingeLossD, self).__init__()

    def forward(self, dis_fake, dis_real):
        loss_real = torch.mean(F.relu(1. - dis_real))
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real + loss_fake

class HingeLossG(nn.Module):
    def __init__(self):
        super(HingeLossG, self).__init__()

    def forward(self, dis_fake):
        loss_fake = -torch.mean(dis_fake)
        return loss_fake

class VGGLoss(nn.Module):
    def __init__(self, gpu_id, device=None):
        super(VGGLoss, self).__init__()  

        self.vgg = Vgg19()
        if device is not None:
            self.vgg = self.vgg.to(device)
        elif gpu_id != '-1' and len(gpu_id) == 1:
            self.vgg.cuda()
        elif gpu_id != '-1' and len(gpu_id) > 1:
            self.vgg = nn.DataParallel(self.vgg)
            self.vgg.cuda()

        self.criterion = nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

################################## Evaluation ##################################
'''https://github.com/Po-Hsun-Su/pytorch-ssim

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_ssim.ssim(img1, img2))

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print(ssim_loss(img1, img2))
'''

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
