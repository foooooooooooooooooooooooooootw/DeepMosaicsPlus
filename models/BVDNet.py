import torch
import torch.nn as nn
from .pix2pixHD_model import *
from .model_util import *
from models import model_util

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super().__init__()

        self.convup = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ReflectionPad2d(padding),
                # EqualConv2d(out_channel, out_channel, kernel_size, padding=padding),
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size)),
                nn.LeakyReLU(0.2),
                # Blur(out_channel),
            )

    def forward(self, input):
        outup = self.convup(input)
        return outup

class Encoder2d(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, activation = nn.LeakyReLU(0.2)):
        super(Encoder2d, self).__init__()        
   
        model = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [  nn.ReflectionPad2d(1),
                        SpectralNorm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0)), 
                        activation]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Encoder3d(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, activation = nn.LeakyReLU(0.2)):
        super(Encoder3d, self).__init__()        
               
        model = [SpectralNorm(nn.Conv3d(input_nc, ngf, kernel_size=3, padding=1)), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [  SpectralNorm(nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)),
                         activation]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class BVDNet(nn.Module):
    def __init__(self, N=2, n_downsampling=3, n_blocks=4, input_nc=3, output_nc=3,activation=nn.LeakyReLU(0.2)):
        super(BVDNet, self).__init__()
        ngf = 64
        padding_type = 'reflect'
        self.N = N

        ### encoder
        self.encoder3d = Encoder3d(input_nc,64,n_downsampling,activation)
        self.encoder2d = Encoder2d(input_nc,64,n_downsampling,activation)

        ### resnet blocks
        self.blocks = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            self.blocks += [ResnetBlockSpectralNorm(ngf * mult, padding_type=padding_type, activation=activation)]
        self.blocks = nn.Sequential(*self.blocks)

        ### decoder
        self.decoder = []        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.decoder += [UpBlock(ngf * mult, int(ngf * mult / 2))]
        self.decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]        
        self.decoder = nn.Sequential(*self.decoder)
        self.limiter = nn.Tanh()

    def forward(self, stream, previous):
        if getattr(self, '_frames_mode', False):
            return self._forward_frames(stream, previous)
        this_shortcut = stream[:,:,self.N]
        stream = self.encoder3d(stream)
        stream = stream.reshape(stream.size(0),stream.size(1),stream.size(3),stream.size(4))
        previous = self.encoder2d(previous)
        x = stream + previous
        x = self.blocks(x)
        x = self.decoder(x)
        x = x+this_shortcut
        x = self.limiter(x)
        return x

def define_G(N=2, n_blocks=1, gpu_id='-1', device=None):
    netG = BVDNet(N = N, n_blocks=n_blocks)
    netG = model_util.todevice(netG,gpu_id,device=device)
    netG.apply(model_util.init_weights)
    return netG

################################Discriminator################################
def define_D(input_nc=6, ndf=64, n_layers_D=1, use_sigmoid=False, num_D=3, gpu_id='-1', device=None):          
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid, num_D)
    netD = model_util.todevice(netD,gpu_id,device=device)
    netD.apply(model_util.init_weights)
    return netD

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, use_sigmoid)
            setattr(self, 'layer'+str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        return self.model(input)        

class GANLoss(nn.Module):
    def __init__(self, mode='D'):
        super(GANLoss, self).__init__()
        if mode == 'D':
            self.lossf = model_util.HingeLossD()
        elif mode == 'G':
            self.lossf = model_util.HingeLossG()
        self.mode = mode
    
    def forward(self, dis_fake = None, dis_real = None):
        if isinstance(dis_fake, list):
            if self.mode == 'D':
                loss = 0
                for i in range(len(dis_fake)):
                    loss += self.lossf(dis_fake[i][-1],dis_real[i][-1])
            elif self.mode =='G':
                loss = 0
                weight = 2**len(dis_fake)
                for i in range(len(dis_fake)):
                    weight = weight/2
                    loss += weight*self.lossf(dis_fake[i][-1])
            return loss
        else:
            if self.mode == 'D':
                return self.lossf(dis_fake[-1],dis_real[-1])
            elif self.mode =='G':
                return self.lossf(dis_fake[-1])


################################ DirectML support ################################
# torch-directml's convolution only accepts 4-D input, so Encoder3d's Conv3d
# layers fail ("input must be 4-dimensional"). A 3-D convolution is exactly a
# sum of 2-D convolutions -- one per temporal slice of its kernel, applied to
# the matching input frame -- so the encoder can be recomputed on a LIST of
# ordinary 4-D frames with identical results, and DirectML never sees a 5-D
# tensor. Used only for inference (weights are frozen at conversion time).

def _effective_weight(conv):
    """The weight a forward pass in eval mode actually uses: spectral norm
    (weight_orig / sigma, computed from the stored u/v vectors without a
    power-iteration update) if present, else the plain weight."""
    for hook in conv._forward_pre_hooks.values():
        if type(hook).__name__ == 'SpectralNorm':
            with torch.no_grad():
                return hook.compute_weight(conv, do_power_iteration=False).detach().clone()
    return conv.weight.detach().clone()

class _Conv3dAs2d(nn.Module):
    """Exact Conv3d on a list of T frames (each B,C,H,W) using only conv2d."""
    def __init__(self, conv):
        super().__init__()
        assert conv.groups == 1 and tuple(conv.dilation) == (1, 1, 1) and conv.padding_mode == 'zeros'
        w = _effective_weight(conv)                       # (out, in, kT, kH, kW)
        self.kt, self.st, self.pt = w.shape[2], conv.stride[0], conv.padding[0]
        self.stride2d, self.padding2d = tuple(conv.stride[1:]), tuple(conv.padding[1:])
        self.w = nn.ParameterList([nn.Parameter(w[:, :, k].contiguous(), requires_grad=False)
                                   for k in range(self.kt)])
        self.b = (nn.Parameter(conv.bias.detach().clone().view(1, -1, 1, 1), requires_grad=False)
                  if conv.bias is not None else None)

    def forward(self, frames):
        T = len(frames)
        out = []
        for t in range((T + 2 * self.pt - self.kt) // self.st + 1):
            acc = None
            for k in range(self.kt):
                ti = t * self.st - self.pt + k                 # zero temporal padding = skip
                if 0 <= ti < T:
                    y = nn.functional.conv2d(frames[ti], self.w[k], None, self.stride2d, self.padding2d)
                    acc = y if acc is None else acc + y
            out.append(acc + self.b if self.b is not None else acc)
        return out

class _Encoder3dAs2d(nn.Module):
    def __init__(self, enc3d):
        super().__init__()
        self.layers = nn.ModuleList(_Conv3dAs2d(m) if isinstance(m, nn.Conv3d) else m
                                    for m in enc3d.model)

    def forward(self, frames):
        for m in self.layers:
            frames = m(frames) if isinstance(m, _Conv3dAs2d) else [m(f) for f in frames]
        return frames

def _forward_frames(self, stream, previous):
    """BVDNet.forward for converted models. The 5-D stream is split into
    frames on the CPU and only 4-D frames are moved to the model's device."""
    dev = next(self.parameters()).device
    # .contiguous(): each frame sliced out of the stack is a strided view, and
    # copying strided tensors to DirectML has been unreliable
    frames = [f.contiguous().to(dev) for f in stream.cpu().unbind(2)]
    this_shortcut = frames[self.N]
    feats = self.encoder3d(frames)
    assert len(feats) == 1, f"expected the encoder to reduce time to 1, got {len(feats)}"
    x = feats[0] + self.encoder2d(previous.contiguous().to(dev))
    x = self.blocks(x)
    x = self.decoder(x)
    return self.limiter(x + this_shortcut)
BVDNet._forward_frames = _forward_frames

def _bilinear_up_matrix(n):
    """(2n x n) interpolation matrix reproducing PyTorch's bilinear x2 upsample
    with align_corners=False, including its edge clamping."""
    U = torch.zeros(2 * n, n)
    for d in range(2 * n):
        src = max((d + 0.5) / 2 - 0.5, 0.0)
        i0 = min(int(src), n - 1)          # src >= 0, so int() == floor()
        i1 = min(i0 + 1, n - 1)
        w = src - i0
        U[d, i0] += 1 - w
        U[d, i1] += w
    return U

class _BilinearUp2x(nn.Module):
    """nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    computed as two matrix multiplications (rows, then columns). Bilinear
    upsampling is linear and separable, so this is exact (verified against
    F.interpolate to ~5e-7). torch-directml's own bilinear upsample returned
    wrong values (tools/check_directml.py: relative error ~1.2 at the first
    decoder Upsample), which showed up as heavy noise in cleaned output."""
    def __init__(self):
        super().__init__()
        self._cache = {}

    def _mat(self, n, x):
        key = (n, str(x.device), x.dtype)
        if key not in self._cache:
            self._cache[key] = _bilinear_up_matrix(n).to(dtype=x.dtype).to(x.device)
        return self._cache[key]

    def forward(self, x):
        uh = self._mat(x.shape[-2], x)
        uw = self._mat(x.shape[-1], x)
        return torch.matmul(torch.matmul(uh, x), uw.t())

def _is_bilinear_up2x(m):
    if not isinstance(m, nn.Upsample) or m.mode != 'bilinear' or m.align_corners:
        return False
    sf = m.scale_factor
    return (sf if isinstance(sf, tuple) else (sf, sf)) in ((2, 2), (2.0, 2.0))

def make_directml_compatible(net):
    """Convert a BVDNet in place for DirectML. Every replacement is exact:
      * Encoder3d's Conv3d -> sums of 2-D convolutions (DirectML convolution
        only accepts 4-D input)
      * bilinear Upsample -> matrix multiplications (DirectML computed it
        incorrectly)
      * spectral norm baked into plain weights. In eval mode it recomputes
        the same fixed weights on every forward pass, and part of that
        (aten::addmv) isn't supported on DirectML, so it bounced through the
        CPU for every layer of every frame."""
    if getattr(net, '_frames_mode', False):
        return net
    net.eval()
    net.encoder3d = _Encoder3dAs2d(net.encoder3d)      # reads spectral-norm weights itself
    for m in net.modules():
        for hook in list(m._forward_pre_hooks.values()):
            if type(hook).__name__ == 'SpectralNorm':
                torch.nn.utils.remove_spectral_norm(m, name=hook.name)   # uses do_power_iteration=False
                break
    for parent in list(net.modules()):
        for name, child in list(parent.named_children()):
            if _is_bilinear_up2x(child):
                setattr(parent, name, _BilinearUp2x())
    net._frames_mode = True
    return net
