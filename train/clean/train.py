import os
import sys

# Windows consoles often default to a non-UTF-8 code page (cp1252/cp437),
# which mangles Unicode characters like em-dashes into '?' or a replacement
# character. Force UTF-8 stdout so any character in this script's output
# prints correctly regardless of the system's console code page.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
# Anchor these to the script's own location rather than the current
# working directory -- sys.path.append('..') only resolves correctly if
# the caller happened to cd into train/clean/ first, which breaks this
# script when run from the project root (python train/clean/train.py ...)
# or launched as a subprocess from anywhere else, both of which are
# common (e.g. tools/train_all_buckets.py does exactly that).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_here, '..'))
sys.path.append(os.path.join(_here, '..', '..'))
# A crash inside native code (GPU driver, ROCm/CUDA, PyTorch) normally kills the
# process without any Python error. faulthandler prints where Python was at that
# moment, so such a crash is no longer silent.
import faulthandler
try:
    faulthandler.enable()
except Exception:
    pass
if __name__ == '__main__':
    # printed before the heavy imports, so even a failure while loading PyTorch
    # leaves a trace of how far it got
    print(f"[train] Starting (Python {sys.version.split()[0]}). Loading PyTorch...", flush=True)
from cores import Options
opt = Options()

import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import time

from util import util,data,dataloader
from util import image_processing as impro
from models import BVDNet,model_util
from skimage.metrics import structural_similarity
from tensorboardX import SummaryWriter
from tqdm import tqdm
if __name__ == '__main__':
    print(f"[train] PyTorch {torch.__version__} loaded.", flush=True)

'''
--------------------------Get options--------------------------
'''
opt.parser.add_argument('--N',type=int,default=2, help='The input tensor shape is H×W×T×C, T = 2N+1')
opt.parser.add_argument('--S',type=int,default=3, help='Stride of 3 frames')
# opt.parser.add_argument('--T',type=int,default=7, help='T = 2N+1')
opt.parser.add_argument('--M',type=int,default=100, help='How many frames read from each videos')
opt.parser.add_argument('--lr',type=float,default=0.0002, help='')
opt.parser.add_argument('--beta1',type=float,default=0.9, help='')
opt.parser.add_argument('--beta2',type=float,default=0.999, help='')
opt.parser.add_argument('--finesize',type=int,default=512, help='size (px) of the square crops the network trains on, and the size it runs at later (default 512; was 256). Larger = finer detail but memory grows with its SQUARE (512 uses ~4x the memory per sample of 256 -- lower --batchsize accordingly, or pass --finesize 256 on a small GPU). The dataset must have been built with a matching --outsize (see make_datasets/make_video_dataset.py and tools/bucket_real_pairs.py)')
opt.parser.add_argument('--loadsize',type=int,default=0, help='size frames are resized to before the random crop. 0 = automatic: finesize x 1.117 (572 for the default 512, 286 for 256)')
opt.parser.add_argument('--batchsize',type=int,default=1, help='')
opt.parser.add_argument('--no_gan', action='store_true', help='if specified, do not use gan')
opt.parser.add_argument('--n_blocks',type=int,default=4, help='')
opt.parser.add_argument('--n_layers_D',type=int,default=2, help='')
opt.parser.add_argument('--num_D',type=int,default=3, help='')
opt.parser.add_argument('--lambda_L2',type=float,default=100, help='')
opt.parser.add_argument('--lambda_VGG',type=float,default=1, help='')
opt.parser.add_argument('--lambda_GAN',type=float,default=0.01, help='')
opt.parser.add_argument('--lambda_D',type=float,default=1, help='')
opt.parser.add_argument('--load_thread',type=int,default=6, help='number of worker PROCESSES preparing training batches (default 6; was 16). Each one is a full Python process with its own copy of PyTorch, so on Windows every extra one adds start-up time and RAM; the evaluation loader uses a quarter of this. Raise it only if the log shows training waiting for data')

opt.parser.add_argument('--dataset',type=str,default='./datasets/face/', help='')
opt.parser.add_argument('--dataset_test',type=str,default='./datasets/face_test/', help='')
opt.parser.add_argument('--dataset_mosaic_pct', type=float, default=None,
    help='EXPERIMENTAL: train a size-specialized model by constraining synthetically-generated '
         'mosaic blocks to a narrow band around this percentage of each training image\'s shorter '
         'side (matching the --auto_model detector\'s own convention), instead of the default wide '
         'random range. Only affects video folders using synthetic mosaic generation (origin_image/ '
         '+ mask/) — folders with real paired data (origin_image/ + mosaic_image/) are unaffected, '
         'since those already use one fixed real size. Leave unset for the original one-size-fits-all '
         'training behavior.')
opt.parser.add_argument('--n_epoch',type=int,default=200, help='')
opt.parser.add_argument('--save_freq',type=int,default=10000, help='')
opt.parser.add_argument('--continue_train', action='store_true', help='resume from the newest checkpoint in checkpoints/<savename>: loads the generator (and discriminator), the optimizer state if it was saved, and continues counting from that iteration (TensorBoard curves continue too). Starts from scratch, with a message, if there is no checkpoint or it was trained at another --finesize')
opt.parser.add_argument('--savename',type=str,default='face', help='')
opt.parser.add_argument('--showresult_freq',type=int,default=1000, help='')
opt.parser.add_argument('--showresult_num',type=int,default=4, help='')

def ImageQualityEvaluation(tensor1,tensor2,showiter,writer,tag):
    batch_len = len(tensor1)
    psnr,ssmi = 0,0
    for i in range(len(tensor1)):
        img1,img2 = data.tensor2im(tensor1,rgb2bgr=False,batch_index=i), data.tensor2im(tensor2,rgb2bgr=False,batch_index=i)
        psnr += impro.psnr(img1,img2)
        # channel_axis is the modern scikit-image API (multichannel= was
        # deprecated then removed). Without it correctly set, some skimage
        # versions try to fit the similarity window across the color-channel
        # axis too (size 3), which is smaller than the default 7px window --
        # producing a "win_size exceeds image extent" error that looks like
        # an image-size problem but isn't one. Falling back to multichannel=
        # keeps this working on older skimage installs that predate
        # channel_axis.
        try:
            ssmi += structural_similarity(img1,img2,channel_axis=-1)
        except TypeError:
            ssmi += structural_similarity(img1,img2,multichannel=True)
    writer.add_scalars('quality/psnr', {tag:psnr/batch_len}, showiter)
    writer.add_scalars('quality/ssmi', {tag:ssmi/batch_len}, showiter)
    return psnr/batch_len,ssmi/batch_len

def ShowImage(tensor1,tensor2,tensor3,showiter,max_num,writer,tag):
    show_imgs = []
    for i in range(max_num):
        show_imgs += [  data.tensor2im(tensor1,rgb2bgr = False,batch_index=i),
                        data.tensor2im(tensor2,rgb2bgr = False,batch_index=i),
                        data.tensor2im(tensor3,rgb2bgr = False,batch_index=i)]
    show_img = impro.splice(show_imgs,  (opt.showresult_num,3))
    writer.add_image(tag, show_img,showiter,dataformats='HWC')

'''
--------------------------Init--------------------------
'''
if __name__ == '__main__':
    # Windows' multiprocessing uses 'spawn' (not 'fork' like Linux/Mac), which
    # re-imports this entire script in every worker process. Without this guard,
    # each worker would re-run everything below -- including spawning its OWN
    # workers -- recursively, crashing with a RuntimeError about starting a
    # process before bootstrapping finishes. dataloader.VideoDataLoader below
    # uses multiprocessing.Process internally, which is what actually triggers
    # this if the guard is missing.
    opt = opt.getparse()
    opt.T = 2*opt.N+1
    if opt.loadsize <= 0:
        opt.loadsize = int(round(opt.finesize * 286 / 256))   # keeps the original 256 -> 286 ratio
    if opt.loadsize < opt.finesize:
        sys.exit(f"--loadsize ({opt.loadsize}) must be at least --finesize ({opt.finesize})")
    if opt.finesize != 256:
        print(f"Training at {opt.finesize}px (loadsize {opt.loadsize}). Expect about {(opt.finesize/256)**2:.1f}x the memory per sample of the default 256px.")
    try:
        _free, _total = torch.cuda.mem_get_info()
        _per_sample = 3.6 * (opt.finesize / 512.0) ** 2          # GB per sample, measured on a 2080 Ti at 512px
        _need = 1.0 + opt.batchsize * _per_sample
        _total_gb = _total / 1024.0 ** 3
        if _need > 0.9 * _total_gb:
            _safe = max(1, int((0.85 * _total_gb - 1.0) / _per_sample))
            print(f"** WARNING: batch {opt.batchsize} at {opt.finesize}px needs about {_need:.0f} GB of GPU memory but the card has "
                  f"{_total_gb:.0f} GB. Expect an out-of-memory error, or a huge slowdown if Windows spills into system RAM. "
                  f"Use --batchsize {_safe} or lower. **", flush=True)
    except Exception:
        pass
    if opt.showresult_num >opt.batchsize:
        opt.showresult_num = opt.batchsize
    dir_checkpoint = os.path.join('checkpoints',opt.savename)
    util.makedirs(dir_checkpoint)
    try:                                      # a run that is started again is not finished until it says so
        os.remove(os.path.join(dir_checkpoint, 'completed.json'))
    except OSError:
        pass
    # start tensorboard
    localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    tensorboard_savedir = os.path.join('checkpoints/tensorboard',localtime+'_'+opt.savename)
    TBGlobalWriter = SummaryWriter(tensorboard_savedir)
    print('Please run "tensorboard --logdir checkpoints/tensorboardX --host=your_server_ip" and input "'+localtime+'" to filter outputs')

    '''
    --------------------------Init Network--------------------------
    '''
    if opt.gpu_id != '-1' and len(opt.gpu_id) == 1:
        # On NVIDIA this picks the fastest convolution algorithm once, cheaply. On AMD
        # ROCm the same switch makes MIOpen benchmark every convolution exhaustively
        # on first use, which can stall the first iterations for many minutes.
        torch.backends.cudnn.benchmark = not model_util.is_rocm()

    def save_generator(path):
        model_util.save(netG, path, opt.gpu_id, device=_device_for_calls)
        # The network is fully convolutional: the .pth doesn't say what size it was trained at,
        # and running it at another size gives quietly worse results. Record it beside the file;
        # inference reads it (copy this .json along with the .pth).
        import json
        with open(os.path.splitext(path)[0] + '.json', 'w') as f:
            json.dump({'finesize': opt.finesize, 'loadsize': opt.loadsize, 'N': opt.N, 'S': opt.S,
                       'n_blocks': opt.n_blocks, 'model': 'BVDNet'}, f, indent=2)

    def save_resume_state(iteration):
        # Optimizer state + iteration, so --continue_train resumes exactly (one file, overwritten each save).
        state = {'iteration': iteration, 'finesize': opt.finesize, 'optimizer_G': optimizer_G.state_dict()}
        if not opt.no_gan:
            state['optimizer_D'] = optimizer_D.state_dict()
        path = os.path.join('checkpoints', opt.savename, 'resume_state.pth')
        torch.save(state, path + '.part')
        os.replace(path + '.part', path)

    # device is None for CUDA (existing gpu_id-string based path, untouched) or
    # CPU; only set for DirectML, so this is purely additive — CUDA training
    # behaves exactly as before. See model_util.setup_device() for the caveat
    # that DirectML training is experimental/best-effort, unlike CUDA.
    _torch_device, _device_type = model_util.setup_device(opt)
    _device_for_calls = _torch_device if _device_type == 'directml' else None

    netG = BVDNet.define_G(opt.N,opt.n_blocks,gpu_id=opt.gpu_id,device=_device_for_calls)
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    lossfun_L2 = nn.MSELoss()
    lossfun_VGG = model_util.VGGLoss(opt.gpu_id,device=_device_for_calls)
    if not opt.no_gan:
        netD = BVDNet.define_D(n_layers_D=opt.n_layers_D,num_D=opt.num_D,gpu_id=opt.gpu_id,device=_device_for_calls)
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        lossfun_GAND = BVDNet.GANLoss('D')
        lossfun_GANG = BVDNet.GANLoss('G')

    start_iter = 0
    if opt.continue_train:
        import re, json
        _ckpts = []
        for _name in os.listdir(dir_checkpoint):
            _m = re.fullmatch(r'(\d+)_G\.pth', _name)
            if _m:
                _ckpts.append(int(_m.group(1)))
        if not _ckpts:
            print("[resume] --continue_train: no checkpoint in " + dir_checkpoint + " yet, starting from the beginning.", flush=True)
        else:
            _it = max(_ckpts)
            _base = os.path.join(dir_checkpoint, str(_it) + '_G')
            _size = None
            try:
                with open(_base + '.json') as _f:
                    _size = json.load(_f).get('finesize')
            except Exception:
                pass
            if _size is not None and int(_size) != opt.finesize:
                print(f"[resume] The newest checkpoint ({_it}_G.pth) was trained at {_size}px but this run is at "
                      f"{opt.finesize}px, so it is NOT resumed: starting from the beginning.", flush=True)
            else:
                getattr(netG, 'module', netG).load_state_dict(torch.load(_base + '.pth', map_location='cpu'))
                _d_path = os.path.join(dir_checkpoint, str(_it) + '_D.pth')
                if not opt.no_gan and os.path.exists(_d_path):
                    getattr(netD, 'module', netD).load_state_dict(torch.load(_d_path, map_location='cpu'))
                _opt_msg = "optimizer state not found (saved by older versions only from now on): it restarts, which costs a few hundred iterations of settling"
                _rs = os.path.join(dir_checkpoint, 'resume_state.pth')
                if os.path.exists(_rs):
                    try:
                        _state = torch.load(_rs, map_location='cpu')
                        if _state.get('iteration') == _it:
                            optimizer_G.load_state_dict(_state['optimizer_G'])
                            if not opt.no_gan and 'optimizer_D' in _state:
                                optimizer_D.load_state_dict(_state['optimizer_D'])
                            _opt_msg = "optimizer state restored"
                    except Exception as _e:
                        _opt_msg = f"optimizer state could not be read ({_e}); it restarts"
                start_iter = _it + 1
                print(f"[resume] Continuing from {_it}_G.pth at iteration {start_iter:,} ({_opt_msg}).", flush=True)

    '''
    --------------------------Init DataLoader--------------------------
    '''
    videolist_tmp = os.listdir(opt.dataset)
    videolist = []
    for video in videolist_tmp:
        video_path = os.path.join(opt.dataset,video)
        if os.path.isdir(video_path):
            # A video folder is either synthetic (origin_image/ + mask/, mosaic
            # generated on-the-fly) or a real pair (origin_image/ + mosaic_image/,
            # already-censored frames used directly) — see util/dataloader.py.
            # Count frames from whichever of the two actually exists.
            if os.path.isdir(os.path.join(video_path,'mosaic_image')):
                frame_dir = os.path.join(video_path,'mosaic_image')
            else:
                frame_dir = os.path.join(video_path,'mask')
            if os.path.isdir(frame_dir) and len(os.listdir(frame_dir))>=opt.M:
                videolist.append(video)
    sorted(videolist)
    # Frames are resized so their short side is --loadsize. If the dataset was built with a
    # smaller --outsize, every frame is ENLARGED: slower and more memory, but no extra detail.
    _sides = []
    for _v in videolist[:8]:
        _d = os.path.join(opt.dataset, _v, 'origin_image')
        _f = sorted(os.listdir(_d))[:1] if os.path.isdir(_d) else []
        _img = impro.imread(os.path.join(_d, _f[0])) if _f else None
        if _img is not None:
            _sides.append(min(_img.shape[:2]))
    if _sides:
        _med = int(np.median(_sides))
        if _med < 0.85 * opt.loadsize:
            print(f"** WARNING: the dataset frames are about {_med}px, but training resizes them to {opt.loadsize}px, so they "
                  f"are enlarged {opt.loadsize/_med:.1f}x -- no extra detail, just more memory and time. Rebuild the dataset "
                  f"with --outsize {opt.loadsize} (or train at --finesize {int(round(_med * 256 / 286))}). **")
    videolist_train = videolist[:int(len(videolist)*0.8)].copy()
    videolist_eval = videolist[int(len(videolist)*0.8):].copy()

    # Every loader process imports PyTorch (~2 GB of memory on Windows) and keeps the frames of a whole
    # batch of clips in RAM. Too many of them fill RAM before the GPU is even used, so cap the number
    # to what the free memory can hold.
    try:
        from util import diagnose
        _m = diagnose.memory()
        if _m:
            _per = 1.0 + opt.batchsize * 2 * opt.S * opt.T * opt.finesize ** 2 * 12 / 1024.0 ** 3 * 1.3
            _budget = max(_per, _m['ram_free'] - 4.0)          # keep ~4 GB for the main process and Windows
            _fit = 1
            while _fit < opt.load_thread and (_fit + 1 + max(1, (_fit + 1) // 4)) * _per <= _budget:
                _fit += 1
            if _fit < opt.load_thread:
                print(f"[memory] {_m['ram_free']:.1f} GB RAM free; each loader process needs about {_per:.1f} GB "
                      f"(batch {opt.batchsize} at {opt.finesize}px). Using {_fit} loader process(es) instead of "
                      f"{opt.load_thread}. A smaller --batchsize allows more.", flush=True)
                opt.load_thread = _fit
    except Exception:
        pass
    n_eval_workers = max(1, opt.load_thread // 4)     # evaluation runs every 5th iteration only
    print(f"Clips: {len(videolist_train)} for training, {len(videolist_eval)} for evaluation. "
          f"Starting up to {opt.load_thread} + {n_eval_workers} data-loading processes...", flush=True)
    Videodataloader_train = dataloader.VideoDataLoader(opt, videolist_train)
    Videodataloader_eval = dataloader.VideoDataLoader(opt, videolist_eval, n_workers=n_eval_workers)
    print(f"Length of this run: {Videodataloader_train.n_iter:,} iterations = {len(videolist_train)} clips x "
          f"{Videodataloader_train.each_video_n_iter} iterations per clip x {opt.n_epoch} epochs / batch {opt.batchsize}. "
          f"A checkpoint is saved every {opt.save_freq:,} iterations and can be used at any point; lower --n_epoch for a shorter run.",
          flush=True)
    print("Waiting for the first batch. The first iterations are slow: the data-loading processes are still "
          "starting (each loads PyTorch)" + (", and ROCm compiles its GPU kernels on first use (cached for later runs)"
          if model_util.is_rocm() else "") + ". Later iterations are much faster.", flush=True)

    '''
    --------------------------Train--------------------------
    '''
    previous_predframe_tmp = 0
    # tqdm gives continuous feedback (iterations/sec, elapsed time, ETA) so
    # it's clear training is actually progressing between the much rarer
    # (every --showresult_freq iterations, 1000 by default) detailed status
    # lines below -- without this, there can be several minutes of complete
    # console silence on a small dataset, which looks identical to a hang.
    if start_iter >= Videodataloader_train.n_iter:
        print(f"[resume] The checkpoint is already at iteration {start_iter - 1:,}, the end of this run "
              f"({Videodataloader_train.n_iter:,} iterations). Nothing left to train; raise --n_epoch to train longer.", flush=True)
    progress_bar = tqdm(range(start_iter, Videodataloader_train.n_iter), desc='training', unit='iter',
                        initial=start_iter, total=Videodataloader_train.n_iter)
    _t_data = _t_gpu = 0.0          # where the time goes: waiting for data vs. the GPU step
    _on_cuda = torch.cuda.is_available() and opt.gpu_id != '-1'
    for train_iter in progress_bar:
        t_start = time.time()
        # train
        ori_stream,mosaic_stream,previous_frame = Videodataloader_train.get_data()
        _data_wait = time.time() - t_start
        ori_stream = data.to_tensor(ori_stream, opt.gpu_id, device=_device_for_calls)
        mosaic_stream = data.to_tensor(mosaic_stream, opt.gpu_id, device=_device_for_calls)
        if previous_frame is None:
            previous_frame = data.to_tensor(previous_predframe_tmp, opt.gpu_id, device=_device_for_calls)
        else:
            previous_frame = data.to_tensor(previous_frame, opt.gpu_id, device=_device_for_calls)

        ############### Forward ####################
        # Fake Generator
        out = netG(mosaic_stream,previous_frame)
        # Discriminator
        if not opt.no_gan:
            dis_real = netD(torch.cat((mosaic_stream[:,:,opt.N],ori_stream[:,:,opt.N].detach()),dim=1))
            dis_fake_D = netD(torch.cat((mosaic_stream[:,:,opt.N],out.detach()),dim=1))
            loss_D = lossfun_GAND(dis_fake_D,dis_real) * opt.lambda_GAN * opt.lambda_D
        # Generator
        loss_L2 = lossfun_L2(out,ori_stream[:,:,opt.N]) * opt.lambda_L2
        loss_VGG = lossfun_VGG(out,ori_stream[:,:,opt.N]) * opt.lambda_VGG
        loss_G = loss_L2+loss_VGG
        if not opt.no_gan:
            dis_fake_G = netD(torch.cat((mosaic_stream[:,:,opt.N],out),dim=1))
            loss_GANG = lossfun_GANG(dis_fake_G) * opt.lambda_GAN
            loss_G = loss_G + loss_GANG

        ############### Backward Pass ####################
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if not opt.no_gan:
            optimizer_D.zero_grad()
            loss_D.backward()        
            optimizer_D.step()

        previous_predframe_tmp = out.detach().cpu().numpy()

        if not opt.no_gan:
            TBGlobalWriter.add_scalars('loss/train', {'L2':loss_L2.item(),'VGG':loss_VGG.item(),
                'loss_D':loss_D.item(),'loss_G':loss_G.item()}, train_iter)
        else:
            TBGlobalWriter.add_scalars('loss/train', {'L2':loss_L2.item(),'VGG':loss_VGG.item()}, train_iter)

        # Live per-iteration feedback on the progress bar itself, distinct from
        # the much rarer (every --showresult_freq iters) detailed print below.
        # .item() above waited for the GPU, so the time since the data arrived is the GPU step.
        _gpu_step = time.time() - t_start - _data_wait
        _t_data += _data_wait; _t_gpu += _gpu_step
        progress_bar.set_postfix(L2=f'{loss_L2.item():.4f}', vgg=f'{loss_VGG.item():.4f}',
                                 data=f'{_data_wait:.2f}s', gpu=f'{_gpu_step:.2f}s')
        if train_iter in (20, 200):
            # One-off report: tells a data-loading bottleneck from a GPU one, and catches the GPU
            # running out of its own memory (Windows then silently uses much slower system RAM).
            n_done = train_iter + 1
            msg = (f"[speed] after {n_done} iterations: waiting for data {_t_data/n_done:.2f}s, "
                   f"GPU step {_t_gpu/n_done:.2f}s per iteration.")
            if _on_cuda:
                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    used = (total_b - free_b) / 1024**3; total = total_b / 1024**3
                    peak = torch.cuda.max_memory_reserved() / 1024**3
                    msg += f" GPU memory: {used:.1f} of {total:.1f} GB in use (this training reserved up to {peak:.1f} GB)."
                    if peak > 0.92 * total or used > 0.97 * total:
                        msg += (" ** The GPU memory is full. On Windows the driver then spills into system RAM, which makes "
                                "every step many times slower. Lower --batchsize (or the model size). **")
                except Exception:
                    pass
            if _t_data > 2 * _t_gpu and _t_data / n_done > 0.5:
                msg += (" ** Most of the time is spent waiting for data, not on the GPU: raise --load_thread if the CPU has "
                        "idle cores, and check that system RAM is not full. **")
            tqdm.write(msg)


        # save network
        if train_iter%opt.save_freq == 0 and train_iter != 0:
            save_generator(os.path.join('checkpoints',opt.savename,str(train_iter)+'_G.pth'))
            if not opt.no_gan:
                model_util.save(netD, os.path.join('checkpoints',opt.savename,str(train_iter)+'_D.pth'), opt.gpu_id, device=_device_for_calls)
            save_resume_state(train_iter)

        # Image quality evaluation 
        if train_iter%(opt.showresult_freq//10) == 0:
            ImageQualityEvaluation(out,ori_stream[:,:,opt.N],train_iter,TBGlobalWriter,'train')

        # Show result
        if train_iter % opt.showresult_freq == 0:
            ShowImage(mosaic_stream[:,:,opt.N],out,ori_stream[:,:,opt.N],train_iter,opt.showresult_num,TBGlobalWriter,'train')

        '''
        --------------------------Eval--------------------------
        '''
        if (train_iter)%5 ==0:
            ori_stream,mosaic_stream,previous_frame = Videodataloader_eval.get_data()
            ori_stream = data.to_tensor(ori_stream, opt.gpu_id, device=_device_for_calls)
            mosaic_stream = data.to_tensor(mosaic_stream, opt.gpu_id, device=_device_for_calls)
            if previous_frame is None:
                previous_frame = data.to_tensor(previous_predframe_tmp, opt.gpu_id, device=_device_for_calls)
            else:
                previous_frame = data.to_tensor(previous_frame, opt.gpu_id, device=_device_for_calls)
            with torch.no_grad():
                out = netG(mosaic_stream,previous_frame)
                loss_L2 = lossfun_L2(out,ori_stream[:,:,opt.N]) * opt.lambda_L2
                loss_VGG = lossfun_VGG(out,ori_stream[:,:,opt.N]) * opt.lambda_VGG
            #TBGlobalWriter.add_scalars('loss/eval', {'L2':loss_L2.item(),'VGG':loss_VGG.item()}, train_iter)
            previous_predframe_tmp = out.detach().cpu().numpy()

            # Image quality evaluation 
            if train_iter%(opt.showresult_freq//10) == 0:
                psnr,ssmi = ImageQualityEvaluation(out,ori_stream[:,:,opt.N],train_iter,TBGlobalWriter,'eval')

            # Show result
            if train_iter % opt.showresult_freq == 0:
                ShowImage(mosaic_stream[:,:,opt.N],out,ori_stream[:,:,opt.N],train_iter,opt.showresult_num,TBGlobalWriter,'eval')
                t_end = time.time()
                tqdm.write('iter:{0:d}  t:{1:.2f}  L2:{2:.4f}  vgg:{3:.4f}  psnr:{4:.2f}  ssmi:{5:.3f}'.format(train_iter,t_end-t_start,
                    loss_L2.item(),loss_VGG.item(),psnr,ssmi) )
                t_strat = time.time()

        '''
        --------------------------Test--------------------------
        '''
        if train_iter % opt.showresult_freq == 0 and os.path.isdir(opt.dataset_test):
            show_imgs = []
            videos = os.listdir(opt.dataset_test)
            sorted(videos)
            for video in videos:
                frames = os.listdir(os.path.join(opt.dataset_test,video,'image'))
                sorted(frames)
                for step in range(5):
                    mosaic_stream = []
                    for i in range(opt.T):
                        _mosaic = impro.imread(os.path.join(opt.dataset_test,video,'image',frames[i*opt.S+step]),loadsize=opt.finesize,rgb=True)
                        mosaic_stream.append(_mosaic)
                    if step == 0:
                        previous = impro.imread(os.path.join(opt.dataset_test,video,'image',frames[opt.N*opt.S-1]),loadsize=opt.finesize,rgb=True)
                        previous = data.im2tensor(previous,bgr2rgb = False, gpu_id = opt.gpu_id, device=_device_for_calls, is0_1 = False)
                    mosaic_stream = (np.array(mosaic_stream).astype(np.float32)/255.0-0.5)/0.5
                    mosaic_stream = mosaic_stream.reshape(1,opt.T,opt.finesize,opt.finesize,3).transpose((0,4,1,2,3))
                    mosaic_stream = data.to_tensor(mosaic_stream, opt.gpu_id, device=_device_for_calls)
                    with torch.no_grad():
                        out = netG(mosaic_stream,previous)
                    previous = out
                show_imgs+= [data.tensor2im(mosaic_stream[:,:,opt.N],rgb2bgr = False),data.tensor2im(out,rgb2bgr = False)]

            show_img = impro.splice(show_imgs, (len(videos),2))
            TBGlobalWriter.add_image('test', show_img,train_iter,dataformats='HWC')

    # Guaranteed final save, regardless of --save_freq. Without this, a run
    # whose total iteration count never happens to land on a multiple of
    # --save_freq (10000 by default) would complete having written NO
    # checkpoint at all -- a real risk for small datasets, where total
    # iterations can easily be a few thousand. Skips re-saving if the very
    # last iteration already triggered the periodic save above.
    final_iter = Videodataloader_train.n_iter - 1
    if final_iter < 0:
        print(f"** WARNING: zero training iterations ran for this bucket (dataset was empty or "
              f"too small for the configured --M/--S/--N/--batchsize) -- NO checkpoint was saved, "
              f"since there is no trained model to save. Check --dataset actually contains usable "
              f"video folders for this bucket before re-running. **")
    elif final_iter % opt.save_freq != 0 and final_iter >= start_iter:
        print(f"Saving final checkpoint at iter {final_iter}...")
        save_generator(os.path.join('checkpoints',opt.savename,str(final_iter)+'_G.pth'))
        print(f"Saved {final_iter}_G.pth (+ {final_iter}_G.json recording the {opt.finesize}px training size -- "
              f"copy the .json along with the .pth when you move or rename the checkpoint).")
        if not opt.no_gan:
            model_util.save(netD, os.path.join('checkpoints',opt.savename,str(final_iter)+'_D.pth'), opt.gpu_id, device=_device_for_calls)
        save_resume_state(final_iter)
    if final_iter >= 0:
        # Marker read by tools/train_all_buckets.py, so a later run skips buckets that are finished.
        try:
            import json
            with open(os.path.join('checkpoints', opt.savename, 'completed.json'), 'w') as _f:
                json.dump({'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'iterations': final_iter + 1,
                           'finesize': opt.finesize, 'batchsize': opt.batchsize, 'n_epoch': opt.n_epoch}, _f, indent=2)
        except OSError:
            pass
