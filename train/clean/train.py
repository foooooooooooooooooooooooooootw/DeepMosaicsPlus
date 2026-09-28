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
opt.parser.add_argument('--finesize',type=int,default=256, help='')
opt.parser.add_argument('--loadsize',type=int,default=286, help='')
opt.parser.add_argument('--batchsize',type=int,default=1, help='')
opt.parser.add_argument('--no_gan', action='store_true', help='if specified, do not use gan')
opt.parser.add_argument('--n_blocks',type=int,default=4, help='')
opt.parser.add_argument('--n_layers_D',type=int,default=2, help='')
opt.parser.add_argument('--num_D',type=int,default=3, help='')
opt.parser.add_argument('--lambda_L2',type=float,default=100, help='')
opt.parser.add_argument('--lambda_VGG',type=float,default=1, help='')
opt.parser.add_argument('--lambda_GAN',type=float,default=0.01, help='')
opt.parser.add_argument('--lambda_D',type=float,default=1, help='')
opt.parser.add_argument('--load_thread',type=int,default=16, help='number of thread for loading data')

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
opt.parser.add_argument('--continue_train', action='store_true', help='')
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
    if opt.showresult_num >opt.batchsize:
        opt.showresult_num = opt.batchsize
    dir_checkpoint = os.path.join('checkpoints',opt.savename)
    util.makedirs(dir_checkpoint)
    # start tensorboard
    localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    tensorboard_savedir = os.path.join('checkpoints/tensorboard',localtime+'_'+opt.savename)
    TBGlobalWriter = SummaryWriter(tensorboard_savedir)
    print('Please run "tensorboard --logdir checkpoints/tensorboardX --host=your_server_ip" and input "'+localtime+'" to filter outputs')

    '''
    --------------------------Init Network--------------------------
    '''
    if opt.gpu_id != '-1' and len(opt.gpu_id) == 1:
        torch.backends.cudnn.benchmark = True

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
    videolist_train = videolist[:int(len(videolist)*0.8)].copy()
    videolist_eval = videolist[int(len(videolist)*0.8):].copy()

    Videodataloader_train = dataloader.VideoDataLoader(opt, videolist_train)
    Videodataloader_eval = dataloader.VideoDataLoader(opt, videolist_eval)

    '''
    --------------------------Train--------------------------
    '''
    previous_predframe_tmp = 0
    # tqdm gives continuous feedback (iterations/sec, elapsed time, ETA) so
    # it's clear training is actually progressing between the much rarer
    # (every --showresult_freq iterations, 1000 by default) detailed status
    # lines below -- without this, there can be several minutes of complete
    # console silence on a small dataset, which looks identical to a hang.
    progress_bar = tqdm(range(Videodataloader_train.n_iter), desc='training', unit='iter')
    for train_iter in progress_bar:
        t_start = time.time()
        # train
        ori_stream,mosaic_stream,previous_frame = Videodataloader_train.get_data()
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
        progress_bar.set_postfix(L2=f'{loss_L2.item():.4f}', vgg=f'{loss_VGG.item():.4f}')

        # save network
        if train_iter%opt.save_freq == 0 and train_iter != 0:
            model_util.save(netG, os.path.join('checkpoints',opt.savename,str(train_iter)+'_G.pth'), opt.gpu_id, device=_device_for_calls)
            if not opt.no_gan:
                model_util.save(netD, os.path.join('checkpoints',opt.savename,str(train_iter)+'_D.pth'), opt.gpu_id, device=_device_for_calls)

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
    elif final_iter % opt.save_freq != 0:
        print(f"Saving final checkpoint at iter {final_iter}...")
        model_util.save(netG, os.path.join('checkpoints',opt.savename,str(final_iter)+'_G.pth'), opt.gpu_id, device=_device_for_calls)
        if not opt.no_gan:
            model_util.save(netD, os.path.join('checkpoints',opt.savename,str(final_iter)+'_D.pth'), opt.gpu_id, device=_device_for_calls)
