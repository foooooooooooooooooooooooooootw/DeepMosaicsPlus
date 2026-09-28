import os
import sys
import shutil

# Windows consoles often default to a non-UTF-8 code page (cp1252/cp437),
# which mangles Unicode characters like em-dashes into '?' or a replacement
# character. Force UTF-8 stdout so any character in this script's output
# prints correctly regardless of the system's console code page.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
# Anchor this to the script's own location rather than the current working
# directory -- sys.path.append("..") only resolves correctly if the caller
# happened to cd into make_datasets/ first, which breaks this script when
# run from the project root or launched as a subprocess from elsewhere.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_here, '..'))
from cores import Options
opt = Options()

import random
import datetime
import time

import numpy as np
import cv2
import torch

from models import runmodel,loadmodel
import util.image_processing as impro
from util import filt, util,mosaic,data,ffmpeg


opt.parser.add_argument('--datadir',type=str,default='your video dir', help='')
opt.parser.add_argument('--savedir',type=str,default='../datasets/video/face', help='')
opt.parser.add_argument('--interval',type=int,default=30, help='interval of split video ')
opt.parser.add_argument('--time',type=int,default=5, help='split video time')
opt.parser.add_argument('--minmaskarea',type=int,default=2000, help='')
opt.parser.add_argument('--quality', type=int ,default= 45,help='minimal quality')
opt.parser.add_argument('--scene_accept_ratio', type=float, default=1.0,
    help='fraction of the --time sampled points within a candidate scene that must pass detection '
         '(area/size/quality) for that scene to be accepted, e.g. 0.6 = 60%% (3 of 5 by default). '
         'The original behavior requires ALL of them to pass (1.0) -- a single weak frame among the '
         'samples (motion blur, brief occlusion, a slightly-below-threshold detection) throws out the '
         'whole scene even if the rest was perfectly usable. Lowering this can substantially increase '
         'how much footage gets accepted, at the cost of occasionally including a scene where the ROI '
         'was only reliably detected part of the time.')
opt.parser.add_argument('--outsize', type=int ,default= 286,help='')
opt.parser.add_argument('--startcnt', type=int ,default= 0,help='')
opt.parser.add_argument('--minsize', type=int ,default= 96,help='minimal roi size')
opt.parser.add_argument('--no_sclectscene', action='store_true', help='')  
opt = opt.getparse()


util.makedirs(opt.savedir)
util.writelog(os.path.join(opt.savedir,'opt.txt'), 
              str(time.asctime(time.localtime(time.time())))+'\n'+util.opt2str(opt))

videopaths = util.Traversal(opt.datadir)
videopaths = util.is_videos(videopaths)
if not videopaths:
    print(f"Error: no video files found in --datadir '{opt.datadir}'.")
    print(f"Recognized video extensions: .mp4 .flv .avi .mov .mkv .wmv .rmvb .mts")
    print(f"Nothing to process — exiting without creating any per-video output.")
    sys.exit(1)
random.shuffle(videopaths)

# def network
net = loadmodel.bisenet(opt,'roi')

result_cnt = opt.startcnt
video_cnt = 1
starttime = datetime.datetime.now()
for videopath in videopaths:
    try:
        if opt.no_sclectscene:
            timestamps=['00:00:00']
        else:
            timestamps=[]
            fps,endtime,height,width = ffmpeg.get_video_infos(videopath)
            for cut_point in range(1,int((endtime-opt.time)/opt.interval)):
                util.clean_tempfiles(opt)
                ffmpeg.video2image(videopath, opt.temp_dir+'/video2image/%05d.'+opt.tempimage_type,fps=1,
                    start_time = util.second2stamp(cut_point*opt.interval),last_time = util.second2stamp(opt.time))
                imagepaths = util.Traversal(opt.temp_dir+'/video2image')
                imagepaths = sorted(imagepaths)
                cnt = 0 
                for i in range(opt.time):
                    img = impro.imread(imagepaths[i])
                    mask = runmodel.get_ROI_position(img,net,opt,keepsize=True)[0]
                    if not opt.all_mosaic_area:
                        mask = impro.find_mostlikely_ROI(mask)
                    x,y,size,area = impro.boundingSquare(mask,Ex_mul=1)
                    if area > opt.minmaskarea and size>opt.minsize and impro.Q_lapulase(img)>opt.quality:
                        cnt +=1
                if cnt >= opt.time * opt.scene_accept_ratio:
                    # print(second)
                    timestamps.append(util.second2stamp(cut_point*opt.interval))
        util.writelog(os.path.join(opt.savedir,'opt.txt'),videopath+'\n'+str(timestamps))
        #print(timestamps)

        #generate datasets
        print('Generate datasets...')
        for timestamp in timestamps:
            savecnt = '%05d' % result_cnt
            origindir = os.path.join(opt.savedir,savecnt,'origin_image')
            maskdir = os.path.join(opt.savedir,savecnt,'mask')
            util.makedirs(origindir)
            util.makedirs(maskdir)

            util.clean_tempfiles(opt)
            ffmpeg.video2image(videopath, opt.temp_dir+'/video2image/%05d.'+opt.tempimage_type,
                fps=opt.fps, start_time = timestamp,last_time = util.second2stamp(opt.time))
            
            endtime = datetime.datetime.now()
            print(str(video_cnt)+'/'+str(len(videopaths))+' ',
                util.get_bar(100*video_cnt/len(videopaths),35),'',
                util.second2stamp((endtime-starttime).seconds)+'/'+util.second2stamp((endtime-starttime).seconds/video_cnt*len(videopaths)))

            imagepaths = util.Traversal(opt.temp_dir+'/video2image')
            imagepaths = sorted(imagepaths)
            imgs=[];masks=[]
            # mask_flag = False
            # for imagepath in imagepaths:
            #     img = impro.imread(imagepath)
            #     mask = runmodel.get_ROI_position(img,net,opt,keepsize=True)[0]
            #     imgs.append(img)
            #     masks.append(mask)
            #     if not mask_flag:
            #         mask_avg = mask.astype(np.float64)
            #         mask_flag = True
            #     else:
            #         mask_avg += mask.astype(np.float64)

            # mask_avg = np.clip(mask_avg/len(imagepaths),0,255).astype('uint8')
            # mask_avg = impro.mask_threshold(mask_avg,20,64)
            # if not opt.all_mosaic_area:
            #     mask_avg = impro.find_mostlikely_ROI(mask_avg)
            # x,y,size,area = impro.boundingSquare(mask_avg,Ex_mul=random.uniform(1.1,1.5))
            
            # for i in range(len(imagepaths)):
            #     img = impro.resize(imgs[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC) 
            #     mask = impro.resize(masks[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
            #     impro.imwrite(os.path.join(origindir,'%05d'%(i+1)+'.jpg'), img)
            #     impro.imwrite(os.path.join(maskdir,'%05d'%(i+1)+'.png'), mask)
            ex_mul = random.uniform(1.2,1.7)
            positions = []
            for imagepath in imagepaths:
                img = impro.imread(imagepath)
                mask = runmodel.get_ROI_position(img,net,opt,keepsize=True)[0]
                imgs.append(img)
                masks.append(mask)
                x,y,size,area = impro.boundingSquare(mask,Ex_mul=ex_mul)
                positions.append([x,y,size])
            positions =np.array(positions)
            for i in range(3):positions[:,i] = filt.medfilt(positions[:,i],opt.medfilt_num)

            n_written = 0
            n_skipped_small = 0
            n_skipped_bounds = 0
            for i,imagepath in enumerate(imagepaths):
                x,y,size = positions[i][0],positions[i][1],positions[i][2]
                tmp_cnt = i
                fallback_steps = 0
                while size<opt.minsize//2 and fallback_steps < len(positions):
                    tmp_cnt = tmp_cnt-1
                    x,y,size = positions[tmp_cnt][0],positions[tmp_cnt][1],positions[tmp_cnt][2]
                    fallback_steps += 1
                if size<opt.minsize//2:
                    # No frame anywhere in this scene had a large-enough detected
                    # ROI/face to fall back to.
                    n_skipped_small += 1
                    continue

                # Clamp the crop to the frame's actual bounds before slicing.
                # x,y,size can end up pointing partly or fully outside the frame
                # (e.g. from median-filtering positions across a scene with
                # inconsistent per-frame detections — see filt.medfilt above),
                # which without clamping produces a crop that's zero-sized on
                # one axis while nonzero on the other (a plain size>0 check
                # like the one above does NOT catch this).
                h_img, w_img = imgs[i].shape[:2]
                y0, y1 = max(0, y-size), min(h_img, y+size)
                x0, x1 = max(0, x-size), min(w_img, x+size)
                if y1 <= y0 or x1 <= x0:
                    n_skipped_bounds += 1
                    continue

                img = impro.resize(imgs[i][y0:y1,x0:x1],opt.outsize,interpolation=cv2.INTER_CUBIC)
                mask = impro.resize(masks[i][y0:y1,x0:x1],opt.outsize,interpolation=cv2.INTER_CUBIC)
                impro.imwrite(os.path.join(origindir,'%05d'%(i+1)+'.jpg'), img)
                impro.imwrite(os.path.join(maskdir,'%05d'%(i+1)+'.png'), mask)
                n_written += 1
                # x_tmp,y_tmp,size_tmp

            if n_skipped_small or n_skipped_bounds:
                print(f"  Skipped {n_skipped_small} frame(s) with no large-enough detected ROI, "
                      f"{n_skipped_bounds} frame(s) with a detected box outside the frame bounds "
                      f"(out of {len(imagepaths)} total).")

            if n_written == 0:
                print(f"  No frame in this scene produced usable output — skipping scene, "
                      f"removing its empty folders.")
                shutil.rmtree(os.path.join(opt.savedir, savecnt), ignore_errors=True)
                continue

            # for imagepath in imagepaths:
            #     img = impro.imread(imagepath)
            #     mask,x,y,halfsize,area = runmodel.get_ROI_position(img,net,opt,keepsize=True)
            #     if halfsize>opt.minsize//4:
            #         if not opt.all_mosaic_area:
            #             mask_avg = impro.find_mostlikely_ROI(mask_avg)
            #         x,y,size,area = impro.boundingSquare(mask_avg,Ex_mul=ex_mul)
            #     img = impro.resize(imgs[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
            #     mask = impro.resize(masks[i][y-size:y+size,x-size:x+size],opt.outsize,interpolation=cv2.INTER_CUBIC)
            #     impro.imwrite(os.path.join(origindir,'%05d'%(i+1)+'.jpg'), img)
            #     impro.imwrite(os.path.join(maskdir,'%05d'%(i+1)+'.png'), mask)


            result_cnt+=1

    except Exception as e:
        video_cnt +=1
        print(f"Error processing '{videopath}': {e}")
        util.writelog(os.path.join(opt.savedir,'opt.txt'), 
              videopath+'\n'+str(result_cnt)+'\n'+str(e))
    video_cnt +=1
    if opt.gpu_id != '-1':
        torch.cuda.empty_cache()

n_succeeded = result_cnt - opt.startcnt
print(f"\nDone: {n_succeeded}/{len(videopaths)} video(s) produced output under {opt.savedir}.")
if n_succeeded == 0:
    print(f"** Every video failed to produce output. Check the error messages above "
          f"(also logged to {os.path.join(opt.savedir, 'opt.txt')}) — common causes: unreadable/corrupt "
          f"video file, no detectable ROI/face in the clip, or --minsize set too large for this content. **")
