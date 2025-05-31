import os
import time
import numpy as np
import cv2
import torch
from models import runmodel
from util import data,util,ffmpeg,filt
from util import image_processing as impro
from .init import video_init
from multiprocessing import Queue, Process
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from pathlib import Path

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print("DirectML available for acceleration")
except ImportError:
    DIRECTML_AVAILABLE = False
    print("DirectML not available, using CUDA/CPU")

def setup_device(opt):
    """Setup optimal device for processing"""
    if DIRECTML_AVAILABLE:
        try:
            device = torch_directml.device()
            device_type = 'directml'
            print(f"Using DirectML device: {device}")
            return device, device_type
        except Exception as e:
            print(f"DirectML initialization failed: {e}, falling back to CUDA/CPU")
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        print("Using CPU device")
    return device, device_type

'''
---------------------Clean Mosaic---------------------
'''
def get_mosaic_positions(opt, netM, imagepaths, savemask=True):
    """Optimized mosaic position detection with batch processing and error handling"""
    device, device_type = setup_device(opt)
    
    try:
        netM.to(device)
        netM.eval()
    except Exception as e:
        print(f"Error moving netM to {device_type}: {e}, using CPU")
        device = torch.device('cpu')
        device_type = 'cpu'
        netM = netM.cpu()
        netM.eval()
    
    # Check for resume capability
    continue_flag = False
    resume_frame = 0
    if os.path.isfile(os.path.join(opt.temp_dir, 'step.json')):
        step = util.loadjson(os.path.join(opt.temp_dir, 'step.json'))
        resume_frame = int(step['frame'])
        if int(step['step']) > 2:
            pre_positions = np.load(os.path.join(opt.temp_dir, 'mosaic_positions.npy'))
            return pre_positions
        if int(step['step']) >= 2 and resume_frame > 0:
            pre_positions = np.load(os.path.join(opt.temp_dir, 'mosaic_positions.npy'))
            continue_flag = True
            imagepaths = imagepaths[resume_frame:]
    
    positions = []
    batch_size = getattr(opt, 'position_batch_size', 4)  # Reduced batch size
    
    print('Step:2/4 -- Find mosaic location (DirectML)')
    
    if not opt.no_preview:
        cv2.namedWindow('mosaic mask', cv2.WINDOW_NORMAL)
    
    t1 = time.time()
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    # Process in batches
    for batch_start in range(0, len(imagepaths), batch_size):
        if consecutive_errors >= max_consecutive_errors:
            print(f"\nToo many consecutive errors in position detection, using CPU")
            device = torch.device('cpu')
            device_type = 'cpu'
            netM = netM.cpu()
            consecutive_errors = 0
        
        batch_end = min(batch_start + batch_size, len(imagepaths))
        batch_paths = imagepaths[batch_start:batch_end]
        
        # Load batch of images
        batch_images = []
        with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers
            futures = [executor.submit(impro.imread, 
                                     os.path.join(opt.temp_dir, 'video2image', path)) 
                      for path in batch_paths]
            
            for future in futures:
                try:
                    img = future.result(timeout=15)  # Increased timeout
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    batch_images.append(None)
                    consecutive_errors += 1
        
        # Process batch through network
        batch_positions = []
        batch_masks = []
        
        for i, (img, path) in enumerate(zip(batch_images, batch_paths)):
            if img is not None:
                try:
                    x, y, size, mask = runmodel.get_mosaic_position(img, netM, opt)
                    batch_positions.append([x, y, size])
                    batch_masks.append((mask, path))
                    consecutive_errors = 0  # Reset on success
                    
                    if not opt.no_preview:
                        cv2.imshow('mosaic mask', mask)
                        cv2.waitKey(1) & 0xFF
                        
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    batch_positions.append([0, 0, 0])
                    consecutive_errors += 1
            else:
                batch_positions.append([0, 0, 0])
        
        positions.extend(batch_positions)
        
        # Save masks asynchronously
        if savemask and batch_masks:
            def save_mask_batch(mask_data):
                for mask, path in mask_data:
                    try:
                        cv2.imwrite(os.path.join(opt.temp_dir, 'mosaic_mask', path), mask)
                    except Exception as e:
                        print(f"Error saving mask for {path}: {e}")
            
            Thread(target=save_mask_batch, args=(batch_masks,)).start()
        
        # Progress and checkpointing
        current_frame = batch_end + resume_frame
        if current_frame % 1000 == 0:
            save_positions = np.array(positions)
            if continue_flag:
                save_positions = np.concatenate((pre_positions, save_positions), axis=0)
            np.save(os.path.join(opt.temp_dir, 'mosaic_positions.npy'), save_positions)
            step = {'step': 2, 'frame': current_frame}
            util.savejson(os.path.join(opt.temp_dir, 'step.json'), step)
        
        t2 = time.time()
        print(f'\r{current_frame}/{len(imagepaths)+resume_frame} '
              f'{util.get_bar(100*current_frame/(len(imagepaths)+resume_frame), num=35)} '
              f'{util.counttime(t1, t2, current_frame, len(imagepaths)+resume_frame)}', end='')
    
    if not opt.no_preview:
        cv2.destroyAllWindows()
    
    print('\nOptimize mosaic locations...')
    positions = np.array(positions)
    if continue_flag:
        positions = np.concatenate((pre_positions, positions), axis=0)
    
    # Apply median filtering
    for i in range(3):
        positions[:, i] = filt.medfilt(positions[:, i], opt.medfilt_num)
    
    step = {'step': 3, 'frame': 0}
    util.savejson(os.path.join(opt.temp_dir, 'step.json'), step)
    np.save(os.path.join(opt.temp_dir, 'mosaic_positions.npy'), positions)
    
    return positions

def cleanmosaic_img(opt,netG,netM):

    path = opt.media_path
    print('Clean Mosaic:',path)
    img_origin = impro.imread(path)
    x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
    #cv2.imwrite('./mask/'+os.path.basename(path), mask)
    img_result = img_origin.copy()
    if size > 100 :
        img_mosaic = img_origin[y-size:y+size,x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
        img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
    else:
        print('Do not find mosaic')
    impro.imwrite(os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.jpg'),img_result)

def cleanmosaic_img_server(opt,img_origin,netG,netM):
    x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
    img_result = img_origin.copy()
    if size > 100 :
        img_mosaic = img_origin[y-size:y+size,x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
        img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
    return img_result

def cleanmosaic_video_byframe(opt, netG, netM):
    path = opt.media_path
    fps, imagepaths, height, width = video_init(opt, path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt, netM, imagepaths, savemask=True)[(start_frame - 1):]

    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)

    # Set up async writer thread and queue
    write_queue = Queue(maxsize=8)

    def async_writer():
        while True:
            item = write_queue.get()
            if item is None:
                break
            save_path, img, delete_path = item
            try:
                cv2.imwrite(save_path, img)
                os.remove(delete_path)
            except Exception as e:
                print(f"Writer error: {e}")
            write_queue.task_done()

    writer_thread = Thread(target=async_writer, daemon=True)
    writer_thread.start()

    # Clean mosaic
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    for i, imagepath in enumerate(imagepaths, 0):
        x, y, size = positions[i]
        img_origin = impro.imread(os.path.join(opt.temp_dir, 'video2image', imagepath))
        img_result = img_origin.copy()

        if size > 100:
            try:  # Avoid unknown errors
                img_mosaic = img_origin[y-size:y+size, x-size:x+size]
                if opt.traditional:
                    img_fake = runmodel.traditional_cleaner(img_mosaic, opt)
                else:
                    img_fake = runmodel.run_pix2pix(img_mosaic, netG, opt)
                mask = cv2.imread(os.path.join(opt.temp_dir, 'mosaic_mask', imagepath), 0)
                img_result = impro.replace_mosaic(img_origin, img_fake, mask, x, y, size, opt.no_feather)
            except Exception as e:
                print('Warning:', e)

        save_path = os.path.join(opt.temp_dir, 'replace_mosaic', imagepath)
        delete_path = os.path.join(opt.temp_dir, 'video2image', imagepath)
        write_queue.put((save_path, img_result, delete_path))

        # Preview result and print progress
        if not opt.no_preview:
            cv2.imshow('clean', img_result)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r', f"{i+1}/{length} {util.get_bar(100*i/length, num=35)} {util.counttime(t1, t2, i+1, length)}", end='')

    # Finalize writer thread
    write_queue.put(None)
    writer_thread.join()

    print()
    if not opt.no_preview:
        cv2.destroyAllWindows()

    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video(
        fps,
        os.path.join(opt.temp_dir, 'replace_mosaic', f'output_%06d.{opt.tempimage_type}'),
        os.path.join(opt.temp_dir, 'voice_tmp.mp3'),
        os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0] + '_clean.mp4')
    )


def cleanmosaic_video_fusion(opt,netG,netM):
    path = opt.media_path
    N,T,S = 2,5,3
    LEFT_FRAME = (N*S)
    POOL_NUM = LEFT_FRAME*2+1
    INPUT_SIZE = 256
    FRAME_POS = np.linspace(0, (T-1)*S,T,dtype=np.int64)
    img_pool = []
    previous_frame = None
    init_flag = True
    
    fps,imagepaths,height,width = video_init(opt,path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt,netM,imagepaths,savemask=True)[(start_frame-1):]
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)
    
    # clean mosaic
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    write_pool = Queue(4)
    show_pool = Queue(4)
    def write_result():
        while True:
            save_ori,imagepath,img_origin,img_fake,x,y,size = write_pool.get()
            if save_ori:
                img_result = img_origin
            else:
                mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask',imagepath),0)
                img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
            if not opt.no_preview:
                show_pool.put(img_result.copy())
            cv2.imwrite(os.path.join(opt.temp_dir+'/replace_mosaic',imagepath),img_result)
            os.remove(os.path.join(opt.temp_dir+'/video2image',imagepath))
    t = Thread(target=write_result,args=())
    t.setDaemon(True)
    t.start()

    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        input_stream = []
        # image read stream
        if i==0 :# init
            for j in range(POOL_NUM):
                img_pool.append(impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepaths[np.clip(i+j-LEFT_FRAME,0,len(imagepaths)-1)])))
        else: # load next frame
            img_pool.pop(0)
            img_pool.append(impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepaths[np.clip(i+LEFT_FRAME,0,len(imagepaths)-1)])))
        img_origin = img_pool[LEFT_FRAME]

        # preview result and print
        if not opt.no_preview:
            if show_pool.qsize()>3:   
                cv2.imshow('clean',show_pool.get())
                cv2.waitKey(1) & 0xFF

        if size>50:
            try:#Avoid unknown errors
                for pos in FRAME_POS:
                    input_stream.append(impro.resize(img_pool[pos][y-size:y+size,x-size:x+size], INPUT_SIZE,interpolation=cv2.INTER_CUBIC)[:,:,::-1])
                if init_flag:
                    init_flag = False
                    previous_frame = input_stream[N]
                    previous_frame = data.im2tensor(previous_frame,bgr2rgb=True,gpu_id=opt.gpu_id)
                
                input_stream = np.array(input_stream).reshape(1,T,INPUT_SIZE,INPUT_SIZE,3).transpose((0,4,1,2,3))
                input_stream = data.to_tensor(data.normalize(input_stream),gpu_id=opt.gpu_id)
                with torch.no_grad():
                    unmosaic_pred = netG(input_stream,previous_frame)
                img_fake = data.tensor2im(unmosaic_pred,rgb2bgr = True)
                previous_frame = unmosaic_pred
                write_pool.put([False,imagepath,img_origin.copy(),img_fake.copy(),x,y,size])
            except Exception as e:
                init_flag = True
                print('Error:',e)
        else:
            write_pool.put([True,imagepath,img_origin.copy(),-1,-1,-1,-1])
            init_flag = True
        
        t2 = time.time()
        print('\r',str(i+1)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i+1,len(imagepaths)),end='')
    print()
    write_pool.close()
    show_pool.close()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video( fps,
                opt.temp_dir+'/replace_mosaic/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.mp4')) 