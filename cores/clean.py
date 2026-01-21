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

torch.set_float32_matmul_precision('high')

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print("DirectML available for acceleration")
except ImportError:
    DIRECTML_AVAILABLE = False
    print("DirectML not available, using CUDA/CPU")

def setup_device(opt):
    """Setup optimal device for processing"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device, device_type
        
    elif DIRECTML_AVAILABLE:
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
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor: 
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
                    mask, x, y, size = runmodel.get_mosaic_position(img, netM, opt)  # Note: get_mosaic_position returns mask first
                    batch_positions.append([x, y, size])
                    if mask is not None:
                        batch_masks.append((mask, path))
                    consecutive_errors = 0  # Reset on success
                    
                    # Fixed preview code - check if mask exists and is valid
                    if not opt.no_preview and mask is not None and isinstance(mask, np.ndarray) and mask.size > 0:
                        if i % 2 == 0:  # Only show every other frame
                            try:
                                cv2.imshow('mosaic mask', mask)
                                cv2.waitKey(1)
                            except Exception as preview_error:
                                print(f"Preview error for {path}: {preview_error}")
                                
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
    
    # Optimized batch processing parameters
    batch_size = getattr(opt, 'clean_batch_size', 12)  # Increased batch size
    max_workers = min(os.cpu_count() or 4, 8)
    
    # Process all frames in order to avoid speed bursts
    print('Step:3/4 -- Clean Mosaic (Optimized):')
    
    # Pre-allocate queues
    write_queue = Queue(maxsize=batch_size * 4)
    preview_queue = Queue(maxsize=16)
    
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)

    def async_writer():
        import shutil
        while True:
            item = write_queue.get()
            if item is None:
                break
            save_path, img, delete_path, is_copy = item
            try:
                if is_copy:
                    shutil.copy2(delete_path, save_path)
                else:
                    cv2.imwrite(save_path, img)
                
                if delete_path and os.path.exists(delete_path):
                    os.remove(delete_path)
            except Exception as e:
                print(f"Writer error: {e}")
            finally:
                write_queue.task_done()

    def preview_handler():
        while True:
            item = preview_queue.get()
            if item is None:
                break
            try:
                cv2.imshow('clean', item)
                cv2.waitKey(1)
            except:
                pass
            finally:
                preview_queue.task_done()

    # Start worker threads
    writer_thread = Thread(target=async_writer, daemon=True)
    writer_thread.start()
    
    preview_thread = None
    if not opt.no_preview:
        preview_thread = Thread(target=preview_handler, daemon=True)
        preview_thread.start()

    def process_frame_batch(frame_data_batch):
        """Process frames in order - fast copy or neural network"""
        results = []
        import shutil
        
        for i, imagepath, x, y, size in frame_data_batch:
            try:
                src_path = os.path.join(opt.temp_dir, 'video2image', imagepath)
                dst_path = os.path.join(opt.temp_dir, 'replace_mosaic', imagepath)
                
                if size <= 100:
                    # Fast path - direct copy
                    shutil.copy2(src_path, dst_path)
                    img_result = None  # Don't load for preview to save time
                else:
                    # Slow path - neural network processing
                    img_origin = impro.imread(src_path)
                    try:
                        img_mosaic = img_origin[y-size:y+size, x-size:x+size]
                        
                        if opt.traditional:
                            img_fake = runmodel.traditional_cleaner(img_mosaic, opt)
                        else:
                            with torch.no_grad():
                                img_fake = runmodel.run_pix2pix(img_mosaic, netG, opt)
                        
                        mask_path = os.path.join(opt.temp_dir, 'mosaic_mask', imagepath)
                        if os.path.exists(mask_path):
                            mask = cv2.imread(mask_path, 0)
                            img_result = impro.replace_mosaic(img_origin, img_fake, mask, x, y, size, opt.no_feather)
                        else:
                            img_result = img_origin
                        
                        cv2.imwrite(dst_path, img_result)
                    except Exception as e:
                        print(f'Warning processing frame {i}: {e}')
                        shutil.copy2(src_path, dst_path)
                        img_result = None
                
                # Always remove source
                if os.path.exists(src_path):
                    os.remove(src_path)
                    
                results.append((i, imagepath, img_result, size > 100))
                
            except Exception as e:
                print(f'Error processing frame {imagepath}: {e}')
                results.append((i, imagepath, None, False))
        
        return results

    
    length = len(imagepaths)
    
    # Process all frames with adaptive batching to smooth out speed
    for batch_start in range(0, length, batch_size):
        batch_end = min(batch_start + batch_size, length)
        
        # Count mosaic frames in this batch to adjust processing
        mosaic_count = 0
        frame_batch = []
        for idx in range(batch_start, batch_end):
            imagepath = imagepaths[idx]
            x, y, size = positions[idx]
            frame_batch.append((idx, imagepath, x, y, size))
            if size > 100:
                mosaic_count += 1
        
        # If batch is all copies, process them one by one for steady progress
        if mosaic_count == 0 and len(frame_batch) > 4:
            # Process copy frames individually to avoid speed bursts
            for idx, imagepath, x, y, size in frame_batch:
                single_batch = [(idx, imagepath, x, y, size)]
                batch_results = process_frame_batch(single_batch)
                
                # Immediate progress update for smooth display
                current_frame = idx + 1
                t2 = time.time()
                print(f'\r{current_frame}/{length} '
                      f'{util.get_bar(100*current_frame/length, num=35)} '
                      f'{util.counttime(t1, t2, current_frame, length)}', end='')
                
                # Preview every 8th frame for copies
                if not opt.no_preview and idx % 8 == 0:
                    try:
                        img = impro.imread(os.path.join(opt.temp_dir, 'replace_mosaic', imagepath))
                        if img is not None:
                            preview_queue.put(img, block=False)
                    except:
                        pass
        else:
            # Normal batch processing for mixed or mosaic-heavy batches
            batch_results = process_frame_batch(frame_batch)
            
            # Handle previews for processed frames only
            for i, imagepath, img_result, was_processed in batch_results:
                if not opt.no_preview and img_result is not None and i % 5 == 0:
                    try:
                        preview_queue.put(img_result.copy(), block=False)
                    except:
                        pass
            
            # Progress reporting
            current_frame = batch_end
            t2 = time.time()
            print(f'\r{current_frame}/{length} '
                  f'{util.get_bar(100*current_frame/length, num=35)} '
                  f'{util.counttime(t1, t2, current_frame, length)}', end='')
    
    # Cleanup threads
    write_queue.put(None)
    writer_thread.join()
    
    if preview_thread:
        preview_queue.put(None)
        preview_thread.join()

    print('\nStep:4/4 -- Convert images to video')
    if not opt.no_preview:
        cv2.destroyAllWindows()

    ffmpeg.image2video(
        fps,
        os.path.join(opt.temp_dir, 'replace_mosaic', f'output_%06d.{opt.tempimage_type}'),
        os.path.join(opt.temp_dir, 'voice_tmp.mp3'),
        os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0] + '_clean.mp4')
    )

def cleanmosaic_video_fusion(opt, netG, netM):
    path = opt.media_path
    N, T, S = 2, 5, 3
    LEFT_FRAME = (N * S)
    POOL_NUM = LEFT_FRAME * 2 + 1
    INPUT_SIZE = 256
    FRAME_POS = np.linspace(0, (T-1)*S, T, dtype=np.int64)
    
    # Pre-allocate arrays to avoid repeated memory allocation
    input_stream_array = np.empty((1, T, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    temp_crop = np.empty((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    
    img_pool = []
    previous_frame = None
    init_flag = True
    
    fps, imagepaths, height, width = video_init(opt, path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt, netM, imagepaths, savemask=True)[(start_frame-1):]
    t1 = time.time()
    
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)
    
    # Optimize thread pools and queues
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    write_pool = Queue(16)  # Reduced queue size to save memory
    show_pool = Queue(8)    # Smaller preview queue
    
    # Pre-compile paths to avoid repeated string operations
    video2image_dir = opt.temp_dir + '/video2image'
    mosaic_mask_dir = opt.temp_dir + '/mosaic_mask'
    replace_mosaic_dir = opt.temp_dir + '/replace_mosaic'
    
    def write_result():
        while True:
            result = write_pool.get()
            if result is None:  # Sentinel to stop thread
                break
                
            save_ori, imagepath, img_origin, img_fake, x, y, size = result
            
            if save_ori:
                img_result = img_origin
            else:
                # Cache mask reading if same positions repeat
                mask_path = os.path.join(mosaic_mask_dir, imagepath)
                
                #VALIDATION HERE
                if not os.path.exists(mask_path):
                    print(f"\nWarning: Mask file not found for {imagepath}, using original image")
                    img_result = img_origin
                else:
                    mask = cv2.imread(mask_path, 0)
                    
                    # Validate mask is not None or empty
                    if mask is None or mask.size == 0:
                        print(f"\nWarning: Invalid mask for {imagepath}, using original image")
                        img_result = img_origin
                    else:
                        img_result = impro.replace_mosaic(img_origin, img_fake, mask, x, y, size, opt.no_feather)
            
            if not opt.no_preview and show_pool.qsize() < 4:
                show_pool.put(img_result.copy())
            
            # Write result
            cv2.imwrite(os.path.join(replace_mosaic_dir, imagepath), img_result)
            
            # Clean up original image file
            try:
                os.remove(os.path.join(video2image_dir, imagepath))
            except OSError:
                pass  # File might already be removed
    
    # Start writer thread
    writer_thread = Thread(target=write_result)
    writer_thread.daemon = True
    writer_thread.start()
    
    # Pre-calculate clipped indices to avoid repeated calculations
    pool_indices = []
    for i in range(length):
        indices = []
        if i == 0:  # init
            for j in range(POOL_NUM):
                indices.append(np.clip(i + j - LEFT_FRAME, 0, length - 1))
        else:
            indices.append(np.clip(i + LEFT_FRAME, 0, length - 1))
        pool_indices.append(indices)
    
    # Main processing loop
    for i, imagepath in enumerate(imagepaths):
        x, y, size = positions[i][0], positions[i][1], positions[i][2]
        
        # Optimized image pool management
        if i == 0:  # init
            img_pool.clear()  # Ensure clean start
            for idx in pool_indices[i]:
                img_path = os.path.join(video2image_dir, imagepaths[idx])
                img_pool.append(impro.imread(img_path))
        else:  # load next frame
            if len(img_pool) >= POOL_NUM:
                img_pool.pop(0)  # Remove oldest frame
            if pool_indices[i]:  # Only load if we have new index
                img_path = os.path.join(video2image_dir, imagepaths[pool_indices[i][0]])
                img_pool.append(impro.imread(img_path))
        
        img_origin = img_pool[LEFT_FRAME]
        
        # Non-blocking preview update
        if not opt.no_preview and not show_pool.empty():
            try:
                cv2.imshow('clean', show_pool.get_nowait())
                cv2.waitKey(1) & 0xFF
            except:
                pass
        
        # Process only if mosaic is large enough
        if size > 50:
            try:
                # Pre-calculate crop bounds to avoid repeated calculations
                y_start, y_end = max(0, y - size), min(img_origin.shape[0], y + size)
                x_start, x_end = max(0, x - size), min(img_origin.shape[1], x + size)
                
                # Build input stream more efficiently
                for idx, pos in enumerate(FRAME_POS):
                    if pos < len(img_pool):
                        crop = img_pool[pos][y_start:y_end, x_start:x_end]
                        # Resize directly into pre-allocated array slice
                        resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
                        input_stream_array[0, idx] = resized[:, :, ::-1]  # BGR to RGB
                
                if init_flag:
                    init_flag = False
                    # Convert middle frame for previous_frame
                    previous_frame = data.im2tensor(input_stream_array[0, N], bgr2rgb=False, gpu_id=opt.gpu_id)
                
                # Transpose in-place and convert to tensor
                input_tensor = data.to_tensor(
                    data.normalize(input_stream_array.transpose((0, 4, 1, 2, 3))), 
                    gpu_id=opt.gpu_id
                )
                
                # Model inference with minimal overhead
                with torch.inference_mode():
                    unmosaic_pred = netG(input_tensor, previous_frame)
                
                img_fake = data.tensor2im(unmosaic_pred, rgb2bgr=True)
                previous_frame = unmosaic_pred
                
                # Queue result for writing
                write_pool.put([False, imagepath, img_origin.copy(), img_fake, x, y, size])
                
            except Exception as e:
                print(f'\nError processing frame {i}: {e}')
                init_flag = True
                write_pool.put([True, imagepath, img_origin.copy(), -1, -1, -1, -1])
        else:
            write_pool.put([True, imagepath, img_origin.copy(), -1, -1, -1, -1])
            init_flag = True
        
        # Progress reporting (less frequent to reduce overhead)
        if i % 10 == 0 or i == length - 1:
            t2 = time.time()
            print(f'\r{i+1}/{length} {util.get_bar(100*i/length, num=35)} {util.counttime(t1, t2, i+1, length)}', end='')
    
    print()  # New line after progress
    
    # Clean shutdown
    write_pool.put(None)  # Sentinel to stop writer thread
    writer_thread.join(timeout=30)  # Wait for writer to finish
    
    if not opt.no_preview:
        cv2.destroyAllWindows()
    
    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video(
        fps,
        opt.temp_dir + '/replace_mosaic/output_%06d.' + opt.tempimage_type,
        opt.temp_dir + '/voice_tmp.mp3',
        os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0] + '_clean.mp4')
    )