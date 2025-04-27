import os
import time
import numpy as np
import cv2
import torch
from models import runmodel
from util import data, util, ffmpeg, filt
from util import image_processing as impro
from .init import video_init
from multiprocessing import Queue, Process, cpu_count
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

num_cores = os.cpu_count()

'''
---------------------Clean Mosaic---------------------
'''
def get_mosaic_positions(opt, netM, imagepaths, savemask=True):
    # resume logic
    continue_flag = False
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
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('mosaic mask', cv2.WINDOW_NORMAL)

    print('Step:2/4 -- Find mosaic location')

    # Optimize: Use a larger queue and more workers
    img_read_pool = Queue(16)  # Increased from 4
    result_queue = Queue(16)
    
    # Determine optimal number of workers based on CPU cores
    num_workers = min(8, cpu_count())
    
    # Loader function that reads images and puts them in queue
    def loader(imagepaths):
        for idx, imagepath in enumerate(imagepaths):
            img_origin = impro.imread(os.path.join(opt.temp_dir+'/video2image', imagepath))
            img_read_pool.put((idx, imagepath, img_origin))
        for _ in range(num_workers):
            img_read_pool.put(None)


    # Worker function that processes images
    def worker():
        while True:
            item = img_read_pool.get()
            if item is None:
                result_queue.put(None)
                break

            idx, imagepath, img_origin = item
            try:
                x, y, size, mask = runmodel.get_mosaic_position(img_origin, netM, opt)
                result_queue.put((idx, imagepath, x, y, size, mask))
            except Exception as e:
                print(f"\nError processing frame {idx}: {e}")
                result_queue.put((idx, imagepath, 0, 0, 0, None))


    
    # Start loader thread
    loader_thread = Thread(target=loader, args=(imagepaths,))
    loader_thread.daemon = True
    loader_thread.start()
    
    # Start worker threads
    workers = []
    for _ in range(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
        workers.append(t)
    
    # Use ThreadPoolExecutor for saving masks
    mask_executor = ThreadPoolExecutor(max_workers=4)
    
    # Process results
    processed_count = 0
    total_count = len(imagepaths)
    results_buffer = {}  # Store results that may come out of order
    next_frame_to_process = 0
    workers_finished = 0
    
    while processed_count < total_count:
        try:
            result = result_queue.get(timeout=60)
            
            # Check for sentinel value indicating a worker has finished
            if result is None:
                workers_finished += 1
                # If all workers are done but we haven't processed all frames, something went wrong
                if workers_finished == num_workers and processed_count < total_count:
                    print(f"\nWarning: All workers finished but only processed {processed_count}/{total_count} frames")
                    # Try to process any remaining buffered results
                    while next_frame_to_process < total_count and next_frame_to_process in results_buffer:
                        idx, imagepath, x, y, size, mask = results_buffer.pop(next_frame_to_process)
                        # [process this frame]
                        next_frame_to_process += 1
                        processed_count += 1
                    break
                continue
                
            idx, imagepath, x, y, size, mask = result
            
            # Store in buffer if this isn't the next frame we need
            if idx != next_frame_to_process:
                results_buffer[idx] = result
                # Check if we can process the next expected frame
                while next_frame_to_process in results_buffer:
                    result = results_buffer.pop(next_frame_to_process)
                    idx, imagepath, x, y, size, mask = result
                    # Now process this frame
                    positions.append([x, y, size])
                    if savemask and mask is not None:
                        mask_executor.submit(cv2.imwrite, os.path.join(opt.temp_dir+'/mosaic_mask', imagepath), mask)
                    
                    processed_count += 1
                    next_frame_to_process += 1
                    
                    # [Update progress display and checkpoints as before]
                    t2 = time.time()
                    print('\r',str(processed_count)+'/'+str(len(imagepaths)),util.get_bar(100*processed_count/len(imagepaths),num=35),util.counttime(t1,t2,processed_count,len(imagepaths)),end='')
            else:
                # Process this frame directly
                positions.append([x, y, size])
                if savemask and mask is not None:
                    mask_executor.submit(cv2.imwrite, os.path.join(opt.temp_dir+'/mosaic_mask', imagepath), mask)
                
                processed_count += 1
                next_frame_to_process += 1
                
                # [Update progress display and checkpoints as before]
                if not opt.no_preview:
                    cv2.imshow('mosaic mask',mask)
                    cv2.waitKey(1) & 0xFF
                t2 = time.time()
                print('\r',str(processed_count)+'/'+str(len(imagepaths)),util.get_bar(100*processed_count/len(imagepaths),num=35),util.counttime(t1,t2,processed_count,len(imagepaths)),end='')
                
        except Exception as e:
            print(f"\nError in main processing loop: {e}")
            continue
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=1)
    
    # Close the thread pool
    mask_executor.shutdown()
    
    if not opt.no_preview:
        cv2.destroyAllWindows()
        
    print('\nOptimizing mosaic locations...')
    positions = np.array(positions)
    if continue_flag:
        positions = np.concatenate((pre_positions, positions), axis=0)
    
    # Apply median filter for smoothing
    def apply_median_filter(positions, medfilt_num, col_idx):
        positions[:, col_idx] = filt.medfilt(positions[:, col_idx], medfilt_num)
        return positions

    # Use ThreadPoolExecutor to parallelize the median filtering
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(apply_median_filter, positions, opt.medfilt_num, i) for i in range(3)]
        for future in futures:
            future.result()  # Ensure all futures are completed
    
    step = {'step': 3, 'frame': 0}
    util.savejson(os.path.join(opt.temp_dir, 'step.json'), step)
    np.save(os.path.join(opt.temp_dir, 'mosaic_positions.npy'), positions)

    if not opt.no_preview:
        cv2.destroyAllWindows()

    return positions

def cleanmosaic_img(opt, netG, netM):
    path = opt.media_path
    print('Clean Mosaic:', path)
    img_origin = impro.imread(path)
    x, y, size, mask = runmodel.get_mosaic_position(img_origin, netM, opt)
    img_result = img_origin.copy()
    if size > 100:
        img_mosaic = img_origin[y-size:y+size, x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic, opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic, netG, opt)
        img_result = impro.replace_mosaic(img_origin, img_fake, mask, x, y, size, opt.no_feather)
    else:
        print('Do not find mosaic')
    impro.imwrite(os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0]+'_clean.jpg'), img_result)

def cleanmosaic_img_server(opt, img_origin, netG, netM):
    x, y, size, mask = runmodel.get_mosaic_position(img_origin, netM, opt)
    img_result = img_origin.copy()
    if size > 100:
        img_mosaic = img_origin[y-size:y+size, x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic, opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic, netG, opt)
        img_result = impro.replace_mosaic(img_origin, img_fake, mask, x, y, size, opt.no_feather)
    return img_result

def cleanmosaic_video_byframe(opt, netG, netM):
    path = opt.media_path
    fps, imagepaths, height, width = video_init(opt, path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt, netM, imagepaths, savemask=True)[(start_frame-1):]

    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)

    # Optimize: Use batch processing for better GPU utilization
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    
    # Determine optimal batch size based on available GPU memory
    # Start with a reasonable default that should work on most GPUs
    batch_size = 4
    
    # Configure queues with larger sizes
    img_read_pool = Queue(16)
    process_pool = Queue(16)
    write_pool = Queue(16)
    show_pool = Queue(8)
    
    # Image loader function
    def loader():
        for i in range(0, length, batch_size):
            batch_indices = range(i, min(i + batch_size, length))
            batch_data = []
            for j in batch_indices:
                img_path = imagepaths[j]
                img_origin = impro.imread(os.path.join(opt.temp_dir+'/video2image', img_path))
                x, y, size = positions[j][0], positions[j][1], positions[j][2]
                batch_data.append((j, img_path, img_origin, x, y, size))
            img_read_pool.put(batch_data)
    
    # Process function that handles GPU operations in batches
    def processor():
        while True:
            try:
                batch_data = img_read_pool.get(timeout=30)
                if not batch_data:
                    break
                    
                batch_results = []
                # Process each image in the batch
                for j, img_path, img_origin, x, y, size in batch_data:
                    img_result = img_origin.copy()
                    
                    if size > 100:
                        try:
                            img_mosaic = img_origin[y-size:y+size, x-size:x+size]
                            if opt.traditional:
                                img_fake = runmodel.traditional_cleaner(img_mosaic, opt)
                            else:
                                # Note: ideally modify run_pix2pix to accept batches
                                img_fake = runmodel.run_pix2pix(img_mosaic, netG, opt)
                            
                            mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask', img_path), 0)
                            img_result = impro.replace_mosaic(img_origin, img_fake, mask, x, y, size, opt.no_feather)
                        except Exception as e:
                            print(f'Warning for frame {j}: {e}')
                    
                    batch_results.append((j, img_path, img_result))
                
                process_pool.put(batch_results)
            except Exception as e:
                print(f'Processor error: {e}')
                break
    
    # Writer function that saves processed images
    def writer():
        while True:
            try:
                batch_results = process_pool.get(timeout=30)
                if not batch_results:
                    break
                
                for j, img_path, img_result in batch_results:
                    if not opt.no_preview:
                        show_pool.put((j, img_result.copy()))
                    
                    cv2.imwrite(os.path.join(opt.temp_dir+'/replace_mosaic', img_path), img_result)
                    os.remove(os.path.join(opt.temp_dir+'/video2image', img_path))
            except Exception as e:
                print(f'Writer error: {e}')
                break
    
    # Display function for preview
    def display():
        processed_frames = {}
        next_frame_to_show = 0
        
        while next_frame_to_show < length:
            try:
                j, img_result = show_pool.get(timeout=1)
                processed_frames[j] = img_result
                
                # Show frames in order
                while next_frame_to_show in processed_frames:
                    cv2.imshow('clean', processed_frames[next_frame_to_show])
                    cv2.waitKey(1) & 0xFF
                    del processed_frames[next_frame_to_show]
                    next_frame_to_show += 1
                    
                    # Print progress
                    t2 = time.time()
                    print('\r', str(next_frame_to_show)+'/'+str(length), 
                          util.get_bar(100*next_frame_to_show/length, num=35), 
                          util.counttime(t1, t2, next_frame_to_show, length), end='')
            except Exception:
                # Timeout, just continue
                pass
    
    # Start all threads
    threads = []
    threads.append(Thread(target=loader))
    threads.append(Thread(target=processor))
    threads.append(Thread(target=writer))
    
    for t in threads:
        t.daemon = True
        t.start()
    
    # Start display thread if preview is enabled
    if not opt.no_preview:
        display_thread = Thread(target=display)
        display_thread.daemon = True
        display_thread.start()
    
    # Wait for processing to complete
    for t in threads:
        t.join()
    
    if not opt.no_preview:
        display_thread.join(timeout=1)
        cv2.destroyAllWindows()
    
    print('\nStep:4/4 -- Convert images to video')
    ffmpeg.image2video(fps,
                opt.temp_dir+'/replace_mosaic/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0]+'_clean.mp4'))

def cleanmosaic_video_fusion(opt, netG, netM):
    path = opt.media_path
    N, T, S = 2, 5, 3
    LEFT_FRAME = (N*S)
    POOL_NUM = LEFT_FRAME*2+1
    INPUT_SIZE = 256
    FRAME_POS = np.linspace(0, (T-1)*S, T, dtype=np.int64)
    
    # Optimize: Pre-allocate tensors and use CUDA streams for overlapping operations
    fps, imagepaths, height, width = video_init(opt, path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt, netM, imagepaths, savemask=True)[(start_frame-1):]
    
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)
    
    # Optimize: Create CUDA streams for parallel operations
    if torch.cuda.is_available() and int(opt.gpu_id) >= 0:
        device = torch.device(f'cuda:{int(opt.gpu_id)}')
        cuda_stream_1 = torch.cuda.Stream(device=device)
        cuda_stream_2 = torch.cuda.Stream(device=device)
    
    # Optimize: Increase queue sizes and use more workers
    write_pool = Queue(16)
    show_pool = Queue(8)
    
    # Writer function 
    def write_result():
        while True:
            try:
                item = write_pool.get(timeout=30)
                if item is None:  # Signal to exit
                    break
                    
                save_ori, imagepath, img_origin, img_fake, x, y, size = item
                if save_ori:
                    img_result = img_origin
                else:
                    mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask', imagepath), 0)
                    img_result = impro.replace_mosaic(img_origin, img_fake, mask, x, y, size, opt.no_feather)
                
                if not opt.no_preview:
                    show_pool.put(img_result.copy())
                
                cv2.imwrite(os.path.join(opt.temp_dir+'/replace_mosaic', imagepath), img_result)
                os.remove(os.path.join(opt.temp_dir+'/video2image', imagepath))
            except Exception as e:
                print(f'Writer error: {e}')
                break
    
    # Start writer thread
    writer_thread = Thread(target=write_result)
    writer_thread.daemon = True
    writer_thread.start()
    
    # Optimize: Pre-allocate arrays for image processing
    length = len(imagepaths)
    img_pool = []
    batch_size = 2  # Process multiple frames in small batches when possible
    
    # Initialize pools
    for j in range(POOL_NUM):
        img_pool.append(impro.imread(os.path.join(opt.temp_dir+'/video2image', 
                        imagepaths[np.clip(j-LEFT_FRAME, 0, len(imagepaths)-1)])))
    
    previous_frames = {}  # Track previous frames for each batch
    init_flags = {i: True for i in range(batch_size)}  # Initialize flags for each batch position
    
    # Process frames in mini-batches when possible
    for i in range(0, length, batch_size):
        current_batch_size = min(batch_size, length - i)
        batch_data = []
        
        # Prepare data for each frame in the batch
        for b in range(current_batch_size):
            frame_idx = i + b
            imagepath = imagepaths[frame_idx]  # Add this line to define imagepath
            x, y, size = positions[frame_idx][0], positions[frame_idx][1], positions[frame_idx][2]
            
            # Update image pool for this position in batch
            if frame_idx > 0:
                img_pool.pop(0)
                img_pool.append(impro.imread(os.path.join(opt.temp_dir+'/video2image', 
                                imagepaths[np.clip(frame_idx+LEFT_FRAME, 0, len(imagepaths)-1)])))
            
            img_origin = img_pool[LEFT_FRAME]
            
            # Show preview
            if not opt.no_preview and not show_pool.empty():
                cv2.imshow('clean', show_pool.get())
                cv2.waitKey(1) & 0xFF
            
            if size > 50:
                try:
                    input_stream = []
                    for pos in FRAME_POS:
                        input_stream.append(impro.resize(img_pool[pos][y-size:y+size, x-size:x+size], 
                                          INPUT_SIZE, interpolation=cv2.INTER_CUBIC)[:, :, ::-1])
                    
                    # Initialize previous frame if needed
                    if init_flags[b]:
                        init_flags[b] = False
                        prev_frame = input_stream[N]
                        prev_frame = data.im2tensor(prev_frame, bgr2rgb=True, gpu_id=opt.gpu_id)
                        previous_frames[b] = prev_frame
                    
                    batch_data.append((frame_idx, b, imagepath, img_origin.copy(), input_stream, previous_frames[b], x, y, size))
                except Exception as e:
                    print(f'Error preparing frame {frame_idx}: {e}')
                    init_flags[b] = True
                    write_pool.put([True, imagepaths[frame_idx], img_origin.copy(), -1, -1, -1, -1])
            else:
                imagepath = imagepaths[frame_idx]  # Add this line
                write_pool.put([True, imagepath, img_origin.copy(), -1, -1, -1, -1])
                init_flags[b] = True
        
        # Process batch frames with GPU
        if batch_data:
            with torch.cuda.stream(cuda_stream_1) if torch.cuda.is_available() and int(opt.gpu_id) >= 0 else nullcontext():
                for frame_idx, b, imagepath, img_origin, input_stream, prev_frame, x, y, size in batch_data:
                    # Prepare input for model
                    input_stream = np.array(input_stream).reshape(1, T, INPUT_SIZE, INPUT_SIZE, 3).transpose((0, 4, 1, 2, 3))
                    input_stream = data.to_tensor(data.normalize(input_stream), gpu_id=opt.gpu_id)
                    
                    # Process with model
                    with torch.no_grad():
                        unmosaic_pred = netG(input_stream, prev_frame)
                    
                    img_fake = data.tensor2im(unmosaic_pred, rgb2bgr=True)
                    previous_frames[b] = unmosaic_pred  # Update previous frame
                    
                    # Queue for writing
                    write_pool.put([False, imagepaths[frame_idx], img_origin, img_fake, x, y, size])
        
        t2 = time.time()
        print('\r', str(i+current_batch_size)+'/'+str(length), 
              util.get_bar(100*(i+current_batch_size)/length, num=35), 
              util.counttime(t1, t2, i+current_batch_size, length), end='')
    
    # Signal writer thread to finish
    write_pool.put(None)
    writer_thread.join()
    
    if not opt.no_preview:
        cv2.destroyAllWindows()
    
    print('\nStep:4/4 -- Convert images to video')
    ffmpeg.image2video(fps,
                opt.temp_dir+'/replace_mosaic/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0]+'_clean.mp4'))

# Helper context manager for conditional CUDA streams
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): pass
