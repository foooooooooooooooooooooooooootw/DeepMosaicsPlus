import cv2
import sys
sys.path.append("..")
import util.image_processing as impro
from util import mosaic
from util import data
import torch
import numpy as np

torch.set_float32_matmul_precision('high')

def run_segment(img,net,size = 360,gpu_id = '-1'):
    img = impro.resize(img,size)
    img = data.im2tensor(img, gpu_id = gpu_id, bgr2rgb = False, is0_1 = True)
    # Make sure the tensor is on the same device as the model
    img = img.to(next(net.parameters()).device).float()
    mask = net(img)
    mask = data.tensor2im(mask, gray=True, is0_1 = True)
    return mask

def run_pix2pix(img, net, opt):
    if opt.netG == 'HD':
        img = impro.resize(img, 512)
    else:
        img = impro.resize(img, 128)

    device = next(net.parameters()).device
    
    # DirectML-specific tensor handling
    if 'directml' in str(device):
        try:
            # Force CPU tensor creation first for DirectML
            img_tensor = data.im2tensor(img, device=torch.device('cpu'))
            img_tensor = img_tensor.contiguous()
            
            # Move to DirectML device only when needed
            img_tensor = img_tensor.to(device)
            
            # Clear any cached tensors before inference
            if hasattr(torch, 'directml') and hasattr(torch.directml, 'empty_cache'):
                torch.directml.empty_cache()
            
            # Ensure model is in eval mode
            net.eval()
            
            with torch.inference_mode():
                img_fake = net(img_tensor)
                # Immediately move to CPU and clone to break DirectML tensor chain
                img_fake = img_fake.detach().cpu().clone()
                
            # Clear DirectML cache after inference
            if hasattr(torch, 'directml') and hasattr(torch.directml, 'empty_cache'):
                torch.directml.empty_cache()
                
            img_fake = data.tensor2im(img_fake)
            return img_fake
            
        except Exception as e:
            print(f"DirectML error, falling back to CPU: {e}")
            # Complete fallback to CPU processing
            try:
                net_cpu = net.cpu()
                img_tensor = data.im2tensor(img, device=torch.device('cpu'))
                net_cpu.eval()
                with torch.inference_mode():
                    img_fake = net_cpu(img_tensor)
                img_fake = data.tensor2im(img_fake)
                return img_fake
            except Exception as e2:
                print(f"CPU fallback also failed: {e2}")
                # Return original image if all else fails
                return img
    else:
        # Normal CUDA/CPU path
        img_tensor = data.im2tensor(img, device=device)
        with torch.inference_mode():
            img_fake = net(img_tensor)
        img_fake = data.tensor2im(img_fake)
        return img_fake


def traditional_cleaner(img,opt):
    h,w = img.shape[:2]
    img = cv2.blur(img, (opt.tr_blur,opt.tr_blur))
    img = img[::opt.tr_down,::opt.tr_down,:]
    img = cv2.resize(img, (w,h),interpolation=cv2.INTER_LANCZOS4)
    return img

def run_styletransfer(opt, net, img):

    if opt.output_size != 0:
        if 'resize' in opt.preprocess and 'resize_scale_width' not in opt.preprocess:
            img = impro.resize(img,opt.output_size)
        elif 'resize_scale_width' in opt.preprocess:
            img = cv2.resize(img, (opt.output_size,opt.output_size))
        img = img[0:4*int(img.shape[0]/4),0:4*int(img.shape[1]/4),:]

    if 'edges' in opt.preprocess:
        if opt.canny > 100:
            canny_low = opt.canny-50
            canny_high = np.clip(opt.canny+50,0,255)
        elif opt.canny < 50:
            canny_low = np.clip(opt.canny-25,0,255)
            canny_high = opt.canny+25
        else:
            canny_low = opt.canny-int(opt.canny/2)
            canny_high = opt.canny+int(opt.canny/2)
        img = cv2.Canny(img,canny_low,canny_high)
        if opt.only_edges:
            return img
        img = data.im2tensor(img,gpu_id=opt.gpu_id,gray=True)
    else:    
        img = data.im2tensor(img,gpu_id=opt.gpu_id)
    img = net(img)
    img = data.tensor2im(img)
    return img

def get_ROI_position(img,net,opt,keepsize=True):
    mask = run_segment(img,net,size=360,gpu_id = opt.gpu_id)
    mask = impro.mask_threshold(mask,opt.mask_extend,opt.mask_threshold)
    if keepsize:
        mask = impro.resize_like(mask, img)
    x,y,halfsize,area = impro.boundingSquare(mask, 1)
    return mask,x,y,halfsize,area

def get_mosaic_position(img_origin, net_mosaic_pos, opt):
    h, w = img_origin.shape[:2]
    
    # Run segmentation
    try:
        mask = run_segment(img_origin, net_mosaic_pos, size=360, gpu_id=opt.gpu_id)
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None, 0, 0, 0
    
    if mask is None or mask.size == 0:
        return None, 0, 0, 0
    
    # Ensure mask is a valid numpy array
    if not isinstance(mask, np.ndarray):
        print(f"Invalid mask type: {type(mask)}")
        return None, 0, 0, 0
    
    # Resize mask to match original image
    try:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        print(f"Mask resize error: {e}")
        return None, 0, 0, 0
    
    # Clean up the mask
    ex_mun = int(min(h, w) / 20)
    mask = impro.mask_threshold(mask, ex_mun=ex_mun, threshold=opt.mask_threshold)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    
    # Remove small components
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        min_area = 300
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask[labels == i] = 0
    except Exception as e:
        print(f"Connected components error: {e}")
        return None, 0, 0, 0
    
    # Find most likely ROI if needed
    if not opt.all_mosaic_area:
        mask = impro.find_mostlikely_ROI(mask)
    
    # Calculate total area and center of mass
    try:
        mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        area = np.sum(mask_binary > 0)
        
        if area == 0:
            return None, 0, 0, 0
        
        # Calculate center of mass
        y_coords, x_coords = np.where(mask_binary > 0)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None, 0, 0, 0
            
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        # Calculate effective "size" as the maximum extent
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        size = max(x_max - x_min, y_max - y_min) // 2
        
        return mask, center_x, center_y, size
        
    except Exception as e:
        print(f"Mask processing error: {e}")
        return None, 0, 0, 0