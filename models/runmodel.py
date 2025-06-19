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
    
    # Cache commonly used values
    min_hw = min(h, w)
    rat = min_hw / 360.0
    ex_mun = int(min_hw / 20)
    
    # Run segmentation with error handling
    try:
        mask = run_segment(img_origin, net_mosaic_pos, size=360, gpu_id=opt.gpu_id)
    except Exception as e:
        print(f"Segmentation error: {e}")
        # Return default values if segmentation fails
        return w//2, h//2, 0, np.zeros((h, w), dtype=np.uint8)
    
    # Early exit if mask is empty
    if mask is None or mask.size == 0:
        return w//2, h//2, 0, np.zeros((h, w), dtype=np.uint8)
    
    # Apply threshold with cached ex_mun
    mask = impro.mask_threshold(mask, ex_mun=ex_mun, threshold=opt.mask_threshold)
    
    # Find most likely ROI if needed
    if not opt.all_mosaic_area:
        mask = impro.find_mostlikely_ROI(mask)
    
    # Get bounding square
    try:
        x, y, size, area = impro.boundingSquare(mask, Ex_mul=opt.ex_mult)
    except Exception as e:
        print(f"Bounding square error: {e}")
        return w//2, h//2, 0, mask
    
    # Location fix with cached ratio
    x = int(rat * x)
    y = int(rat * y) 
    size = int(rat * size)
    
    # Efficient clipping in one step
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    size = max(0, min(size, min(w - x, h - y)))
    
    return x, y, size, mask