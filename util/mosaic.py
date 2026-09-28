import cv2
import numpy as np
import os
import random
from .image_processing import resize,ch_one2three,mask_area

# ── Mosaic block-size (granularity) detection ────────────────────────────────
# This is unrelated to `get_autosize` above, which estimates a *plausible*
# mosaic size to synthetically apply when building training data. The
# functions below instead measure the *actual* pixelation block size already
# present in a real, already-mosaiced image/frame, so a "size-matched" clean
# model can be picked automatically (see cores/options.py --auto_model).

def _axis_period(diff_signal, min_block_px, max_block_px, prefer_smallest_ratio=0.85):
    """Given a 1-D signal of pixel-to-pixel differences along one axis,
    find the period (in pixels) of its strongest periodic component via
    autocorrelation, plus a 0-1 confidence score (peak height relative to
    the signal's own energy). Returns (period_px, confidence) or (None, 0)
    if the signal is too short or has no clear peak.

    Among lags whose autocorrelation is within `prefer_smallest_ratio` of
    the single best peak, the smallest lag is preferred rather than the
    global argmax. This matters because resizing an already-mosaiced image
    can introduce its own aliasing periodicity — usually a small integer
    multiple of the true block size — that sometimes scores *higher* than
    the true, smaller period. Real mosaic blocks are never larger than a
    handful of these near-tied candidates, so biasing toward the smallest
    one meaningfully reduces (without eliminating) that failure mode.
    """
    n = len(diff_signal)
    if n < min_block_px * 2:
        return None, 0.0

    x = diff_signal.astype(np.float64)
    x = x - x.mean()
    energy = float(np.dot(x, x))
    if energy < 1e-8:
        return None, 0.0

    max_lag = min(max_block_px, n // 2)
    if max_lag <= min_block_px:
        return None, 0.0

    # Full autocorrelation via FFT (fast, and exact for this length range)
    autocorr = np.correlate(x, x, mode='full')[n - 1:]
    autocorr = autocorr / autocorr[0]

    search = autocorr[min_block_px:max_lag + 1]
    if search.size == 0:
        return None, 0.0
    peak_val = float(np.max(search))
    if peak_val <= 0:
        return None, 0.0
    near_peak = np.where(search >= peak_val * prefer_smallest_ratio)[0]
    best_idx = int(near_peak[0]) if near_peak.size else int(np.argmax(search))
    best_lag = min_block_px + best_idx
    confidence = max(0.0, min(1.0, float(search[best_idx])))
    return best_lag, confidence


def estimate_mosaic_block_pct(img, mask, min_block_px=2, max_block_px=128, margin=4,
                               min_pct=None, max_pct=None, par=1.0, squareness_tol=0.20):
    """Estimate the pixel size of the mosaic blocks already present inside
    `mask`'s region of `img`, expressed as a percentage of the frame's
    shorter side (matches the convention used by get_autosize() above, so
    it lines up with how this codebase already generates synthetic
    training mosaics).

    Design choice: mosaic blocks are assumed square by default (the normal
    case). Rather than blending a width estimate and a height estimate
    into one number the moment they disagree — which would silently hide a
    genuine problem — this reports both axes and flags disagreement beyond
    `squareness_tol` (default: 20%) via `is_square=False` and a reduced
    `confidence`, so a caller can choose to skip auto-selection rather than
    trust a guess on an oddly-shaped detection.

    par: pixel aspect ratio (width_scale/height_scale) from
    util.ffmpeg.get_pixel_aspect_ratio(). If the source has non-square
    pixels (rare for full-raster 1080p+, more common on older/anamorphic
    masters), pass it here to correct the horizontal measurement before
    the squareness check — otherwise a real PAR mismatch looks identical
    to "these axes disagree, something's wrong."

    Returns a dict:
        {'pct': float,          # combined estimate for bucket lookup
         'pct_w': float,        # block width as % of frame width
         'pct_h': float,        # block height as % of frame height
         'block_w_px': float, 'block_h_px': float,
         'is_square': bool,     # False if axes disagree beyond squareness_tol
         'confidence': float}
    or None if the region is too small / ambiguous to measure at all.

    When is_square is True, 'pct' is block_px / min(h,w) * 100 using the
    average of the two (agreeing) axis measurements — directly comparable
    to how get_autosize() sizes synthetic training mosaics.
    When is_square is False, 'pct' falls back to the geometric mean of
    pct_w/pct_h (~sqrt of the area ratio) purely as a rough diagnostic
    number — treat it as "something unusual here, go look," not as a
    trustworthy bucket key.

    Other caveats:
      - Re-compression after the mosaic was applied barely affects accuracy
        in testing.
      - Rescaling the frame after the mosaic was applied is the main real
        risk: it can introduce its own periodic aliasing pattern that
        sometimes scores as a *more* confident (but wrong, usually larger)
        period than the true block size. Constraining min_pct/max_pct to
        your actual expected range guards against this.
    """
    if img is None or mask is None or img.size == 0 or mask.size == 0:
        return None

    h, w = img.shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None

    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin)
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin)
    roi = img[y0:y1, x0:x1]
    if roi.shape[0] < min_block_px * 4 or roi.shape[1] < min_block_px * 4:
        return None

    frame_short_side = min(h, w)
    lo_px, hi_px = min_block_px, max_block_px
    if min_pct is not None:
        lo_px = max(lo_px, int(frame_short_side * min_pct / 100.0))
    if max_pct is not None:
        hi_px = min(hi_px, int(np.ceil(frame_short_side * max_pct / 100.0)))
    if lo_px >= hi_px:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = gray.astype(np.float64)

    # Column-to-column difference, averaged over rows -> spikes at multiples
    # of the block width along the x axis.
    diff_cols = np.mean(np.abs(np.diff(gray, axis=1)), axis=0)
    # Row-to-row difference, averaged over columns -> block height along y.
    diff_rows = np.mean(np.abs(np.diff(gray, axis=0)), axis=1)

    period_x, conf_x = _axis_period(diff_cols, lo_px, hi_px)
    period_y, conf_y = _axis_period(diff_rows, lo_px, hi_px)

    if period_x is None and period_y is None:
        return None
    # Correct the horizontal measurement for non-square pixels before
    # comparing axes, so a real PAR mismatch isn't mistaken for detector noise.
    block_w_px = period_x * par if period_x is not None else None
    block_h_px = period_y if period_y is not None else None

    if block_w_px is None:
        block_w_px = block_h_px
    if block_h_px is None:
        block_h_px = block_w_px

    pct_w = (block_w_px / w) * 100.0
    pct_h = (block_h_px / h) * 100.0

    ratio = block_w_px / block_h_px if block_h_px else 1.0
    is_square = (1 - squareness_tol) <= ratio <= (1 + squareness_tol)

    confidences = [c for c in (conf_x, conf_y) if c is not None]
    base_confidence = float(np.mean(confidences)) if confidences else 0.0

    if is_square:
        block_px = float(np.mean([p for p in (block_w_px, block_h_px)]))
        pct = (block_px / frame_short_side) * 100.0
        confidence = base_confidence
    else:
        # Axes disagree beyond tolerance: don't quietly average them into a
        # number that matches neither axis. Surface a diagnostic estimate
        # and penalize confidence so callers can choose to skip auto-select.
        pct = float(np.sqrt(pct_w * pct_h))
        confidence = base_confidence * 0.5

    return {
        'pct': pct, 'pct_w': pct_w, 'pct_h': pct_h,
        'block_w_px': block_w_px, 'block_h_px': block_h_px,
        'is_square': is_square, 'confidence': confidence,
    }


def estimate_mosaic_block_pct_multi(frame_mask_pairs, min_confidence=0.15, min_pct=None, max_pct=None, par=1.0):
    """Run estimate_mosaic_block_pct over several (img, mask) samples (e.g.
    frames spread across a video) and return the median across the
    confident ones. Sampling several frames instead of one is strongly
    recommended — single-frame estimates are noisy, especially near motion
    blur, compression, or partial occlusion of the mosaic region. Note this
    does NOT protect against a systematic resize-aliasing bias (see
    estimate_mosaic_block_pct's docstring) since that bias is the same on
    every frame of a given source.

    Returns {'pct': float, 'confidence': float, 'n_samples': int,
             'n_square': int, 'n_nonsquare': int} or None if no sample was
    confident enough to trust. If a meaningful fraction of samples come
    back non-square, that's a real signal worth investigating (see
    is_square in estimate_mosaic_block_pct) rather than something to
    average past.
    """
    estimates = []
    for img, mask in frame_mask_pairs:
        try:
            est = estimate_mosaic_block_pct(img, mask, min_pct=min_pct, max_pct=max_pct, par=par)
        except Exception:
            est = None
        if est is not None and est['confidence'] >= min_confidence:
            estimates.append(est)
    if not estimates:
        return None
    pct = float(np.median([e['pct'] for e in estimates]))
    confidence = float(np.median([e['confidence'] for e in estimates]))
    n_square = sum(1 for e in estimates if e['is_square'])
    return {
        'pct': pct, 'confidence': confidence, 'n_samples': len(estimates),
        'n_square': n_square, 'n_nonsquare': len(estimates) - n_square,
    }


def addmosaic(img,mask,opt):
    if opt.mosaic_mod == 'random':
        img = addmosaic_random(img,mask)
    elif opt.mosaic_size == 0:
        img = addmosaic_autosize(img, mask, opt.mosaic_mod)
    else:
        img = addmosaic_base(img,mask,opt.mosaic_size,opt.output_size,model = opt.mosaic_mod)
    return img

def addmosaic_base(img,mask,n,out_size = 0,model = 'squa_avg',rect_rat = 1.6,feather=0,start_point=[0,0]):
    '''
    img: input image
    mask: input mask
    n: mosaic size
    out_size: output size  0->original
    model : squa_avg squa_mid squa_random squa_avg_circle_edge rect_avg
    rect_rat: if model==rect_avg , mosaic w/h=rect_rat
    feather : feather size, -1->no 0->auto
    start_point : [0,0], please not input this parameter
    '''
    n = int(n)
    
    h_start = np.clip(start_point[0], 0, n)
    w_start = np.clip(start_point[1], 0, n)
    pix_mid_h = n//2+h_start
    pix_mid_w = n//2+w_start
    h, w = img.shape[:2]
    h_step = (h-h_start)//n
    w_step = (w-w_start)//n
    if out_size:
        img = resize(img,out_size)      
    if mask.shape[0] != h:
        mask = cv2.resize(mask,(w,h))
    img_mosaic = img.copy()

    if model=='squa_avg':
        for i in range(h_step):
            for j in range(w_step):
                if mask[i*n+pix_mid_h,j*n+pix_mid_w]:
                    img_mosaic[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:]=\
                           img[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:].mean(axis=(0,1))

    elif model=='squa_mid':
        for i in range(h_step):
            for j in range(w_step):
                if mask[i*n+pix_mid_h,j*n+pix_mid_w]:
                    img_mosaic[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:]=\
                           img[i*n+n//2+h_start,j*n+n//2+w_start,:]

    elif model == 'squa_random':
        for i in range(h_step):
            for j in range(w_step):
                if mask[i*n+pix_mid_h,j*n+pix_mid_w]:
                    img_mosaic[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:]=\
                    img[h_start+int(i*n-n/2+n*random.random()),w_start+int(j*n-n/2+n*random.random()),:]

    elif model == 'squa_avg_circle_edge':
        for i in range(h_step):
            for j in range(w_step):
                img_mosaic[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:]=\
                       img[i*n+h_start:(i+1)*n+h_start,j*n+w_start:(j+1)*n+w_start,:].mean(axis=(0,1))
        mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1]
        _mask = ch_one2three(mask)
        mask_inv = cv2.bitwise_not(_mask)
        imgroi1 = cv2.bitwise_and(_mask,img_mosaic)
        imgroi2 = cv2.bitwise_and(mask_inv,img)
        img_mosaic = cv2.add(imgroi1,imgroi2)

    elif model =='rect_avg':
        n_h = n
        n_w = int(n*rect_rat)
        n_h_half = n_h//2+h_start
        n_w_half = n_w//2+w_start
        for i in range((h-h_start)//n_h):
            for j in range((w-w_start)//n_w):
                if mask[i*n_h+n_h_half,j*n_w+n_w_half]:
                    img_mosaic[i*n_h+h_start:(i+1)*n_h+h_start,j*n_w+w_start:(j+1)*n_w+w_start,:]=\
                           img[i*n_h+h_start:(i+1)*n_h+h_start,j*n_w+w_start:(j+1)*n_w+w_start,:].mean(axis=(0,1))
    
    if feather != -1:
        if feather==0:
            mask = (cv2.blur(mask, (n, n)))
        else:
            mask = (cv2.blur(mask, (feather, feather)))
        mask = mask/255.0
        for i in range(3):img_mosaic[:,:,i] = (img[:,:,i]*(1-mask)+img_mosaic[:,:,i]*mask)
        img_mosaic = img_mosaic.astype(np.uint8)
    
    return img_mosaic

def get_autosize(img,mask,area_type = 'normal'):
    h,w = img.shape[:2]
    size = np.min([h,w])
    mask = resize(mask,size)
    alpha = size/512
    try:
        if area_type == 'normal':
            area = mask_area(mask)
        elif area_type == 'bounding':
            w,h = cv2.boundingRect(mask)[2:]
            area = w*h
    except:
        area = 0
    area = area/(alpha*alpha)
    if area>50000:
        size = alpha*((area-50000)/50000+12)
    elif 20000<area<=50000:
        size = alpha*((area-20000)/30000+8)
    elif 5000<area<=20000:
        size = alpha*((area-5000)/20000+7)
    elif 0<=area<=5000:
        size = alpha*((area-0)/5000+6)
    else:
        pass
    return size

def get_random_parameter(img,mask,target_pct=None):
    # mosaic size
    if target_pct is not None:
        # Constrain to a narrow band around a specific bucket size (e.g. for
        # training a size-specialized model — see --dataset_mosaic_pct in
        # train/clean/train.py), rather than the wide free-ranging default
        # below. Size is computed relative to this image's own shorter side,
        # matching the same convention used by the --auto_model detector
        # (util.mosaic.estimate_mosaic_block_pct), so a model trained this
        # way stays calibrated with how it'll be selected at inference time.
        h, w = img.shape[:2]
        base_px = target_pct / 100.0 * min(h, w)
        mosaic_size = int(base_px * random.uniform(0.85, 1.15))
        mosaic_size = max(2, mosaic_size)
    else:
        p = np.array([0.5,0.5])
        mod = np.random.choice(['normal','bounding'], p = p.ravel())
        mosaic_size = get_autosize(img,mask,area_type = mod)
        mosaic_size = int(mosaic_size*random.uniform(0.9,2.5))

    # mosaic mod
    p = np.array([0.25, 0.3, 0.45])
    mod = np.random.choice(['squa_mid','squa_avg','rect_avg'], p = p.ravel())

    # rect_rat for rect_avg
    rect_rat = random.uniform(1.1,1.6)
    
    # feather size
    feather = -1
    if random.random()<0.7:
        feather = int(mosaic_size*random.uniform(0,1.5))

    return mosaic_size,mod,rect_rat,feather


def addmosaic_autosize(img,mask,model,area_type = 'normal'):
    mosaic_size = get_autosize(img,mask,area_type = 'normal')
    img_mosaic = addmosaic_base(img,mask,mosaic_size,model = model)
    return img_mosaic

def addmosaic_random(img,mask):
    mosaic_size,mod,rect_rat,feather = get_random_parameter(img,mask)
    img_mosaic = addmosaic_base(img,mask,mosaic_size,model = mod,rect_rat=rect_rat,feather=feather)
    return img_mosaic

def get_random_startpos(num,bisa_p,bisa_max,bisa_max_part):
    pos = np.zeros((num,2), dtype=np.int64)
    if random.random()<bisa_p:
        indexs = random.sample((np.linspace(1,num-1,num-1,dtype=np.int64)).tolist(), random.randint(1, bisa_max_part))
        indexs.append(0)
        indexs.append(num)
        indexs.sort()
        for i in range(len(indexs)-1):
            pos[indexs[i]:indexs[i+1]] = [random.randint(0,bisa_max),random.randint(0,bisa_max)]
    return pos