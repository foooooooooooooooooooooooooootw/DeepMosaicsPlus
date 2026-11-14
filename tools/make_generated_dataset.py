"""
Build training clips for the video clean model WITHOUT the ROI detector.

Two kinds of clip, both written in the same layout make_datasets/make_video_dataset.py
produces (<savedir>/<clip>/origin_image/00001.jpg ... + mask/00001.png ...), so they can
go straight into the synthetic pool and be trained with tools/train_all_buckets.py or
train/clean/train.py. Training applies the mosaic itself, at random block sizes (or at one
bucket's size with --dataset_mosaic_pct), inside the white area of each mask -- so one set
of clips serves every mosaic size.

  generated   Procedural animation in two looks (--style):
                anime -- flat colours, hard-stepped (cel-shaded) gradients, dark outlines,
                         sharp and noise-free
                real  -- smooth multi-shade gradients with uneven light, soft-edged shaded
                         shapes, fine surface texture, lens blur and sensor noise
                mixed -- each clip is one or the other (default)
              --skin_tone sets how much of the colour is beige / pink / tan (default 0.6).
              Needs no footage at all. Good for teaching the model block removal at a given
              resolution and for flat-colour (anime-like) content; it contains no real
              anatomy or texture, so mix it with real footage rather than train on it alone.

  crops       Random square regions of YOUR clean videos or images (--source), with the
              mosaic placed on a random area. No detection involved, so it works on content
              the ROI model does not recognise (anime, drawings, unusual footage).
              Still images become clips by slowly panning and zooming over them.

Examples (run from the project folder):
  python tools/make_generated_dataset.py --mode generated --clips 200
  python tools/make_generated_dataset.py --clips 300 --style anime --skin_tone 0.8
  python tools/make_generated_dataset.py --mode crops --source D:/clean_anime --clips 500
  python tools/make_generated_dataset.py --mode generated --clips 4 --preview preview.jpg

Clips are named gen_00000, crop_00000, ... (continuing after existing ones), so they never
collide with make_video_dataset.py's numbered folders and are easy to remove again.
Only needs numpy and OpenCV.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_EXT = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.m4v', '.mts', '.gif')
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')


# ----------------------------------------------------------------------------- helpers

def imwrite(path, img, params=None):
    """cv2.imwrite fails on non-ASCII Windows paths; encode + tofile does not."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img, params or [])
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(path)


def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


class Palette:
    """Colours for one clip. `style` is 'anime' or 'real'; `skin` (0..1) is the share of
    colours drawn from beige / pink / tan tones instead of the style's general range."""

    def __init__(self, rng, style, skin):
        self.rng, self.style, self.skin = rng, style, float(skin)

    @staticmethod
    def _bgr(h, s, v):
        bgr = cv2.cvtColor(np.uint8([[[h % 180, np.clip(s, 0, 255), np.clip(v, 0, 255)]]]), cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(int(c) for c in bgr)

    def skin_tone(self):
        """Beige, pink and tan shades (OpenCV hue is 0-179: ~8-16 is beige/orange-brown,
        ~168-179 and 0-6 is pink/rose)."""
        rng = self.rng
        family = rng.choice(['beige', 'pink', 'tan'], p=[0.45, 0.4, 0.15])
        if family == 'beige':
            return self._bgr(rng.uniform(8, 17), rng.uniform(25, 110), rng.uniform(165, 252))
        if family == 'pink':
            return self._bgr(rng.uniform(166, 187), rng.uniform(35, 140), rng.uniform(185, 255))
        return self._bgr(rng.uniform(6, 15), rng.uniform(90, 170), rng.uniform(85, 185))

    def color(self, kind='any'):
        rng = self.rng
        if kind == 'dark':                       # outlines and deep shadows
            if self.style == 'anime':
                return self._bgr(rng.uniform(0, 180), rng.uniform(20, 160), rng.uniform(10, 70))
            return self._bgr(rng.uniform(0, 180), rng.uniform(20, 120), rng.uniform(15, 90))
        if rng.random() < self.skin:
            return self.skin_tone()
        if self.style == 'anime':                # clean, bright, fairly saturated
            if rng.random() < 0.5:
                return self._bgr(rng.uniform(0, 180), rng.uniform(30, 110), rng.uniform(195, 255))    # pastel
            return self._bgr(rng.uniform(0, 180), rng.uniform(130, 245), rng.uniform(140, 255))       # vivid
        # real: muted, wide brightness range
        return self._bgr(rng.uniform(0, 180), rng.uniform(10, 130), rng.uniform(30, 235))

    def shade(self, bgr, amount):
        """The same colour lighter (amount > 0) or darker (< 0); darker also gets a little
        more saturated, the way shadows on skin and cel-shaded art do."""
        h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
        return self._bgr(h, s * (1 - 0.35 * amount), v * (1 + amount))


def gradient_image(rng, size, pal):
    """Full-canvas gradient (float32 BGR), in the clip's style:
    anime -- few colours; usually hard-stepped bands (cel shading), sometimes a clean
             two-colour sky-like blend.
    real  -- many closely related shades of one or two colours with soft, uneven light
             falloff, like lighting on skin, fabric or a wall."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    if rng.random() < 0.6:
        ang = rng.uniform(0, 2 * math.pi)
        t = xs * math.cos(ang) + ys * math.sin(ang)
    else:
        cx, cy = rng.uniform(0.1, 0.9, 2)
        t = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
    if pal.style == 'anime':
        base = pal.color()
        if rng.random() < 0.5:
            cols = [pal.shade(base, a) for a in sorted(rng.uniform(-0.45, 0.25, int(rng.integers(2, 5))))]
        else:
            cols = [base, pal.color()]
        if rng.random() < 0.65:                  # cel shading: flat bands with hard edges
            bands = len(cols) if len(cols) > 2 else int(rng.integers(2, 6))
            t = np.floor(np.clip(t, 0, 0.9999) * bands) / max(1, bands - 1)
    else:
        base = pal.color()
        cols = [pal.shade(base, a) for a in np.sort(rng.uniform(-0.55, 0.3, int(rng.integers(3, 6))))]
        if rng.random() < 0.35:
            cols[int(rng.integers(len(cols)))] = pal.color()      # a second hue bleeding in
        # uneven light: bend the ramp with a smooth random field so the falloff isn't geometric
        bend = cv2.resize(rng.uniform(-0.25, 0.25, (4, 4)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        t = np.clip(t + bend, 0, 1)
    pos = np.linspace(0, 1, len(cols))
    out = np.empty((size, size, 3), np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(t, pos, [col[c] for col in cols])
    return out


def smooth_noise(rng, size, cells, pal):
    """Soft cloudy colour field made of the clip's own palette colours."""
    small = np.float32([[pal.color() for _ in range(cells)] for _ in range(cells)])
    return np.clip(cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC), 0, 255)


def pattern_image(rng, size, pal):
    """Fine regular detail (stripes / checks / dots): the kind of high-frequency content a
    mosaic destroys completely, so the model has to learn not to invent or smear it."""
    a, b = np.float32(pal.color()), np.float32(pal.color())
    period = int(rng.integers(4, 40))
    ys, xs = np.mgrid[0:size, 0:size]
    kind = rng.choice(['stripes', 'checks', 'dots'])
    if kind == 'stripes':
        ang = rng.uniform(0, math.pi)
        m = ((xs * math.cos(ang) + ys * math.sin(ang)) // period) % 2
    elif kind == 'checks':
        m = ((xs // period) + (ys // period)) % 2
    else:
        m = (((xs % period) - period / 2) ** 2 + ((ys % period) - period / 2) ** 2 < (period * 0.3) ** 2)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.6 if pal.style == 'anime' else 1.2)[:, :, None]
    return a * m + b * (1 - m)


# ----------------------------------------------------------------------------- generated clips

class Shape:
    """One moving object. Drawn each frame from its own motion parameters.
    anime: hard edges, dark outlines, flat fill with an optional hard-edged shadow band.
    real:  soft edges, no outline, smoothly shaded fill."""

    def __init__(self, rng, size, pal, anchored=False):
        self.pal = pal
        anime = pal.style == 'anime'
        kinds = ['ellipse', 'polygon', 'blob', 'stroke', 'text', 'rect']
        self.kind = rng.choice(kinds, p=[0.25, 0.25, 0.1, 0.2, 0.1, 0.1] if anime else [0.35, 0.1, 0.35, 0.1, 0.03, 0.07])
        self.pos = rng.uniform(-0.1, 1.1, 2) * size
        self.vel = rng.normal(0, 0.0015, 2) * size           # px per frame
        if anchored:     # stays near the middle, where the mosaic goes, so there is always detail to restore
            self.pos = rng.uniform(0.25, 0.75, 2) * size
            self.vel = rng.normal(0, 0.0004, 2) * size
        self.sway = rng.uniform(0, 0.02, 2) * size
        self.sway_f = rng.uniform(0.02, 0.15, 2)
        self.phase = rng.uniform(0, 2 * math.pi, 2)
        self.radius = rng.uniform(0.03, 0.35) * size
        self.aspect = rng.uniform(0.35, 1.0)
        self.angle = rng.uniform(0, 360)
        self.spin = rng.normal(0, 0.8 if anime else 0.3)      # degrees per frame
        self.pulse = rng.uniform(0, 0.12)
        self.pulse_f = rng.uniform(0.02, 0.12)
        self.color = pal.color()
        self.outline = int(rng.integers(1, 5)) if (anime and rng.random() < 0.85) else 0
        self.outline_color = pal.color('dark')
        self.alpha = 1.0 if rng.random() < (0.9 if anime else 0.55) else float(rng.uniform(0.45, 0.9))
        # anime: a darker band of the same colour, offset to one side (cel shadow)
        self.cel_shadow = pal.shade(self.color, -rng.uniform(0.18, 0.4)) if (anime and rng.random() < 0.5) else None
        self.shadow_dir = rng.normal(0, 1, 2)
        self.shadow_dir = self.shadow_dir / max(1e-6, float(np.linalg.norm(self.shadow_dir))) * rng.uniform(0.15, 0.45)
        # real: every shape is softly shaded; anime: only sometimes a gradient fill
        self.soft = 0.0 if anime else float(rng.uniform(0.6, 2.5) if rng.random() < 0.8 else rng.uniform(0.03, 0.08) * self.radius)
        self.fill = gradient_image(rng, size, _OneColor(pal, self.color)) if (not anime or rng.random() < 0.2) else None
        n = int(rng.integers(3, 9))
        ang = np.sort(rng.uniform(0, 2 * math.pi, n))
        rad = rng.uniform(0.45, 1.0, n)
        self.poly = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)       # unit polygon
        self.stroke = np.cumsum(rng.normal(0, 0.35, (int(rng.integers(3, 8)), 2)), 0)
        self.thickness = int(rng.integers(1, max(2, int(size * 0.03))))
        self.text = ''.join(chr(int(c)) for c in rng.integers(33, 126, int(rng.integers(2, 8))))
        self.font = int(rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX,
                                    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX]))

    def _points(self, unit, centre, r, angle):
        a = math.radians(angle)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts = (unit * [1.0, self.aspect]) @ rot.T * r + centre
        return np.round(pts * 16).astype(np.int32)            # 4 fractional bits -> sub-pixel motion

    def _draw(self, target, col, ocol, centre, r, angle, zoom, outline):
        aa, sh = cv2.LINE_AA, 4
        if self.kind == 'ellipse':
            c = tuple(int(v) for v in np.round(centre * 16))
            axes = (max(1, int(r * 16)), max(1, int(r * self.aspect * 16)))
            cv2.ellipse(target, c, axes, angle, 0, 360, col, -1, aa, sh)
            if outline:
                cv2.ellipse(target, c, axes, angle, 0, 360, ocol, outline, aa, sh)
        elif self.kind in ('polygon', 'blob', 'rect'):
            unit = self.poly if self.kind != 'rect' else np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
            pts = self._points(unit, centre, r, angle)
            cv2.fillPoly(target, [pts], col, aa, sh)
            if outline:
                cv2.polylines(target, [pts], True, ocol, outline, aa, sh)
        elif self.kind == 'stroke':
            pts = self._points(self.stroke - self.stroke.mean(0), centre, r, angle)
            cv2.polylines(target, [pts], False, col, max(1, int(self.thickness * zoom)), aa, sh)
        else:  # text
            scale = max(0.3, r / 40.0)
            cv2.putText(target, self.text, (int(centre[0] - r), int(centre[1])), self.font, scale, col,
                        max(1, int(scale * 1.5)), aa)

    def draw(self, canvas, t, cam):
        size = canvas.shape[0]
        centre = self.pos + self.vel * t + self.sway * np.sin(self.sway_f * t + self.phase)
        centre = (centre - size / 2) * cam['zoom'] + size / 2 + cam['shift']
        r = self.radius * (1 + self.pulse * math.sin(self.pulse_f * t)) * cam['zoom']
        angle = self.angle + self.spin * t
        simple = self.alpha >= 1.0 and self.fill is None and self.cel_shadow is None and self.soft == 0
        if simple:
            self._draw(canvas, self.color, self.outline_color, centre, r, angle, cam['zoom'], self.outline)
            return
        mask = np.zeros(canvas.shape[:2], np.uint8)
        self._draw(mask, 255, 255, centre, r, angle, cam['zoom'], 0)
        bx, by, bw, bh = cv2.boundingRect(mask)
        if bw == 0 or bh == 0:
            return                                # entirely outside the frame
        sigma = max(0.8, self.soft, r * 0.08 if self.kind == 'blob' else 0) if (self.soft or self.kind == 'blob') else 0
        pad = int(3 * sigma) + 2                  # only the shape's own area is blended (much faster than the whole frame)
        x0, y0, x1, y1 = max(0, bx - pad), max(0, by - pad), min(size, bx + bw + pad), min(size, by + bh + pad)
        m = mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        if sigma:
            m = cv2.GaussianBlur(m, (0, 0), sigma)
        paint = self.fill[y0:y1, x0:x1] if self.fill is not None else np.float32(self.color)
        a3 = (m * self.alpha)[:, :, None]
        region = canvas[y0:y1, x0:x1]
        out = region * (1 - a3) + paint * a3
        if self.cel_shadow is not None:           # the part of the shape NOT covered by a shifted copy of itself
            dx, dy = self.shadow_dir * r
            shifted = cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, dy]]), (m.shape[1], m.shape[0]))
            band = (np.clip(m - shifted, 0, 1) > 0.5).astype(np.float32)[:, :, None] * self.alpha
            out = out * (1 - band) + np.float32(self.cel_shadow) * band
        region[:] = out.astype(np.uint8)
        if self.outline:
            self._outline_only(canvas, centre, r, angle)

    def _outline_only(self, canvas, centre, r, angle):
        aa, sh = cv2.LINE_AA, 4
        if self.kind == 'ellipse':
            c = tuple(int(v) for v in np.round(centre * 16))
            cv2.ellipse(canvas, c, (max(1, int(r * 16)), max(1, int(r * self.aspect * 16))), angle, 0, 360,
                        self.outline_color, self.outline, aa, sh)
        elif self.kind in ('polygon', 'rect'):
            unit = self.poly if self.kind != 'rect' else np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
            cv2.polylines(canvas, [self._points(unit, centre, r, angle)], True, self.outline_color, self.outline, aa, sh)


class _OneColor:
    """Palette stand-in that always returns shades around one colour, so a shape's
    gradient fill reads as that object being lit rather than as a rainbow."""

    def __init__(self, pal, bgr):
        self.pal, self.bgr, self.style, self.rng = pal, bgr, pal.style, pal.rng

    def color(self, kind='any'):
        return self.pal.shade(self.bgr, self.rng.uniform(-0.35, 0.2))

    def shade(self, bgr, amount):
        return self.pal.shade(bgr, amount)


def render_generated_clip(rng, n_frames, size, style='anime', skin=0.5):
    """Frames (uint8 BGR, size x size) of one procedural clip in the given style."""
    pal = Palette(rng, style, skin)
    anime = style == 'anime'
    kind = rng.choice(['flat', 'gradient', 'noise', 'pattern'], p=[0.25, 0.45, 0.15, 0.15] if anime else [0.05, 0.55, 0.3, 0.1])
    big = int(size * 1.5)                                     # background larger than the view, so the camera can move over it
    if kind == 'flat':
        bg = np.empty((big, big, 3), np.float32); bg[:] = pal.color()
    elif kind == 'gradient':
        bg = gradient_image(rng, big, pal)
    elif kind == 'noise':
        bg = smooth_noise(rng, big, int(rng.integers(3, 9)), pal)
    else:
        bg = pattern_image(rng, big, pal)
    if not anime and rng.random() < 0.7:
        # fine surface texture (skin, fabric, film): faint mid-frequency brightness variation
        tex = cv2.GaussianBlur(rng.normal(0, 1, (big, big)).astype(np.float32), (0, 0), rng.uniform(0.7, 2.5))
        bg = bg * (1 + (tex / max(1e-6, float(tex.std())) * rng.uniform(0.01, 0.05))[:, :, None])
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    n_shapes = int(rng.integers(5, 18)) if anime else int(rng.integers(4, 12))
    shapes = [Shape(rng, size, pal, anchored=(i % 2 == 1)) for i in range(n_shapes)]
    pan = rng.normal(0, 0.0025, 2) * size                    # camera pan, px per frame
    zoom_rate = rng.normal(0, 0.0012)
    if anime:      # clean digital picture: steady camera, sharp, at most the softness of video scaling
        shake, grain = 0.0, 0.0
        blur = float(rng.uniform(0.3, 0.7)) if rng.random() < 0.4 else 0.0
    else:          # camera footage: handheld shake, lens softness, sensor noise
        shake = rng.uniform(0.2, 1.2) if rng.random() < 0.4 else 0.0
        blur = float(rng.uniform(0.5, 1.6)) if rng.random() < 0.75 else 0.0
        grain = float(rng.uniform(1.0, 6.0)) if rng.random() < 0.8 else 0.0
    margin = (big - size) / 2.0
    frames = []
    for t in range(n_frames):
        cam = {'zoom': float(np.clip(1 + zoom_rate * t, 0.75, 1.35)),
               'shift': pan * t + (rng.normal(0, shake, 2) if shake else 0)}
        off = np.clip(margin - cam['shift'], 0, big - size)   # background slides opposite to the pan
        M = np.float32([[1, 0, -off[0]], [0, 1, -off[1]]])
        canvas = cv2.warpAffine(bg, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for s in shapes:
            s.draw(canvas, t, cam)
        if blur:
            canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
        if grain:
            canvas = np.clip(canvas.astype(np.float32) + rng.normal(0, grain, canvas.shape), 0, 255).astype(np.uint8)
        frames.append(canvas)
    return frames


# ----------------------------------------------------------------------------- masks

def render_masks(rng, n_frames, size):
    """Where training will put the mosaic: one soft-moving region per clip, sized like the
    detector-based datasets (the region spans 1/1.7 .. 1/1.2 of the crop, as the 1.2-1.7x
    margin around a detected area does), white = mosaic."""
    extent = size / rng.uniform(1.2, 1.7)
    half = np.array([extent / 2, extent / 2 * rng.uniform(0.45, 1.0)])
    if rng.random() < 0.5:
        half = half[::-1]
    centre = size / 2 + rng.normal(0, 0.03, 2) * size
    drift = rng.normal(0, 0.0008, 2) * size
    angle, spin = rng.uniform(0, 180), rng.normal(0, 0.15)
    kind = rng.choice(['ellipse', 'rect', 'blob'], p=[0.4, 0.25, 0.35])
    n = int(rng.integers(6, 12))
    ang = np.sort(rng.uniform(0, 2 * math.pi, n)); rad = rng.uniform(0.6, 1.0, n)
    unit = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)
    masks = []
    for t in range(n_frames):
        m = np.zeros((size, size), np.uint8)
        c = centre + drift * t
        a = angle + spin * t
        if kind == 'ellipse':
            cv2.ellipse(m, (int(c[0]), int(c[1])), (int(half[0]), int(half[1])), a, 0, 360, 255, -1)
        else:
            base = unit if kind == 'blob' else np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
            ar = math.radians(a)
            rot = np.array([[math.cos(ar), -math.sin(ar)], [math.sin(ar), math.cos(ar)]])
            pts = np.round((base * half) @ rot.T + c).astype(np.int32)
            cv2.fillPoly(m, [pts], 255)
            if kind == 'blob':                                # round the corners off
                k = max(3, int(size * 0.06)) | 1
                m = cv2.threshold(cv2.GaussianBlur(m, (k, k), 0), 127, 255, cv2.THRESH_BINARY)[1]
        masks.append(m)
    return masks


# ----------------------------------------------------------------------------- crops of real footage

def list_sources(folder):
    vids, imgs = [], []
    for base, _, files in os.walk(folder):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXT:
                vids.append(os.path.join(base, f))
            elif ext in IMAGE_EXT:
                imgs.append(os.path.join(base, f))
    return vids, imgs


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def resize_square(img, size):
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def crop_clip_from_video(rng, path, n_frames, size, min_side, min_sharpness, cut_threshold):
    """n_frames consecutive frames of a random square region, or (None, reason)."""
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total < n_frames + 2 or min(w, h) < min_side:
            return None, 'too short or too small'
        start = int(rng.integers(0, total - n_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        side = int(rng.uniform(max(min_side / min(w, h), 0.4), 1.0) * min(w, h))
        x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
        dx, dy = rng.normal(0, 0.0005, 2) * side              # slow drift of the crop window
        frames, prev_small, native = [], None, side
        for t in range(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                return None, 'could not read frames'
            small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
            if prev_small is not None and float(np.abs(small - prev_small).mean()) > cut_threshold:
                return None, 'scene cut inside the clip'
            prev_small = small
            x = int(np.clip(x0 + dx * t, 0, w - side)); y = int(np.clip(y0 + dy * t, 0, h - side))
            frames.append(resize_square(frame[y:y + side, x:x + side], size))
        if np.mean([sharpness(f) for f in frames[::10]]) < min_sharpness:
            return None, 'blank or blurry region'
        return frames, native
    finally:
        cap.release()


def crop_clip_from_image(rng, path, n_frames, size, min_side, min_sharpness):
    """Slow pan + zoom over a still image, so it has frame-to-frame motion like video."""
    img = imread(path)
    if img is None or min(img.shape[:2]) < min_side:
        return None, 'unreadable or too small'
    h, w = img.shape[:2]
    s0 = rng.uniform(max(min_side / min(w, h), 0.4), 0.95) * min(w, h)
    s1 = float(np.clip(s0 * rng.uniform(0.9, 1.1), min_side, min(w, h)))
    c0 = np.array([rng.uniform(s0 / 2, w - s0 / 2), rng.uniform(s0 / 2, h - s0 / 2)])
    c1 = c0 + rng.normal(0, 0.04, 2) * s0
    native = int(min(s0, s1))
    f = size / min(s0, s1)
    if f < 1:        # shrink once with area averaging (no aliasing); the per-frame warp then works near 1:1
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        s0, s1, c0, c1 = s0 * f, s1 * f, c0 * f, c1 * f
    frames = []
    for t in range(n_frames):
        k = t / max(1, n_frames - 1)
        s = s0 + (s1 - s0) * k
        c = c0 + (c1 - c0) * k
        s = min(s, w, h)
        c = np.clip(c, [s / 2, s / 2], [w - s / 2, h - s / 2])
        # sub-pixel accurate window -> smooth motion instead of jumping whole pixels
        M = np.float32([[size / s, 0, -(c[0] - s / 2) * size / s], [0, size / s, -(c[1] - s / 2) * size / s]])
        frames.append(cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
    if np.mean([sharpness(fr) for fr in frames[::10]]) < min_sharpness:
        return None, 'blank or blurry region'
    return frames, native


# ----------------------------------------------------------------------------- output

def next_index(savedir, prefix):
    used = [int(n[len(prefix):]) for n in os.listdir(savedir)
            if n.startswith(prefix) and n[len(prefix):].isdigit()] if os.path.isdir(savedir) else []
    return max(used) + 1 if used else 0


def save_clip(savedir, name, frames, masks, jpg_quality):
    odir, mdir = os.path.join(savedir, name, 'origin_image'), os.path.join(savedir, name, 'mask')
    os.makedirs(odir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    for i, (f, m) in enumerate(zip(frames, masks), 1):
        imwrite(os.path.join(odir, '%05d.jpg' % i), f, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        imwrite(os.path.join(mdir, '%05d.png' % i), m)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['generated', 'crops'], default='generated')
    p.add_argument('--savedir', default=os.path.join(ROOT, 'datasets', 'synthetic_pool'),
                   help='dataset folder to add clips to (default: datasets/synthetic_pool)')
    p.add_argument('--clips', type=int, default=100, help='number of clips to create')
    p.add_argument('--frames', type=int, default=100,
                   help="frames per clip; must be at least train.py's --M (default 100)")
    p.add_argument('--outsize', type=int, default=572,
                   help='stored frame size in px: 572 for 512px training (the default), 286 for 256px')
    p.add_argument('--style', choices=['mixed', 'anime', 'real'], default='mixed',
                   help='[generated] look of the clips: anime (flat, cel-shaded, outlined), real (smooth shading, '
                        'texture, blur, noise) or mixed (--anime_share of the clips are anime, the rest real)')
    p.add_argument('--anime_share', type=float, default=0.5, help='[generated] with --style mixed: fraction of clips in the anime look')
    p.add_argument('--skin_tone', type=float, default=0.6,
                   help='[generated] 0..1: share of colours taken from beige / pink / tan shades. '
                        '0 = any colour, 1 = only those shades')
    p.add_argument('--source', default=None, help='[crops] folder of clean videos and/or images (searched recursively)')
    p.add_argument('--min_side', type=int, default=0,
                   help='[crops] smallest source region (px) to accept. Default: half of --outsize, so no crop is enlarged more than 2x')
    p.add_argument('--min_sharpness', type=float, default=15.0,
                   help='[crops] skip regions with less detail than this (blank walls, black frames, heavy blur). 0 = keep everything')
    p.add_argument('--cut_threshold', type=float, default=28.0,
                   help='[crops] a jump this large between two frames is treated as a scene cut and the clip is skipped')
    p.add_argument('--jpg_quality', type=int, default=95)
    p.add_argument('--seed', type=int, default=None, help='set for reproducible output')
    p.add_argument('--preview', default=None,
                   help='also save a contact sheet (first/middle/last frame + mask of up to 6 clips) to this image file')
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.savedir, exist_ok=True)
    prefix = 'gen_' if a.mode == 'generated' else 'crop_'
    index = next_index(a.savedir, prefix)
    min_side = a.min_side or a.outsize // 2
    vids = imgs = []
    if a.mode == 'crops':
        if not a.source or not os.path.isdir(a.source):
            sys.exit("--mode crops needs --source <folder of clean videos/images>")
        vids, imgs = list_sources(a.source)
        if not vids and not imgs:
            sys.exit(f"No videos or images found in {a.source}")
        print(f"Source: {len(vids)} video(s), {len(imgs)} image(s)")

    made, skipped, natives, sheet, styles = 0, {}, [], [], {}
    skin = float(np.clip(a.skin_tone, 0, 1))
    attempts, max_attempts = 0, a.clips * 30
    while made < a.clips and attempts < max_attempts:
        attempts += 1
        if a.mode == 'generated':
            style = a.style if a.style != 'mixed' else ('anime' if rng.random() < a.anime_share else 'real')
            styles[style] = styles.get(style, 0) + 1
            frames, native = render_generated_clip(rng, a.frames, a.outsize, style, skin), a.outsize
        else:
            use_video = vids and (not imgs or rng.random() < len(vids) / (len(vids) + len(imgs) * 0.25))
            if use_video:
                frames, native = crop_clip_from_video(rng, vids[int(rng.integers(len(vids)))], a.frames, a.outsize,
                                                      min_side, a.min_sharpness, a.cut_threshold)
            else:
                frames, native = crop_clip_from_image(rng, imgs[int(rng.integers(len(imgs)))], a.frames, a.outsize,
                                                      min_side, a.min_sharpness)
            if frames is None:
                skipped[native] = skipped.get(native, 0) + 1
                continue
            natives.append(native)
        masks = render_masks(rng, a.frames, a.outsize)
        name = '%s%05d' % (prefix, index + made)
        save_clip(a.savedir, name, frames, masks, a.jpg_quality)
        if a.preview and len(sheet) < 6:
            mid = len(frames) // 2
            sheet.append(np.hstack([frames[0], frames[mid], frames[-1], cv2.cvtColor(masks[mid], cv2.COLOR_GRAY2BGR)]))
        made += 1
        print(f"\r{made}/{a.clips} clips", end='', flush=True)
    print()

    if a.preview and sheet:
        img = np.vstack(sheet)
        scale = min(1.0, 1600 / img.shape[1])
        imwrite(a.preview, cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        print(f"Preview saved: {a.preview}")
    print(f"Created {made} clip(s) of {a.frames} frames at {a.outsize}px in {a.savedir} ({prefix}{index:05d} onward).")
    if skipped:
        print("Skipped regions: " + ", ".join(f"{v} x {k}" for k, v in skipped.items()))
    if natives:
        ns = np.array(natives)
        enlarged = 100.0 * float((ns < a.outsize).mean())
        print(f"Source region size: median {int(np.median(ns))}px; {enlarged:.0f}% of clips were smaller than {a.outsize}px and enlarged.")
        if enlarged > 50:
            print("  Most regions had to be enlarged, so this footage does not hold much real detail at this size. "
                  "Higher-resolution sources (or a smaller --outsize / model size) would match better.")
    if made < a.clips:
        print(f"Only {made} of {a.clips} clips could be made: the source ran out of usable regions "
              f"(try --min_sharpness 0, a lower --min_side, or more footage).")
    if a.mode == 'generated':
        print("Styles: " + ", ".join(f"{v} {k}" for k, v in sorted(styles.items())) + f"; skin-tone share {skin:.0%}")
        print("Note: generated clips contain no real texture or anatomy. Use them alongside real footage, not instead of it.")


if __name__ == '__main__':
    main()
