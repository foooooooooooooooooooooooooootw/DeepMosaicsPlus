import os
import sys
import time
import glob
import traceback
def _startup():
    """Everything that should happen ONCE, in the main process only.
    This used to run at module level. On Windows, multiprocessing starts
    each frame-extraction worker by re-running this file from the top
    (spawn), so every worker repeated it: imported torch/DirectML, parsed
    options, initialised the GPU device and probed ffmpeg -- the source of
    the repeated 'DirectML available' / '[device] Inference' /
    '[ffmpeg] Hardware decode' lines, plus the memory and startup time
    each worker wasted on it. Workers now only load what they need."""
    global init, Options, add, clean, style, util, loadmodel, opt, model_util, ffmpeg_util
    from cores import init

    try:
        from cores import Options,add,clean,style
        from util import util
        from models import loadmodel
    except Exception as e:
        print(e)
        input('Please press any key to exit.\n')
        sys.exit(0)

    opt = Options().getparse(test_flag = True)
    from models import model_util
    model_util.resolve_inference_device(opt)

    # Fail fast with a clear, actionable message if ffmpeg/ffprobe are missing,
    # rather than loading models (slow, and on GPU it reserves memory) only to
    # hit a bare OSError/WinError deep inside video processing. Only checked when
    # the input actually contains video: image-only runs never call ffmpeg, so
    # they neither require it nor pay for probing it.
    from util import ffmpeg as ffmpeg_util
    _inputs = util.Traversal(opt.media_path) if os.path.isdir(opt.media_path) else [opt.media_path]
    if any(util.is_video(f) for f in _inputs):
        try:
            ffmpeg_util.check_ffmpeg_available(raise_on_missing=True)
        except FileNotFoundError as e:
            print(e)
            input('Please press any key to exit.\n')
            sys.exit(0)

    if not os.path.isdir(opt.temp_dir):
        util.file_init(opt)

_user_frame_settings = None

def _apply_gif_settings(src):
    """Per-file frame/encode settings. Called at the start of every file.
    GIF inputs always get PNG temp frames (lossless extraction; JPEG adds
    ringing on the hard flat-colour edges GIFs are made of). If the result is
    going back to a GIF, the intermediate video is lossless RGB too -- JPEG +
    standard H.264 left an average error of 2.6/255 on a round trip, PNG +
    lossless is exact. With --gif_to_mp4 the MP4 IS the result, so it uses the
    user's codec/CRF (a normal, widely playable video) instead. Other files
    keep the user's --tempimage_type / --encode_vcodec / --encode_crf, so a
    mixed folder of GIFs and videos is handled per file."""
    global _user_frame_settings
    if _user_frame_settings is None:
        _user_frame_settings = (opt.tempimage_type, opt.encode_vcodec, opt.encode_crf)
    ffmpeg_util.VIDEO_OUTPUTS.clear()          # outputs are tracked per input file
    user_type, user_codec, user_crf = _user_frame_settings
    if os.path.splitext(src)[1].lower() == '.gif':
        if getattr(opt, 'gif_to_mp4', False):
            opt.tempimage_type, opt.encode_vcodec, opt.encode_crf = 'png', user_codec, user_crf
        else:
            opt.tempimage_type, opt.encode_vcodec, opt.encode_crf = 'png', 'rgb_lossless', 0
    else:
        opt.tempimage_type, opt.encode_vcodec, opt.encode_crf = user_type, user_codec, user_crf

def _gif_postprocess(src, started):
    """GIF inputs run through the video pipeline, which writes an .mp4.
    Convert exactly the file(s) the encoder wrote for this input back to a GIF
    and remove the intermediate .mp4. (This used to rebuild the .mp4 name from
    the input's name, which broke whenever the output name was sanitised.)"""
    if os.path.splitext(src)[1].lower() != '.gif' or getattr(opt, 'gif_to_mp4', False):
        return
    for mp4 in list(ffmpeg_util.VIDEO_OUTPUTS):
        if not os.path.isfile(mp4):
            continue
        gif = os.path.splitext(mp4)[0] + '.gif'
        print('Converting result to GIF...')
        ffmpeg_util.video_to_gif(mp4, gif)
        os.remove(mp4)
        print('GIF result:', gif)

def main():
    
    if os.path.isdir(opt.media_path):
        files = util.Traversal(opt.media_path)
    else:
        files = [opt.media_path]        
    if opt.mode == 'add':
        netS = loadmodel.bisenet(opt,'roi')
        for file in files:
            opt.media_path = file
            _started = time.time()
            _apply_gif_settings(file)
            if util.is_img(file):
                add.addmosaic_img(opt,netS)
            elif util.is_video(file):
                add.addmosaic_video(opt,netS)
                _gif_postprocess(file, _started)
                util.clean_tempfiles(opt, tmp_init = False)
            else:
                print('This type of file is not supported')
            util.clean_tempfiles(opt, tmp_init = False)

    elif opt.mode == 'clean':
        netM = loadmodel.bisenet(opt,'mosaic')
        if opt.traditional:
            netG = None
        elif opt.netG == 'video':
            netG = loadmodel.video(opt)
        else:
            netG = loadmodel.pix2pix(opt)
        
        for file in files:
            opt.media_path = file
            _started = time.time()
            _apply_gif_settings(file)
            if util.is_img(file):
                clean.cleanmosaic_img(opt,netG,netM)
            elif util.is_video(file):
                if opt.netG == 'video' and not opt.traditional:            
                    clean.cleanmosaic_video_fusion(opt,netG,netM)
                else:
                    clean.cleanmosaic_video_byframe(opt,netG,netM)
                _gif_postprocess(file, _started)
                util.clean_tempfiles(opt, tmp_init = False)
            else:
                print('This type of file is not supported')

    elif opt.mode == 'style':
        netG = loadmodel.style(opt)
        for file in files:
            opt.media_path = file
            _started = time.time()
            _apply_gif_settings(file)
            if util.is_img(file):
                style.styletransfer_img(opt,netG)
            elif util.is_video(file):
                style.styletransfer_video(opt,netG)
                _gif_postprocess(file, _started)
                util.clean_tempfiles(opt, tmp_init = False)
            else:
                print('This type of file is not supported')

    util.clean_tempfiles(opt, tmp_init = False)

if __name__ == '__main__':
    _startup()
    if opt.debug:
        main()
        sys.exit(0)
    try:
        main()
        print('Finished!')
    except Exception as ex:
        print('--------------------ERROR--------------------')
        print('--------------Environment--------------')
        print('DeepMosaicsPlus: 1.2.0 (fork of DeepMosaics 0.5.1)')
        print('Python:',sys.version)
        import torch
        print('Pytorch:',torch.__version__)
        import cv2
        print('OpenCV:',cv2.__version__)
        import platform
        print('Platform:',platform.platform())
        _ff_ok, _ff_ver, _fp_ok = ffmpeg_util.ffmpeg_status()
        print('ffmpeg:', _ff_ver if _ff_ok else 'NOT FOUND on PATH')
        print('ffprobe:', 'found' if _fp_ok else 'NOT FOUND on PATH')

        print('--------------BUG--------------')
        ex_type, ex_val, ex_stack = sys.exc_info()
        print('Error Type:',ex_type)
        print(ex_val)
        for stack in traceback.extract_tb(ex_stack):
            print(stack)

        # Common-cause hint: a FileNotFoundError raised from subprocess.run()/Popen()
        # has .filename set to the executable it tried and failed to launch (e.g.
        # 'ffmpeg'). Surfacing that name turns a cryptic "[WinError 2] The system
        # cannot find the file specified" into something a user can actually act on.
        missing_exe = getattr(ex, 'filename', None)
        if isinstance(ex, FileNotFoundError) and missing_exe:
            print('--------------LIKELY CAUSE--------------')
            print(f"Windows/OS could not find and launch '{missing_exe}'.")
            if missing_exe.lower().startswith('ffmpeg') or missing_exe.lower().startswith('ffprobe'):
                print("This almost always means ffmpeg is not installed, or its folder isn't on your PATH.")
                print("Download it from https://ffmpeg.org/download.html, add its 'bin' folder to PATH,")
                print("then close and reopen your terminal before trying again.")
            else:
                print(f"Check that '{missing_exe}' is installed and reachable on your system PATH.")

        input('Please press any key to exit.\n')
        #util.clean_tempfiles(tmp_init = False)
        sys.exit(0)