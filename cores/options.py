import argparse
import os
import sys


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        #base
        self.parser.add_argument('--version', action='version', version='DeepMosaicsPlus 1.2.0 (fork of DeepMosaics 0.5.1)')
        self.parser.add_argument('--debug', action='store_true', help='if specified, start debug mode')
        self.parser.add_argument('--gpu_id', type=str,default='0', help='if -1, use cpu')
        self.parser.add_argument('--media_path', type=str, default='./imgs/ruoruo.jpg',help='your videos or images path')
        self.parser.add_argument('-ss', '--start_time', type=str, default='00:00:00',help='start position of video, default is the beginning of video')
        self.parser.add_argument('-t', '--last_time', type=str, default='00:00:00',help='duration of the video, default is the entire video')
        self.parser.add_argument('--mode', type=str, default='auto',help='Program running mode. auto | add | clean | style')
        self.parser.add_argument('--model_path', type=str, default='./pretrained_models/mosaic/add_face.pth',help='pretrained model path')
        self.parser.add_argument('--result_dir', type=str, default='./result',help='output media will be saved here')
        self.parser.add_argument('--temp_dir', type=str, default='./tmp', help='Temporary files will go here')
        self.parser.add_argument('--tempimage_type', type=str, default='jpg',help='type of temp image, png | jpg, png is better but occupy more storage space')
        self.parser.add_argument('--netG', type=str, default='auto',
            help='select model to use for netG(Clean mosaic and Transfer style) -> auto | unet_128 | unet_256 | resnet_9blocks | HD | video')
        self.parser.add_argument('--fps', type=int, default=0,help='read and output fps, if 0-> origin')
        self.parser.add_argument('--no_preview', action='store_true', help='if specified,do not preview images when processing video. eg.(when run it on server)')
        self.parser.add_argument('--output_size', type=int, default=0,help='size of output media, if 0 -> origin')
        self.parser.add_argument('--mask_threshold', type=int, default=48,help='Mosaic detection threshold (0~255). The smaller is it, the more likely judged as a mosaic area.')

        #AddMosaic
        self.parser.add_argument('--mosaic_mod', type=str, default='squa_avg',help='type of mosaic -> squa_avg | squa_random | squa_avg_circle_edge | rect_avg | random')
        self.parser.add_argument('--mosaic_size', type=int, default=0,help='mosaic size,if 0 auto size')
        self.parser.add_argument('--mask_extend', type=int, default=10,help='extend mosaic area')
        
        #CleanMosaic     
        self.parser.add_argument('--mosaic_position_model_path', type=str, default='auto',help='name of model use to find mosaic position')
        self.parser.add_argument('--traditional', action='store_true', help='if specified, use traditional image processing methods to clean mosaic')
        self.parser.add_argument('--tr_blur', type=int, default=10, help='ksize of blur when using traditional method, it will affect final quality')
        self.parser.add_argument('--tr_down', type=int, default=10, help='downsample when using traditional method,it will affect final quality')
        self.parser.add_argument('--no_feather', action='store_true', help='if specified, no edge feather and color correction, but run faster')
        self.parser.add_argument('--gif_to_mp4', action='store_true', help='save the result of a GIF input as an MP4 (using --encode_vcodec/--encode_crf) instead of converting it back to a GIF')
        self.parser.add_argument('--keep_temp', action='store_true', help='if specified, keep the temp frames after a finished run (so a UI can keep previewing them); only the resume marker is removed. The next run clears them as usual')
        self.parser.add_argument('--keep_frames', action='store_true', help='if specified, do not delete source frames from video2image dir during cleaning (useful for UI seeking)')
        self.parser.add_argument('--encode_crf', type=int, default=None,
            help='CRF/CQ quality for output video encode. If unset, uses the selected '
                 'codec\'s own default (h264=18, hevc=22, av1=30, vp9=31, nvenc=19). '
                 'Note CRF scales differ per codec: x264/x265/nvenc are 0-51, av1/vp9 are 0-63.')
        self.parser.add_argument('--encode_vcodec', type=str, default='h264',
            help='video codec for output encode -> h264 | hevc(h265) | av1 | vp9 | h264_nvenc | hevc_nvenc')
        self.parser.add_argument('--decode_qv', type=int, default=1, help='JPEG quality for extracted frames (1=best, 31=worst, default 1)')
        self.parser.add_argument('--luma_sharpen', action='store_true',
            help='apply luma-channel unsharp-mask to cleaned region')
        self.parser.add_argument('--luma_sharpen_amount', type=float, default=1.0,
            help='luma sharpening strength (0.5=mild, 1.0=normal, 2.0=strong)')
        self.parser.add_argument('--bilateral_sharpen', action='store_true',
            help='edge-preserving bilateral sharpening on cleaned region')
        self.parser.add_argument('--bilateral_sharpen_amount', type=float, default=0.5,
            help='bilateral sharpening strength (0.2=subtle, 0.5=moderate, 1.0=strong)')
        self.parser.add_argument('--freq_inject', action='store_true',
            help='inject high-frequency edges from original mosaic into cleaned patch')
        self.parser.add_argument('--freq_inject_amount', type=float, default=0.3,
            help='frequency injection blend strength (0.1=subtle, 0.3=moderate, 0.6=strong)')
        self.parser.add_argument('--all_mosaic_area', action='store_true', help='if specified, find all mosaic area, else only find the largest area')
        self.parser.add_argument('--min_mosaic_area', type=int, default=300,
            help='minimum connected component area (pixels) to keep in mosaic mask. '
                 'Lower values detect smaller mosaic regions. Default 300.')
        self.parser.add_argument('--min_mosaic_size', type=int, default=100,
            help='minimum bounding-box half-size (pixels) to attempt cleaning. '
                 'Lower values process smaller detected regions. Default 100.')
        self.parser.add_argument('--medfilt_num', type=int, default=5,help='medfilt window of mosaic movement in the video')
        self.parser.add_argument('--ex_mult', type=str, default='auto',help='mosaic area expansion')

        #Auto model selection by detected mosaic size (experimental, opt-in)
        self.parser.add_argument('--auto_model', action='store_true',
            help='EXPERIMENTAL: detect the mosaic block size (as %% of frame short side) and '
                 'automatically pick the matching pretrained model from --model_manifest, instead '
                 'of always using --model_path. AI clean mode only (ignored with --traditional). '
                 'Falls back to --model_path on a low-confidence detection or any error.')
        self.parser.add_argument('--model_manifest', type=str,
            default=os.path.join('.', 'pretrained_models', 'mosaic', 'size_manifest.json'),
            help='JSON manifest mapping mosaic-size %% buckets to model checkpoint paths, '
                 'used when --auto_model is set. See pretrained_models/mosaic/README.md.')
        self.parser.add_argument('--auto_model_min_confidence', type=float, default=0.15,
            help='minimum detector confidence (0-1) required before --auto_model swaps the '
                 'model; below this, --model_path is used unchanged.')
        self.parser.add_argument('--auto_model_samples', type=int, default=5,
            help='number of frames to sample (video only) when estimating mosaic size for '
                 '--auto_model. More samples = more robust to per-frame noise, slower startup.')

        #StyleTransfer
        self.parser.add_argument('--preprocess', type=str, default='resize', help='resize and cropping of images at load time [ resize | resize_scale_width | edges | gray] or resize,edges(use comma to split)')
        self.parser.add_argument('--edges', action='store_true', help='if specified, use edges to generate pictures,(input_nc = 1)')  
        self.parser.add_argument('--canny', type=int, default=150,help='threshold of canny')
        self.parser.add_argument('--only_edges', action='store_true', help='if specified, output media will be edges')

        self.initialized = True


    def getparse(self, test_flag = False):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        model_name = os.path.basename(self.opt.model_path)
        self.opt.temp_dir = os.path.join(self.opt.temp_dir, 'DeepMosaics_temp')

        # Keep the user's choice: the reset below turns gpu_id into '-1' whenever
        # CUDA is unavailable, which also erased it for DirectML users.
        self.opt.requested_gpu_id = self.opt.gpu_id
        if self.opt.gpu_id != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.opt.gpu_id)
            import torch
            if not torch.cuda.is_available():
                self.opt.gpu_id = '-1'
        # else:
        #     self.opt.gpu_id = '-1'

        if test_flag:
            if not os.path.exists(self.opt.media_path):
                print('Error: Media does not exist!')
                input('Please press any key to exit.\n')
                sys.exit(0)
            if not os.path.exists(self.opt.model_path):
                print('Error: Model does not exist!')
                input('Please press any key to exit.\n')
                sys.exit(0)

            if self.opt.mode == 'auto':
                if 'clean' in model_name or self.opt.traditional:
                    self.opt.mode = 'clean'
                elif 'add' in model_name:
                    self.opt.mode = 'add'
                elif 'style' in model_name or 'edges' in model_name:
                    self.opt.mode = 'style'
                else:
                    print('Please check model_path!')
                    input('Please press any key to exit.\n')
                    sys.exit(0)

            if self.opt.output_size == 0 and self.opt.mode == 'style':
                self.opt.output_size = 512

            if 'edges' in model_name or 'edges' in self.opt.preprocess:
                self.opt.edges = True

            if self.opt.netG == 'auto' and self.opt.mode =='clean':
                if 'unet_128' in model_name:
                    self.opt.netG = 'unet_128'
                elif 'resnet_9blocks' in model_name:
                    self.opt.netG = 'resnet_9blocks'
                elif 'HD' in model_name and 'video' not in model_name:
                    self.opt.netG = 'HD'
                elif 'video' in model_name:
                    self.opt.netG = 'video'
                else:
                    print(f"Error: could not auto-detect the generator architecture from --model_path "
                          f"'{self.opt.model_path}'.")
                    print("This project infers the architecture from the checkpoint's filename, expecting it "
                          "to contain one of: unet_128, resnet_9blocks, HD, video.")
                    print("If --model_path is actually your mosaic-position/detection model, that goes in "
                          "--mosaic_position_model_path instead — --model_path must be a clean/generator model.")
                    print("If your checkpoint's filename just doesn't follow this convention, pass the "
                          "architecture explicitly, e.g.: --netG unet_128  (or resnet_9blocks / HD / video)")
                    input('Please press any key to exit.\n')
                    sys.exit(0)

            if self.opt.ex_mult == 'auto':
                if 'face' in model_name:
                    self.opt.ex_mult = 1.1
                else:
                    self.opt.ex_mult = 1.5
            else:
                self.opt.ex_mult = float(self.opt.ex_mult)

            if self.opt.mosaic_position_model_path == 'auto' and self.opt.mode == 'clean':
                _path = os.path.join(os.path.split(self.opt.model_path)[0],'mosaic_position.pth')
                if os.path.isfile(_path):
                    self.opt.mosaic_position_model_path = _path
                else:
                    input('Please check mosaic_position_model_path!')
                    input('Please press any key to exit.\n')
                    sys.exit(0)

        return self.opt