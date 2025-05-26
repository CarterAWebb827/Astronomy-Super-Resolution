import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util

# Set memory optimization flags
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cuda.matmul.allow_tf32 = True

base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                    'gray_dn, color_dn, jpeg_car, color_jpeg_car')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                    'Just used to differentiate two different settings in Table 2 of the paper. '
                                    'Images are NOT tested patch by patch.')
parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
parser.add_argument('--model_path', type=str, default=f'{base}/models/swinIR/')
parser.add_argument('--model_name', type=str, 
                    default='003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR-with-dict-keys-params-and-params_ema.pth')
parser.add_argument('--dir_name', type=str, default=None)
parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
args = parser.parse_args()

MODEL_PATH = os.path.join(args.model_path, args.model_name)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    if os.path.exists(MODEL_PATH):
        print(f'loading model from {MODEL_PATH}')
    else:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(MODEL_PATH))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {MODEL_PATH}')
        open(MODEL_PATH, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    if args.dir_name is not None:
        save_dir = os.path.join(save_dir, args.dir_name)
    os.makedirs(save_dir, exist_ok=True)

    output_suffix = "_SwinIR.png"

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # Get the base filename without extension
        imgname = os.path.splitext(os.path.basename(path))[0]

        # Check if output file already exists
        output_path = os.path.join(save_dir, f"{imgname}{output_suffix}")
        if os.path.exists(output_path):
            # print(f"Skipping {imgname} - output already exists at {output_path}")
            continue

        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        # Try processing normally first
        try:
            print(f"Processing {imgname} normally...")
            with torch.no_grad():
                output = process_image(img_lq, model, args, window_size, tile_size=args.tile, overlap=args.tile_overlap)
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"Memory error: {e}")
                print("Falling back to tiled processing with reduced tile size...")
                
                # Determine optimal tile size
                _, _, h, w = img_lq.shape
                tile_size = determine_optimal_tile_size(h, w, window_size, args.min_tile_size)
                
                with torch.no_grad():
                    output = process_image(img_lq, model, args, window_size, tile_size=tile_size, overlap=args.tile_overlap)
            else:
                raise e

        # Save output
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

        # Evaluate if ground truth available
        if img_gt is not None:
            evaluate_results(output, img_gt, imgname, idx, test_results, border, args)

    # Summarize results
    summarize_results(test_results, save_dir, args)

def process_image(img_lq, model, args, window_size, tile_size=None, overlap=32):
    # Process image either as whole or in tiles
    _, _, h_old, w_old = img_lq.size()
    
    # Pad image to be multiple of window_size
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

    if tile_size is None:
        # Process whole image
        output = model(img_lq)
    else:
        # Process in tiles
        output = process_tiled_image(img_lq, model, args.scale, tile_size, overlap)
    
    return output[..., :h_old * args.scale, :w_old * args.scale]

def process_tiled_image(img_lq, model, scale, tile_size, overlap):
    # Process image in tiles with overlap
    b, c, h, w = img_lq.size()
    stride = tile_size - overlap
    
    h_idx_list = list(range(0, h-tile_size, stride)) + [h-tile_size]
    w_idx_list = list(range(0, w-tile_size, stride)) + [w-tile_size]
    
    E = torch.zeros(b, c, h*scale, w*scale).type_as(img_lq)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size]
            out_patch = model(in_patch)
            
            # Apply window function for smooth blending
            window = torch.hann_window(tile_size, device=img_lq.device)
            window = window.unsqueeze(0) * window.unsqueeze(1)
            window = window.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            
            # Accumulate results with window weighting
            E[..., h_idx*scale:(h_idx+tile_size)*scale, 
                w_idx*scale:(w_idx+tile_size)*scale].add_(out_patch * window)
            W[..., h_idx*scale:(h_idx+tile_size)*scale, 
                w_idx*scale:(w_idx+tile_size)*scale].add_(window)
            
            # Clean up
            del in_patch, out_patch
            torch.cuda.empty_cache()
    
    return E.div_(W)

def determine_optimal_tile_size(h, w, window_size, min_tile_size=64):
    # Determine optimal tile size based on image dimensions
    tile_size = min(h, w, 512)  # Start with reasonable size
    tile_size = max(tile_size, min_tile_size)
    
    # Ensure tile_size is multiple of window_size
    tile_size = ((tile_size + window_size - 1) // window_size) * window_size
    
    return tile_size

def evaluate_results(output, img_gt, imgname, idx, test_results, border, args):
    # Evaluate and store PSNR/SSIM results
    img_gt = (img_gt * 255.0).round().astype(np.uint8)
    img_gt = img_gt[:output.shape[0] * args.scale, :output.shape[1] * args.scale, ...]
    img_gt = np.squeeze(img_gt)

    psnr = util.calculate_psnr(output, img_gt, crop_border=border)
    ssim = util.calculate_ssim(output, img_gt, crop_border=border)
    test_results['psnr'].append(psnr)
    test_results['ssim'].append(ssim)
    
    if img_gt.ndim == 3:  # RGB image
        psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
        ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
        test_results['psnr_y'].append(psnr_y)
        test_results['ssim_y'].append(ssim_y)
    
    print(f'Testing {idx} {imgname:20s} - PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}; PSNR_Y: {psnr_y:.2f} dB; SSIM_Y: {ssim_y:.4f}')

def summarize_results(test_results, save_dir, args):
    # Print summary of evaluation metrics
    if test_results['psnr']:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print(f'\n{save_dir} \n-- Average PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}')
        
        if test_results['psnr_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print(f'-- Average PSNR_Y/SSIM_Y: {ave_psnr_y:.2f} dB; {ave_ssim_y:.4f}')

def define_model(args):
    if not args.large_model:
        # use 'nearest+conv' to avoid block artifacts
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    else:
        # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'

    pretrained_model = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model

def setup(args):
    # save_dir = f'results/swinir_{args.task}_x{args.scale}'
    save_dir = f'output/swinir/{args.task}_x{args.scale}'
    if args.large_model:
        save_dir += '_large'
    folder = args.folder_lq
    border = 0
    window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_gt = None
    img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
