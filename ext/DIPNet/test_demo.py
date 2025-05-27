import os.path
import logging
import torch
import argparse
import json
import glob

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import traceback

from pprint import pprint
from utils.model_summary import get_model_activation, get_model_flops
from utils import utils_logger
from utils import utils_image as util

base = os.path.dirname(__file__)

def prepare_dataset_folders(data_dir, scale_factor=4):
    # Creates train_HR, train_LR, valid_HR, valid_LR folders and populates them
    # with corresponding high-res and low-res versions of the images.

    # Supported extensions - now using PIL which handles more formats
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    
    # Create output directories
    folders = ['train_HR', 'train_LR', 'valid_HR', 'valid_LR']
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created directory: {folder_path}")

    def process_images(source_dir, hr_dest_dir, lr_dest_dir, set_name):
        files = sorted([f for f in os.listdir(source_dir) 
                      if f.lower().endswith(valid_extensions)])
        
        success_count = 0
        skip_count = 0
        failure_count = 0
        failure_examples = []
        
        for filename in tqdm(files, desc=f"Processing {set_name}"):
            try:
                # Check if output files already exist
                base, ext = os.path.splitext(filename)
                lr_filename = f"{base}x{scale_factor}{ext}"
                hr_dest_path = os.path.join(hr_dest_dir, filename)
                lr_dest_path = os.path.join(lr_dest_dir, lr_filename)
                
                if os.path.exists(hr_dest_path) and os.path.exists(lr_dest_path):
                    skip_count += 1
                    continue
                
                # Read HR image using PIL (more reliable than OpenCV)
                hr_path = os.path.join(source_dir, filename)
                with Image.open(hr_path) as img:
                    hr_img = np.array(img)
                    if len(hr_img.shape) == 2:  # Convert grayscale to RGB
                        hr_img = np.stack([hr_img]*3, axis=-1)
                    
                    # Verify image is valid
                    if hr_img.size == 0:
                        raise ValueError("Empty image data")
                
                # Save HR image
                Image.fromarray(hr_img).save(hr_dest_path)
                
                # Create LR image
                h, w = hr_img.shape[:2]
                lr_img = cv2.resize(
                    hr_img, 
                    (max(4, w//scale_factor), max(4, h//scale_factor)),  # Minimum 4px to avoid too small images
                    interpolation=cv2.INTER_AREA  # Better for downscaling
                )
                
                # Save LR image
                Image.fromarray(lr_img).save(lr_dest_path)
                
                # Verify LR image was created
                if not os.path.exists(lr_dest_path):
                    raise FileNotFoundError("LR image not created")
                
                success_count += 1
                
            except Exception as e:
                failure_count += 1
                failure_examples.append((filename, str(e)))
                print(f"\nError processing {filename}: {str(e)}")
                continue
        
        # Print summary
        print(f"\n{set_name} processing complete!")
        print(f"Successfully processed: {success_count} images")
        print(f"Skipped existing: {skip_count} images")
        print(f"Failed to process: {failure_count} images")
        if failure_count > 0:
            print("\nExample failures:")
            for i, (fname, error) in enumerate(failure_examples[:3], 1):
                print(f"{i}. {fname}: {error}")
            if failure_count > 3:
                print(f"... plus {failure_count-3} more")

    # Process training set
    train_hr_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_hr_dir):
        raise FileNotFoundError(f"Training directory not found: {train_hr_dir}")
    
    process_images(
        train_hr_dir,
        os.path.join(data_dir, 'train_HR'),
        os.path.join(data_dir, 'train_LR'),
        "Training Set"
    )

    # Process validation set with extra verification
    valid_hr_dir = os.path.join(data_dir, 'valid')
    if not os.path.exists(valid_hr_dir):
        raise FileNotFoundError(f"Validation directory not found: {valid_hr_dir}")
    
    process_images(
        valid_hr_dir,
        os.path.join(data_dir, 'valid_HR'),
        os.path.join(data_dir, 'valid_LR'),
        "Validation Set"
    )

    # Verify only the existing LR files
    print("\nVerifying existing LR files...")
    actual_lr_files = sorted([
        f for f in os.listdir(os.path.join(data_dir, 'valid_LR'))
        if f.endswith('.png') and 'x4' in f
    ])
    print(f"Found {len(actual_lr_files)} LR files in validation set")

    print("\nDataset preparation complete!")

def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        from models.DIPNet import DIPNet
        name, data_range = f"{model_id:02}_DIPNet", 1.0
        model_path = os.path.join(base, 'model_zoo', 'DIPNet.pth')
        model = DIPNet()
        model.load_state_dict(torch.load(model_path), strict=True)
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model, name, data_range, tile


def select_dataset(data_dir, mode):
    if mode == "test":
        path = [
            (
                os.path.join(data_dir, f"test_LR/{i:04}.png"),
                os.path.join(data_dir, f"test_HR/{i:04}.png")
            ) for i in range(901, 1001)
        ]
    elif mode == "valid":
        # Get actual validation files that exist
        valid_lr_dir = os.path.join(data_dir, "valid_LR")
        valid_hr_dir = os.path.join(data_dir, "valid_HR")
        
        # Find all LR files and match them with HR files
        lr_files = sorted([
            f for f in os.listdir(valid_lr_dir) 
            if f.endswith('.png') and 'x4' in f
        ])
        
        path = []
        for lr_file in lr_files:
            # Extract base number (e.g., "0801" from "0801x4.png")
            base_num = lr_file.split('x')[0]
            hr_file = f"{base_num}.png"
            
            lr_path = os.path.join(valid_lr_dir, lr_file)
            hr_path = os.path.join(valid_hr_dir, hr_file)
            
            if os.path.exists(hr_path):
                path.append((lr_path, hr_path))
            else:
                print(f"Warning: Missing HR counterpart for {lr_file}")

        print(f"Found {len(path)} valid image pairs for validation")
        
    elif mode == "hybrid_test":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "LSDIR_DIV2K_test_HR/*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    return path


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

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


def run(model, model_name, data_range, tile, logger, device, args, mode="test"):
    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []
    # results[f"{mode}_psnr_y"] = []
    # results[f"{mode}_ssim_y"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_lr, img_hr) in enumerate(data_path):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        start.record()
        img_sr = forward(img_lr, model, tile)
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        img_sr = util.tensor2uint(img_sr, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        # print(img_sr.shape, img_hr.shape)
        psnr = util.calculate_psnr(img_sr, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_sr, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        # if np.ndim(img_hr) == 3:  # RGB image
        #     img_sr_y = util.rgb2ycbcr(img_sr, only_y=True)
        #     img_hr_y = util.rgb2ycbcr(img_hr, only_y=True)
        #     psnr_y = util.calculate_psnr(img_sr_y, img_hr_y, border=border)
        #     ssim_y = util.calculate_ssim(img_sr_y, img_hr_y, border=border)
        #     results[f"{mode}_psnr_y"].append(psnr_y)
        #     results[f"{mode}_ssim_y"].append(ssim_y)
        # print(os.path.join(save_path, img_name+ext))
        util.imsave(img_sr, os.path.join(save_path, img_name+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"]) #/ 1000.0
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memery", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} seconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))
    print(results[f"{mode}_ave_psnr"])
    return results


def main(args):
    # Prepare dataset folders first
    prepare_dataset_folders(args.data_dir)

    utils_logger.logger_info("NTIRE2023-EfficientSR", log_path="NTIRE2023-EfficientSR.log")
    logger = logging.getLogger("NTIRE2023-EfficientSR")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(args.save_dir, "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        if args.hybrid_test:
            # inference on the DIV2K and LSDIR test set
            valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="hybrid_test")
            # record PSNR, runtime
            results[model_name] = valid_results
        else:
            # inference on the validation set
            valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
            # record PSNR, runtime
            results[model_name] = valid_results

            if args.include_test:
                # inference on the test set
                test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
                results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # set the input dimension
        activations, num_conv = get_model_activation(model, input_dim)
        activations = activations/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        flops = get_model_flops(model, input_dim, False)
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        # print(v.keys())
        if args.hybrid_test:
            val_psnr = f"{v['hybrid_test_ave_psnr']:2.2f}"
            val_time = f"{v['hybrid_test_ave_runtime']:3.2f}"
            mem = f"{v['hybrid_test_memory']:2.2f}"
        else:
            val_psnr = f"{v['valid_ave_psnr']:2.2f}"
            val_time = f"{v['valid_ave_runtime']:3.2f}"
            mem = f"{v['valid_memory']:2.2f}"
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(args.save_dir, 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2023-EfficientSR")
    parser.add_argument("--data_dir", default="/cluster/work/cvl/yawli/data/NTIRE2023_Challenge", type=str)
    parser.add_argument("--save_dir", default="/cluster/work/cvl/yawli/data/NTIRE2023_Challenge/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K test set")
    parser.add_argument("--hybrid_test", action="store_true", help="Hybrid test on DIV2K and LSDIR test set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")

    args = parser.parse_args()
    pprint(args)

    main(args)
