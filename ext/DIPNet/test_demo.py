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
    folders = ['train_HR', 'train_LR', 'valid_HR', 'valid_LR', 'test_HR', 'test_LR']
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
                
                # # Create LR image
                # h, w = hr_img.shape[:2]
                # lr_img = cv2.resize(
                #     hr_img, 
                #     (max(4, w//scale_factor), max(4, h//scale_factor)),  # Minimum 4px to avoid too small images
                #     interpolation=cv2.INTER_AREA  # Better for downscaling
                # )
                
                # # Save LR image
                # Image.fromarray(lr_img).save(lr_dest_path)

                # Create LR image with proper normalization
                h, w = hr_img.shape[:2]
                # lr_img = cv2.resize(
                #     hr_img.astype(np.float32)/255.0,  # Convert to [0,1] range first
                #     (max(32, w//scale_factor), max(32, h//scale_factor)),  # Increased minimum size
                #     interpolation=cv2.INTER_CUBIC  # Better for quality
                # )
                # Use bicubic for downscaling, then bicubic for loading
                lr_img = cv2.resize(hr_img, (w//scale_factor, h//scale_factor), 
                                interpolation=cv2.INTER_AREA)  # Better for downscaling
                
                # Convert back to 8-bit and save
                lr_img = (lr_img * 255).clip(0, 255).astype(np.uint8)
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

    # Process validation set
    valid_hr_dir = os.path.join(data_dir, 'valid')
    if not os.path.exists(valid_hr_dir):
        raise FileNotFoundError(f"Validation directory not found: {valid_hr_dir}")
    
    process_images(
        valid_hr_dir,
        os.path.join(data_dir, 'valid_HR'),
        os.path.join(data_dir, 'valid_LR'),
        "Validation Set"
    )

    # Process test set with
    test_hr_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_hr_dir):
        raise FileNotFoundError(f"Validation directory not found: {test_hr_dir}")
    
    process_images(
        test_hr_dir,
        os.path.join(data_dir, 'test_HR'),
        os.path.join(data_dir, 'test_LR'),
        "Test Set"
    )

    print("\nVerifying existing LR files...")
    valid_lr_dir = os.path.join(data_dir, 'valid_LR')
    
    # Find all LR files using multiple patterns
    lr_patterns = [f"*x{scale_factor}.png", f"*x{scale_factor}.jpg", f"*x{scale_factor}.jpeg"]
    actual_lr_files = []
    
    for pattern in lr_patterns:
        actual_lr_files.extend(glob.glob(os.path.join(valid_lr_dir, pattern)))
    
    # Also check for files that might not have the x4 suffix but are in LR dir
    # additional_lr_files = [f for f in os.listdir(valid_lr_dir) 
    #                       if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
    #                       f not in actual_lr_files]
    # actual_lr_files.extend([os.path.join(valid_lr_dir, f) for f in additional_lr_files])
    
    # Get just the filenames for display
    lr_filenames = [os.path.basename(f) for f in actual_lr_files]
    
    if lr_filenames:
        print(f"Found {len(lr_filenames)} LR files in validation set:")
        for i, fname in enumerate(lr_filenames[:5], 1):
            print(f"{i}. {fname}")
        if len(lr_filenames) > 5:
            print(f"... plus {len(lr_filenames)-5} more")
        
        # Verify corresponding HR files exist
        missing_hr = []
        for lr_file in actual_lr_files:
            lr_name = os.path.basename(lr_file)
            if f"x{scale_factor}" in lr_name:
                hr_name = lr_name.replace(f"x{scale_factor}", "")
            else:
                hr_name = lr_name
            hr_path = os.path.join(data_dir, 'valid_HR', hr_name)
            if not os.path.exists(hr_path):
                missing_hr.append(hr_name)
        
        if missing_hr:
            print(f"\nWarning: Missing HR counterparts for {len(missing_hr)} LR files")
            for i, fname in enumerate(missing_hr[:3], 1):
                print(f"{i}. {fname}")
            if len(missing_hr) > 3:
                print(f"... plus {len(missing_hr)-3} more")
    else:
        print("Warning: No LR files found in validation set!")
        print(f"Checked directory: {valid_lr_dir}")
        print("Tried patterns:", lr_patterns)

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
        test_lr_dir = os.path.join(data_dir, "test_LR")
        test_hr_dir = os.path.join(data_dir, "test_HR")
        
        # Only look in test_LR directory for LR files
        lr_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            lr_files.extend(glob.glob(os.path.join(test_lr_dir, f'*{ext}')))
        
        path = []
        for lr_path in lr_files:
            lr_filename = os.path.basename(lr_path)
            
            # Generate HR filename - remove x4 if present
            hr_filename = lr_filename.replace('x4', '')
            hr_path = os.path.join(test_hr_dir, hr_filename)
            
            if os.path.exists(hr_path):
                path.append((lr_path, hr_path))
            else:
                print(f"Warning: Missing HR counterpart for {lr_filename}")

        if not path:
            raise FileNotFoundError(f"No valid image pairs found in {test_lr_dir}")
            
        print(f"Found {len(path)} valid image pairs for testing")
        return path

    elif mode == "valid":
        valid_lr_dir = os.path.join(data_dir, "valid_LR")
        valid_hr_dir = os.path.join(data_dir, "valid_HR")
        
        # Only look in valid_LR directory for LR files
        lr_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            lr_files.extend(glob.glob(os.path.join(valid_lr_dir, f'*{ext}')))
        
        path = []
        for lr_path in lr_files:
            lr_filename = os.path.basename(lr_path)
            
            # Generate HR filename - remove x4 if present
            hr_filename = lr_filename.replace('x4', '')
            hr_path = os.path.join(valid_hr_dir, hr_filename)
            
            if os.path.exists(hr_path):
                path.append((lr_path, hr_path))
            else:
                print(f"Warning: Missing HR counterpart for {lr_filename}")

        if not path:
            raise FileNotFoundError(f"No valid image pairs found in {valid_lr_dir}")
            
        print(f"Found {len(path)} valid image pairs for validation")
        return path

    elif mode == "hybrid_test":
        path = [
            (p.replace("_HR", "_LR").replace(".png", "x4.png"),
            p) for p in sorted(glob.glob(os.path.join(data_dir, "LSDIR_DIV2K_test_HR/*.png")))
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

def comprehensive_debug(img_lr_path, img_lr_uint, img_lr_tensor, img_sr_tensor, img_sr_uint, img_name):
    print(f"\n{'='*60}")
    print(f"DEBUGGING: {img_name}")
    print(f"{'='*60}")
    
    # 1. Check original LR file
    print("1. ORIGINAL LR IMAGE FILE:")
    try:
        orig_lr = cv2.imread(img_lr_path)
        if orig_lr is not None:
            print(f"   Shape: {orig_lr.shape}")
            print(f"   Range: [{orig_lr.min()}, {orig_lr.max()}]")
            print(f"   Mean: {orig_lr.mean():.2f}")
        else:
            print("   ❌ Could not load LR image file!")
    except Exception as e:
        print(f"   ❌ Error loading LR file: {e}")
    
    # 2. Check loaded LR uint image
    print("\n2. LOADED LR UINT IMAGE:")
    print(f"   Shape: {img_lr_uint.shape}")
    print(f"   Range: [{img_lr_uint.min()}, {img_lr_uint.max()}]")
    print(f"   Mean: {img_lr_uint.mean():.2f}")
    print(f"   Dtype: {img_lr_uint.dtype}")
    
    # 3. Check LR tensor (after uint2tensor4)
    print("\n3. LR TENSOR (after uint2tensor4):")
    print(f"   Shape: {img_lr_tensor.shape}")
    print(f"   Range: [{img_lr_tensor.min().item():.6f}, {img_lr_tensor.max().item():.6f}]")
    print(f"   Mean: {img_lr_tensor.mean().item():.6f}")
    print(f"   Std: {img_lr_tensor.std().item():.6f}")
    print(f"   Has NaN: {torch.isnan(img_lr_tensor).any().item()}")
    print(f"   Has Inf: {torch.isinf(img_lr_tensor).any().item()}")
    
    # 4. Check SR tensor (model output)
    print("\n4. SR TENSOR (model output):")
    print(f"   Shape: {img_sr_tensor.shape}")
    print(f"   Range: [{img_sr_tensor.min().item():.6f}, {img_sr_tensor.max().item():.6f}]")
    print(f"   Mean: {img_sr_tensor.mean().item():.6f}")
    print(f"   Std: {img_sr_tensor.std().item():.6f}")
    print(f"   Has NaN: {torch.isnan(img_sr_tensor).any().item()}")
    print(f"   Has Inf: {torch.isinf(img_sr_tensor).any().item()}")
    
    # 5. Check final SR uint image
    print("\n5. FINAL SR UINT IMAGE:")
    print(f"   Shape: {img_sr_uint.shape}")
    print(f"   Range: [{img_sr_uint.min()}, {img_sr_uint.max()}]")
    print(f"   Mean: {img_sr_uint.mean():.2f}")
    print(f"   Dtype: {img_sr_uint.dtype}")
    
    # 6. Sanity checks
    print("\n6. SANITY CHECKS:")
    
    # Check if LR tensor is reasonable
    if img_lr_tensor.min() < -0.01 or img_lr_tensor.max() > 1.01:
        print("   ❌ LR tensor values outside [0,1] range!")
    else:
        print("   ✅ LR tensor values in [0,1] range")
    
    # Check if SR tensor is reasonable  
    if img_sr_tensor.min() < -0.1 or img_sr_tensor.max() > 1.1:
        print("   ❌ SR tensor values significantly outside [0,1] range!")
        print(f"      Range: [{img_sr_tensor.min().item():.6f}, {img_sr_tensor.max().item():.6f}]")
    else:
        print("   ✅ SR tensor values in reasonable range")
    
    # Check for extreme values
    if torch.isnan(img_sr_tensor).any():
        print("   ❌ SR tensor contains NaN values!")
    elif torch.isinf(img_sr_tensor).any():
        print("   ❌ SR tensor contains Inf values!")
    else:
        print("   ✅ SR tensor has no NaN/Inf values")
    
    print(f"{'='*60}\n")

def run(model, model_name, data_range, tile, logger, device, args, mode="test"):
    sf = 4
    border = sf
    results = {
        f"{mode}_runtime": [],
        f"{mode}_psnr": [],
        f"{mode}_memory": 0,
        f"{mode}_ave_runtime": 0,
        f"{mode}_ave_psnr": 0
    }
    if args.ssim:
        results[f"{mode}_ssim"] = []
        results[f"{mode}_ave_ssim"] = 0

    # Get dataset paths
    try:
        data_path = select_dataset(args.data_dir, mode)
    except FileNotFoundError as e:
        logger.error(str(e))
        return results  # Return empty results structure

    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_lr, img_hr) in enumerate(data_path):
        try:
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

            # print(f"LR image range: {img_lr.min().item():.3f}-{img_lr.max().item():.3f}")
            # print(f"SR image range: {img_sr.min().item():.3f}-{img_sr.max().item():.3f}")

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
        except Exception as e:
            logger.error(f"Error processing {img_hr}: {str(e)}")
            continue

    # Calculate averages only if we have results
    if results[f"{mode}_runtime"]:
        results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
        results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"])
        results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
        
        if args.ssim:
            results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
        
        logger.info("{:>16s} : {:<.3f} [M]".format("Max Memory", results[f"{mode}_memory"]))
        logger.info("------> Average runtime of ({}) is : {:.6f} seconds".format(
            "test" if mode == "test" else "valid", 
            results[f"{mode}_ave_runtime"]))
    else:
        logger.warning(f"No images were successfully processed for {mode} set")

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
