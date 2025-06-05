import os
import subprocess
from enum import Enum
import tempfile
import cv2
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import glob

from eval import SREvaluator

base = os.path.dirname(__file__)

class GANType(Enum):
    SRGAN = 1
    REAL_ESRGAN = 2
    SwinIR = 3
    SinSR = 4

def print_welcome():
    print("\n" + "="*50)
    print("Super Resolution Comparison Tool".center(50))
    print("="*50)
    print("\nThis tool helps you compare different apporaches to super resolution (SR).")
    print("Available options:")
    print("1. SRGAN")
    print("2. Real-ESRGAN")
    print("3. SwinIR")
    print("4. SinSR")
    print("0. Evaluate SR Results")
    print("\nPlease select which approach you'd like to use:")

def get_gan_choice():
    while True:
        try:
            choice = int(input("Enter a number (1-4): "))
            if choice in [1, 2, 3, 4, 4]:
                return GANType(choice)
            elif choice == 0:
                return 0
            print("Please enter a valid number (1-4)")
        except ValueError:
            print("Please enter a number")

def get_mode(choice):
    while True:
        if choice < 3:
            mode = input("\nDo you want to (t)rain or (i)nference? [t/i]: ").lower()
            if mode in ['t', 'train']:
                return 'train'
            elif mode in ['i', 'inference']:
                return 'inference'
            print("Please enter 't' for train or 'i' for inference")
        else:
            return "i"

def get_input_path(prompt):
    while True:
        base = os.path.dirname(__file__)
        path = input(prompt)
        path = os.path.join(base, f"images/{path}")
        if os.path.exists(path):
            return path
        elif path == "":
            return None
        print(f"Path '{path}' does not exist. Please try again.")

def get_output_path(prompt):
    path = input(prompt)
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    return path

def get_integer(prompt, default=None):
    while True:
        try:
            user_input = input(prompt)
            if not user_input and default is not None:
                return default
            return int(user_input)
        except ValueError:
            print("Please enter a valid integer")

def get_float(prompt, default=None):
    while True:
        try:
            user_input = input(prompt)
            if not user_input and default is not None:
                return default
            return float(user_input)
        except ValueError:
            print("Please enter a valid number")

def get_string(prompt, default=None):
    while True:
        try:
            user_input = input(prompt)
            if not user_input and default is not None:
                return default
            return user_input
        except ValueError:
            print("Please enter a valid string")

def get_yes_no(prompt, default=None):
    while True:
        user_input = input(prompt).lower()
        if not user_input and default is not None:
            return default
        if user_input in ['y', 'yes']:
            return True
        if user_input in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")

def get_evaluation_mode():
    # Get the evaluation mode from user.
    print("\nEvaluation Options:")
    print("1. Single image evaluation")
    print("2. Batch evaluation (multiple images)")
    print("3. Model comparison (compare different SR models)")
    
    while True:
        try:
            choice = int(input("Select evaluation mode (1-3): "))
            if choice in [1, 2, 3]:
                return choice
            print("Please enter a valid number (1-3)")
        except ValueError:
            print("Please enter a number")

def get_image_files(directory):
    # Get all image files from a directory.
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
    image_files = []
    
    for ext in extensions:
        # First check files directly in the directory
        pattern = os.path.join(directory, ext)
        files_in_dir = glob.glob(pattern)
        image_files.extend(files_in_dir)
        
        # Then check subdirectories (without duplicating files in the base directory)
        pattern = os.path.join(directory, '**', ext)
        files_in_subdirs = glob.glob(pattern, recursive=True)
        # Only add files that aren't already in our list
        new_files = [f for f in files_in_subdirs if f not in files_in_dir]
        image_files.extend(new_files)
    
    return sorted(list(set(image_files)))  # Use set() to ensure no duplicates

def handle_srgan(mode):
    print("\nSRGAN selected (original implementation)")
    
    if mode == 'train':
        print("\nTraining SRGAN requires the following:")
        epochs = get_integer("Number of epochs [100]: ", 100)
        crop_size = get_integer("Batch size [88]: ", 88)
        up = get_integer("Upscale factor [4]: ", 4)
        name = get_string("Name of Directory: ", "")
        
        print("\nStarting SRGAN training with:")
        print(f"- Epochs: {epochs}")
        print(f"- Crop size: {crop_size}")
        print(f"- Upscale factor: {up}")
        print(f"- Name of directory: {name}")
        
        print("\n[Training SRGAN]")
        subprocess.run(["python", "ext/SRGAN/train.py", "--num_epochs", str(epochs), "--crop_size", 
                        str(crop_size), "--upscale_factor", str(up), "--name", name])
        
    else:  # inference
        input_path = get_input_path("Path to input image, directory, or video: ")
        up = get_integer("Upscale factor [4]: ", 4)
        if up == 2:
            name = get_string("Model name [netG_epoch_2_100.pth]: ", "netG_epoch_2_100.pth")
        else:
            name = get_string("Model name [netG_epoch_4_250.pth]: ", "netG_epoch_4_250.pth")
        memory = get_float("Percentage use of GPU [0.9]: ", 0.9)
        direc = get_string("Model directory: ", "")

        # Supported image extensions
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        # Supported video extensions
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
        
        if os.path.isdir(input_path):
            # Handle directory case
            print(f"\nProcessing all images in directory: {input_path}")
            image_files = [f for f in os.listdir(input_path) 
                          if f.lower().endswith(image_extensions)]
            
            if not image_files:
                print("No valid image files found in the directory!")
                return
            
            print(f"Found {len(image_files)} images to process")
            
            for img_file in image_files:
                img_path = os.path.join(input_path, img_file)
                print(f"\nProcessing: {img_file}")
                subprocess.run([
                    "python", "ext/SRGAN/test_image.py",
                    "--image_name", img_path,
                    "--upscale_factor", str(up),
                    "--model_name", name,
                    "--memory", str(memory),
                    "--direc", direc
                ])
            
            print("\nFinished processing all images in directory")
        
        elif input_path.lower().endswith(video_extensions):
            # Handle video case
            print("\nRunning SRGAN video processing with:")
            print(f"- Input video: {input_path}")
            print(f"- Upscale factor: {up}")
            print(f"- Model name: {name}")
            print(f"- GPU percentage: {memory}")
            
            print("\n[Processing Video]")
            subprocess.run([
                "python", "ext/SRGAN/test_video.py",
                "--video_name", input_path,
                "--upscale_factor", str(up),
                "--model_name", name,
                # "--memory", str(memory),
                # "--direc", direc
            ])
        
        else:
            # Handle single image case
            if not input_path.lower().endswith(image_extensions):
                print("\nWarning: File extension not recognized as supported image format!")
                proceed = get_yes_no("Continue anyway? [y/n]: ", False)
                if not proceed:
                    return
            
            print("\nRunning SRGAN inference with:")
            print(f"- Input image: {input_path}")
            print(f"- Upscale factor: {up}")
            print(f"- Model name: {name}")
            print(f"- GPU percentage: {memory}")
            
            print("\n[Testing SRGAN]")
            subprocess.run([
                "python", "ext/SRGAN/test_image.py",
                "--image_name", input_path,
                "--upscale_factor", str(up),
                "--model_name", name,
                "--memory", str(memory),
                "--direc", direc
            ])

def handle_real_esrgan(mode):
    print("\nReal-ESRGAN selected (enhanced practical version)")
    
    if mode == 'train':
        print("\nTraining Real-ESRGAN with the following options:")
        # epochs = get_integer("Number of epochs [100]: ", 100)
        crop_size = get_integer("Crop size [64]: ", 64)
        batch_size = get_integer("Batch size [16]: ", 16) 
        upscale = get_integer("Upscale factor [4]: ", 4)
        # Num batches per epoch = total num images // batch_size
        resume_batch = get_integer("Resume from stopped training [0]: ", 0)
        warmup = get_integer("Warmup batches [1000]: ", 1000) # Num batches per epoch * desired num of epochs
        total_batches = get_integer("Total batches [1200]: ", 1200)
        res_blocks = get_integer("Number of residual blocks [23]: ", 23)
        lr = get_float("Learning rate [0.0002]: ", 0.0002)
        name = get_string("Name of Directory: ", "")

        print("\nStarting Real-ESRGAN training with:")
        # print(f"- Epochs: {epochs}")
        print(f"- Crop size: {crop_size}")
        print(f"- Batch size: {batch_size}")
        print(f"- Upscale factor: {upscale}")
        print(f"- Warmup batches: {warmup}")
        print(f"- Total batches: {total_batches}")
        print(f"- Learning rate: {lr}")
        print(f"- Name of directory: {name}")
        
        print("\n[Training Real-ESRGAN]")
        subprocess.run([
            "python", "ext/Real-ESRGAN/train.py",
            # "--n_batches", str(epochs),
            "--crop_size", str(crop_size),
            "--batch_size", str(batch_size),
            "--upscale_factor", str(upscale),
            "--batch", str(resume_batch),
            "--warmup_batches", str(warmup),
            "--n_batches", str(total_batches),
            "--lr", str(lr),
            "--name", name
        ])
        
    else:  # inference
        input_path = get_input_path("Path to input image, directory, or video: ")
        num_batches = get_integer("How many batches did you train?: ", 1200)
        len_train = get_integer("How many batches per iteration were there?: ", 45)
        num_batches_mod = num_batches % len_train
        n_batch = num_batches + num_batches_mod + 5
        model_name = get_string(f"Model name [generator_{n_batch}.pth]: ", f"generator_{n_batch}.pth")
        test_mode = get_yes_no("Use GPU acceleration? [y/n]: ", True)
        
        # Supported image extensions
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        if os.path.isdir(input_path):
            # Handle directory case
            print(f"\nProcessing all images in directory: {input_path}")
            image_files = [f for f in os.listdir(input_path) 
                          if f.lower().endswith(image_extensions)]
            
            if not image_files:
                print("No valid image files found in the directory!")
                return
            
            print(f"Found {len(image_files)} images to process")
            
            for img_file in image_files:
                img_path = os.path.join(input_path, img_file)
                print(f"\nProcessing: {img_file}")
                subprocess.run([
                    "python", "ext/Real-ESRGAN/test_images.py",
                    "--n_batch", str(num_batches),
                    "--image_name", img_path,
                    "--model_name", model_name,
                    "--test_mode", "GPU" if test_mode else "CPU"
                ])
            
            print("\nFinished processing all images in directory")
        else:
            # Handle single image case
            if not input_path.lower().endswith(image_extensions):
                print("\nWarning: File extension not recognized as supported image format!")
                proceed = get_yes_no("Continue anyway? [y/n]: ", False)
                if not proceed:
                    return
            
            print("\nRunning Real-ESRGAN inference with:")
            print(f"- Input image: {input_path}")
            print(f"- Model name: {model_name}")
            print(f"- GPU acceleration: {'Yes' if test_mode else 'No'}")
            
            print("\n[Testing Real-ESRGAN]")
            subprocess.run([
                "python", "ext/Real-ESRGAN/test_images.py",
                "--n_batch", str(num_batches),
                "--image_name", input_path,
                "--model_name", model_name,
                "--test_mode", "GPU" if test_mode else "CPU"
            ])

def handle_swinIR(mode):
    print("\nSwinIR selected (state-of-the-art image restoration)")
    
    # Get input path
    input_path = get_input_path("Path to input image or directory: ")
    
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # Task selection
    print("\nAvailable SwinIR tasks:")
    print("1. DFO only, GAN with dict keys + params, and params ema")
    print("2. DFO only, PSNR with dict keys + params, and params ema")
    print("3. DFO with MFC, GAN with dict keys + params, and params ema")
    print("4. DFO with MFC, PSNR with dict keys + params, and params ema")
    task_choice = get_integer("Select model [1-4]: ", 1)
    
    tasks = {
        1: '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN-with-dict-keys-params-and-params_ema.pth',
        2: '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR-with-dict-keys-params-and-params_ema.pth',
        3: '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN-with-dict-keys-params-and-params_ema.pth',
        4: '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR-with-dict-keys-params-and-params_ema.pth'
    }
    
    if task_choice == 1:
        dir_name = "DFO_GAN"
        large_model = False
    elif task_choice == 2:
        dir_name = "DFO_PSNR"
        large_model = False
    elif task_choice == 3:
        dir_name = "DFOWMFC_GAN"
        large_model = True
    else:
        dir_name = "DFOWMFC_PSNR"
        large_model = True

    task = tasks[task_choice]
    
    scale = get_integer("Upscale factor [2, 3, 4, 8]: ", 4)
    noise = None
    jpeg = None
    
    # Tile settings for large images
    tile_size = get_integer("Tile size for processing (0 for no tiling) [0 or 128-512]: ", 0)
    tile_overlap = 32 if tile_size > 0 else 0
    
    # GPU acceleration
    test_mode = get_yes_no("Use GPU acceleration? [y/n]: ", True)
    
    if os.path.isdir(input_path):
        # Handle directory case
        print(f"\nProcessing all images in directory: {input_path}")
        image_files = [f for f in os.listdir(input_path) 
                        if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print("No valid image files found in the directory!")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for img_file in image_files:
            img_path = os.path.join(input_path, img_file)
            print(f"\nProcessing: {img_file}")
            
            cmd = [
                "python", "ext/SwinIR/main_test_swinir.py",
                "--task", "real_sr",
                "--scale", str(scale),
                "--folder_lq", input_path,
                "--model_name", task,
                "--dir_name", dir_name
            ]
            
            if large_model:
                cmd.append("--large_model")
            
            if tile_size > 0:
                cmd.extend([
                    "--tile", str(tile_size),
                    "--tile_overlap", str(tile_overlap)
                ])
            
            subprocess.run(cmd)
        
        print("\nFinished processing all images in directory")
    else:
        # Handle single image case
        if not input_path.lower().endswith(image_extensions):
            print("\nWarning: File extension not recognized as supported image format!")
            proceed = get_yes_no("Continue anyway? [y/n]: ", False)
            if not proceed:
                return
        
        print("\nRunning SwinIR inference with:")
        print(f"- Task: {task}")
        if scale: print(f"- Scale: {scale}")
        if noise: print(f"- Noise level: {noise}")
        if jpeg: print(f"- JPEG quality: {jpeg}")
        print(f"- GPU acceleration: {'Yes' if test_mode else 'No'}")
        if tile_size > 0: 
            print(f"- Tile processing: {tile_size}x{tile_size} with {tile_overlap}px overlap")
        
        # Create temp directory for single image processing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input structure SwinIR expects
            input_dir = os.path.join(temp_dir, "input")
            os.makedirs(input_dir, exist_ok=True)
            
            # Copy/link the input image
            import shutil
            shutil.copy(input_path, input_dir)
            
            cmd = [
                "python", "main_test_swinir.py",
                "--task", "real_sr",
                "--scale", str(scale),
                "--folder_lq", input_dir,
                "--model_name", task,
                "--dir_name", dir_name
            ]
            
            if large_model:
                cmd.append("--large_model")
            
            if tile_size > 0:
                cmd.extend([
                    "--tile", str(tile_size),
                    "--tile_overlap", str(tile_overlap)
                ])
            
            print("\n[Testing SwinIR]")
            subprocess.run(cmd)
            
            # Copy results back to original directory
            output_dir = os.path.join(temp_dir, "results", f"swinir_{task}_x{scale}")
            for result_file in os.listdir(output_dir):
                src = os.path.join(output_dir, result_file)
                dst = os.path.join(os.path.dirname(input_path), 
                                    f"swinir_{os.path.splitext(os.path.basename(input_path))[0]}.png")
                shutil.copy(src, dst)
                print(f"\nSaved result to: {dst}")

def handle_sinSR(mode):
    cmd = [
        "python", "ext/SinSR/app.py"
    ]
    
    subprocess.run(cmd)

def evaluate_sr(input_data: Union[str, np.ndarray, List[str]], 
                gt_dir: Optional[str] = None,
                output_dir: str = "evaluation_results",
                model_name: str = "model",
                image_name: str = "image",
                save_report: bool = True,
                verbose: bool = True) -> Tuple[Dict, Optional[pd.DataFrame]]:
    
    # Initialize evaluator
    evaluator = SREvaluator(gt_dir=gt_dir)
    
    # Determine evaluation mode
    if isinstance(input_data, str):
        # Single image path
        results = _evaluate_single_image(evaluator, input_data, model_name, image_name, verbose)
        df = None
    elif isinstance(input_data, np.ndarray):
        # Single image array
        results = _evaluate_single_array(evaluator, input_data, model_name, image_name, gt_dir, verbose)
        df = None
    elif isinstance(input_data, list):
        # Batch evaluation
        results, df = _evaluate_batch(evaluator, input_data, verbose)
    else:
        raise ValueError("input_data must be a string (path), numpy array, or list of paths")
    
    # Generate comprehensive report if requested
    if save_report and results:
        _generate_comprehensive_report(evaluator, results, df, output_dir, verbose)
    
    return results, df

def _evaluate_single_image(evaluator: 'SREvaluator', 
                          image_path: str, 
                          model_name: str, 
                          image_name: str, 
                          verbose: bool) -> Dict:
    # Evaluate a single image from file path.
    if verbose:
        print(f"Evaluating single image: {image_path}")
    
    try:
        # Load the image
        sr_img = evaluator.load_image(image_path)
        
        # Perform comprehensive evaluation
        results = _perform_comprehensive_evaluation(evaluator, sr_img, model_name, image_name, verbose)
        
        if verbose:
            _print_single_results(results)
        
        return {(image_name, model_name): results}
    
    except Exception as e:
        print(f"Error evaluating {image_path}: {str(e)}")
        return {}

def _evaluate_single_array(evaluator: 'SREvaluator', 
                          image_array: np.ndarray, 
                          model_name: str, 
                          image_name: str, 
                          gt_dir: Optional[str],
                          verbose: bool) -> Dict:
    # Evaluate a single image from numpy array.
    if verbose:
        print(f"Evaluating image array: {image_name}")
    
    try:
        # Perform comprehensive evaluation
        results = _perform_comprehensive_evaluation(evaluator, image_array, model_name, image_name, verbose)
        
        if verbose:
            _print_single_results(results)
        
        return {(image_name, model_name): results}
    
    except Exception as e:
        print(f"Error evaluating image array: {str(e)}")
        return {}

def _evaluate_batch(evaluator: 'SREvaluator', 
                   image_paths: List[str], 
                   verbose: bool) -> Tuple[Dict, pd.DataFrame]:
    # Evaluate multiple images for batch processing.
    if verbose:
        print(f"Evaluating batch of {len(image_paths)} images")
    
    all_results = {}
    
    for i, path in enumerate(image_paths):
        if verbose:
            print(f"\nProcessing {i+1}/{len(image_paths)}: {path}")
        
        try:
            # Extract model and image names from path
            model_name, image_name = _extract_names_from_path(path)
            
            # Load and evaluate image
            sr_img = evaluator.load_image(path)
            results = _perform_comprehensive_evaluation(evaluator, sr_img, model_name, image_name, verbose)
            
            all_results[(image_name, model_name)] = results
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    # Create DataFrame for analysis
    if all_results:
        # Convert results to list format for DataFrame
        results_list = []
        for key, result in all_results.items():
            if isinstance(key, tuple) and len(key) == 2:
                result_copy = result.copy()
                result_copy['image'] = key[0]
                result_copy['model'] = key[1]
                results_list.append(result_copy)
        
        if results_list:
            df = pd.DataFrame(results_list)
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    # Perform model consistency analysis
    if len(df) > 1 and 'PSNR' in df.columns and 'SSIM' in df.columns:
        if verbose:
            print("\nPerforming model consistency analysis...")
        try:
            consistency = evaluator.model_consistency_analysis(df)
            all_results['consistency_analysis'] = consistency
        except Exception as e:
            if verbose:
                print(f"Model consistency analysis failed: {str(e)}")
    
    return all_results, df

def _perform_comprehensive_evaluation(evaluator: 'SREvaluator', 
                                    sr_img: np.ndarray, 
                                    model_name: str, 
                                    image_name: str, 
                                    verbose: bool) -> Dict:
    # Perform all available evaluation metrics on a single image.
    results = {'model': model_name, 'image': image_name}
    
    # Load ground truth if available
    gt_img = None
    if evaluator.gt_dir:
        try:
            gt_path = os.path.join(evaluator.gt_dir, f"{image_name}.png")
            if os.path.exists(gt_path):
                gt_img = evaluator.load_image(gt_path)
            else:
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.tiff', '.bmp']:
                    alt_path = os.path.join(evaluator.gt_dir, f"{image_name}{ext}")
                    if os.path.exists(alt_path):
                        gt_img = evaluator.load_image(alt_path)
                        break
        except Exception as e:
            if verbose:
                print(f"Could not load ground truth: {str(e)}")
    
    # 1. Traditional Metrics (PSNR, SSIM, MSE)
    if gt_img is not None:
        if verbose:
            print("  Computing traditional metrics...")
        
        for metric_name, metric_func in evaluator.metrics.items():
            try:
                if metric_name == 'VMAF':
                    # VMAF requires special handling
                    results[metric_name] = metric_func(gt_img, sr_img)
                else:
                    results[metric_name] = metric_func(gt_img, sr_img)
                
                if verbose and results[metric_name] is not None:
                    print(f"    {metric_name}: {results[metric_name]:.4f}")
            except Exception as e:
                if verbose:
                    print(f"    {metric_name}: Failed ({str(e)})")
                results[metric_name] = None
    
    # 2. Visual Analysis
    if verbose:
        print("  Computing visual analysis...")
    try:
        visual_results = evaluator.visual_analysis(sr_img, gt_img)
        results.update(visual_results)
    except Exception as e:
        if verbose:
            print(f"  Visual analysis failed: {str(e)}")
    
    # 3. Perceptual Analysis
    if verbose:
        print("  Computing perceptual analysis...")
    try:
        perceptual_results = evaluator.perceptual_analysis(sr_img)
        results.update(perceptual_results)
        
        if verbose:
            for key, value in perceptual_results.items():
                if value is not None:
                    print(f"    {key}: {value:.4f}")
    except Exception as e:
        if verbose:
            print(f"    Perceptual analysis failed: {str(e)}")
    
    # 4. Additional Quality Metrics
    if verbose:
        print("  Computing additional metrics...")
    
    try:
        # Image statistics
        results['mean_brightness'] = np.mean(sr_img)
        results['std_brightness'] = np.std(sr_img)
        results['contrast'] = np.std(cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY))
        
        # Sharpness metrics
        gray = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        results['sharpness_laplacian'] = laplacian_var
        
        # Artifact detection
        results['blocking_artifacts'] = _detect_blocking_artifacts(sr_img)
        results['ringing_artifacts'] = _detect_ringing_artifacts(sr_img)
    except Exception as e:
        if verbose:
            print(f"  Additional metrics failed: {str(e)}")
    
    return results

def _detect_blocking_artifacts(img: np.ndarray) -> float:
    # Simple blocking artifact detection.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Detect horizontal and vertical edges
    h_edges = np.abs(np.diff(gray, axis=0)).mean()
    v_edges = np.abs(np.diff(gray, axis=1)).mean()
    return (h_edges + v_edges) / 2

def _detect_ringing_artifacts(img: np.ndarray) -> float:
    # Simple ringing artifact detection using high-pass filtering.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply high-pass filter
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(gray, -1, kernel)
    return np.std(filtered)

def _extract_names_from_path(path: str) -> Tuple[str, str]:
    # Extract model and image names from file path.
    # Example: "output/SwinIR/image001_SwinIR.png" -> ("SwinIR", "image001")
    filename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))
    
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Try to extract image name (remove model suffix if present)
    if '_' in name_without_ext:
        parts = name_without_ext.split('_')
        if len(parts) > 1 and parts[-1] == dirname:
            image_name = '_'.join(parts[:-1])
        else:
            image_name = name_without_ext
    else:
        image_name = name_without_ext
    
    model_name = dirname if dirname else "unknown"
    
    return model_name, image_name

def _print_single_results(results: Dict):
    # Print formatted results for a single image evaluation.
    print(f"\nResults for {results['image']} ({results['model']}):")
    print("-" * 50)
    
    # Traditional metrics
    metrics_to_show = ['PSNR', 'SSIM', 'MSE', 'VMAF']
    for metric in metrics_to_show:
        if metric in results and results[metric] is not None:
            print(f"{metric:20}: {results[metric]:.4f}")
    
    # Visual metrics
    if 'edge_intensity' in results:
        print(f"{'Edge Intensity':20}: {results['edge_intensity']:.4f}")
    if 'texture_score' in results:
        print(f"{'Texture Score':20}: {results['texture_score']:.4f}")
    if 'sharpness_laplacian' in results:
        print(f"{'Sharpness':20}: {results['sharpness_laplacian']:.4f}")
    
    # Perceptual metrics
    if 'lpips' in results and results['lpips'] is not None:
        print(f"{'LPIPS':20}: {results['lpips']:.4f}")
    if 'niqe' in results and results['niqe'] is not None:
        print(f"{'NIQE':20}: {results['niqe']:.4f}")

def _generate_comprehensive_report(evaluator: 'SREvaluator', 
                                 results: Dict, 
                                 df: Optional[pd.DataFrame], 
                                 output_dir: str, 
                                 verbose: bool):
    # Generate comprehensive evaluation reports.
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\nGenerating comprehensive report in {output_dir}/")
    
    # Save detailed results
    if df is not None:
        # Multi-image evaluation
        df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        # Only generate summary statistics for metrics that exist
        available_metrics = [col for col in df.columns if col not in ['image', 'model']]
        
        if available_metrics:
            # Generate summary statistics only for available metrics
            summary = df.groupby('model').agg({
                col: ['mean', 'std', 'min', 'max'] 
                for col in available_metrics 
                if col in df.columns and df[col].dtype in ['float64', 'int64']
            }).round(4)
            summary.to_csv(os.path.join(output_dir, 'model_summary.csv'))
        
        # Generate visual comparisons if possible
        try:
            evaluator.generate_visual_comparisons(df, output_dir)
        except Exception as e:
            if verbose:
                print(f"Could not generate visual comparisons: {str(e)}")
        
        # Create performance plots for available metrics
        _create_performance_plots(df, output_dir)
        
    else:
        # Single image evaluation
        single_df = pd.DataFrame([list(results.values())[0]])
        single_df.to_csv(os.path.join(output_dir, 'single_evaluation.csv'), index=False)
    
    # Save consistency analysis if available
    if 'consistency_analysis' in results:
        consistency = results['consistency_analysis']
        with open(os.path.join(output_dir, 'consistency_analysis.txt'), 'w') as f:
            f.write("Model Consistency Analysis\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("Pairwise PSNR Correlations:\n")
            for pair, corr in consistency['pairwise_psnr_corr'].items():
                f.write(f"  {pair}: {corr:.4f}\n")
            
            f.write(f"\nMean PSNR Standard Deviation: {consistency['mean_psnr_std']:.4f}\n")
            f.write(f"Mean SSIM Standard Deviation: {consistency['mean_ssim_std']:.4f}\n")
    
    if verbose:
        print("Report generation complete!")

def _create_performance_plots(df: pd.DataFrame, output_dir: str):
    # Create performance visualization plots.
    available_metrics = [col for col in df.columns if col not in ['image', 'model']]
    
    # Plot whatever metrics we have
    for i in range(0, len(available_metrics), 2):
        fig, axs = plt.subplots(1, min(2, len(available_metrics)-i), figsize=(15, 6))
        
        for j, metric in enumerate(available_metrics[i:i+2]):
            if df[metric].dtype in ['float64', 'int64']:
                df.boxplot(column=metric, by='model', ax=axs[j] if len(available_metrics[i:i+2]) > 1 else axs)
                axs[j].set_title(f'{metric} Distribution by Model')
                axs[j].set_xlabel('Model')
                axs[j].set_ylabel(metric)
        
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_plots_{i//2}.png'), dpi=300)
        plt.close()

def handle_evaluation():
    # Handle the SR evaluation process.
    print("\n" + "="*50)
    print("Super Resolution Evaluation".center(50))
    print("="*50)
    
    eval_mode = get_evaluation_mode()
    
    if eval_mode == 1:
        # Single image evaluation
        handle_single_image_evaluation()
    elif eval_mode == 2:
        # Batch evaluation
        handle_batch_evaluation()
    elif eval_mode == 3:
        # Model comparison
        handle_model_comparison()

def handle_single_image_evaluation():
    # Handle single image evaluation.
    print("\n--- Single Image Evaluation ---")
    
    # Get SR image path
    sr_path = None
    while sr_path is None:
        path_input = input("Enter path to SR image (relative to images/ folder): ")
        sr_path = os.path.join(base, "images", path_input)
        if not os.path.exists(sr_path):
            print(f"File not found: {sr_path}")
            sr_path = None
    
    # Get ground truth directory (optional)
    gt_dir = None
    if get_yes_no("Do you have ground truth images for comparison? [y/n]: "):
        gt_input = input("Enter path to ground truth directory (relative to images/ folder): ")
        gt_dir = os.path.join(base, "images", gt_input)
        if not os.path.exists(gt_dir):
            print(f"Ground truth directory not found: {gt_dir}")
            gt_dir = None
    
    # Get model and image names
    model_name = get_string("Enter model name (e.g., SwinIR, SRGAN): ", "Unknown")
    image_name = get_string("Enter image name (without extension): ", 
                           os.path.splitext(os.path.basename(sr_path))[0])
    
    # Get output directory
    output_dir = get_string("Enter output directory for results [evaluation_results]: ", 
                           "evaluation_results")
    
    # Run evaluation
    print(f"\nEvaluating {sr_path}...")
    try:
        results, _ = evaluate_sr(
            input_data=sr_path,
            gt_dir=gt_dir,
            output_dir=output_dir,
            model_name=model_name,
            image_name=image_name,
            save_report=True,
            verbose=True
        )
        print(f"\nEvaluation complete! Results saved to {output_dir}/")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

def handle_batch_evaluation():
    # Handle batch evaluation of multiple images.
    print("\n--- Batch Evaluation ---")
    
    # Get input directory
    input_dir = None
    while input_dir is None:
        dir_input = input("Enter directory containing SR images (relative to output/ folder): ")
        input_dir = os.path.join(base, "output", dir_input)
        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            input_dir = None
    
    # Find all image files
    image_files = get_image_files(input_dir)
    if not image_files:
        print("No image files found in the directory!")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Get ground truth directory (optional)
    gt_dir = None
    if get_yes_no("Do you have ground truth images for comparison? [y/n]: "):
        gt_input = input("Enter path to ground truth directory (relative to images/ folder): ")
        gt_dir = os.path.join(base, "images", gt_input)
        if not os.path.exists(gt_dir):
            print(f"Ground truth directory not found: {gt_dir}")
            gt_dir = None
    
    # Get output directory
    output_dir = get_string("Enter output directory for results [batch_evaluation]: ", 
                           "batch_evaluation")
    
    # Run batch evaluation
    print(f"\nEvaluating {len(image_files)} images...")
    try:
        output_dir = os.path.join(output_dir, dir_input)
        os.makedirs(output_dir, exist_ok=True)

        results, df = evaluate_sr(
            input_data=image_files,
            gt_dir=gt_dir,
            output_dir=output_dir,
            save_report=True,
            verbose=True
        )
        
        if df is not None and len(df) > 0:
            print(f"\nBatch evaluation complete!")
            print(f"Results saved to {output_dir}/")
            print(f"\nSummary:")
            print(f"- Total images evaluated: {len(df)}")
            
            # Only show metrics that actually exist
            available_metrics = [col for col in df.columns if col not in ['image', 'model']]
            print(f"- Available metrics: {', '.join(available_metrics)}")
            
            if 'PSNR' in df.columns:
                print(f"- Average PSNR: {df['PSNR'].mean():.2f} dB")
            if 'SSIM' in df.columns:
                print(f"- Average SSIM: {df['SSIM'].mean():.4f}")
            if 'lpips' in df.columns:
                print(f"- Average LPIPS: {df['lpips'].mean():.4f}")
        else:
            print("No valid results obtained from batch evaluation.")
            
    except Exception as e:
        print(f"Batch evaluation failed: {str(e)}")

def handle_model_comparison():
    # Handle comparison between different SR models.
    print("\n--- Model Comparison ---")
    
    # Get model directories
    model_dirs = []
    print("Enter directories for each model you want to compare:")
    print("(Press Enter with empty input when done)")
    
    while True:
        model_input = input(f"Model {len(model_dirs)+1} directory (relative to images/): ")
        if not model_input:
            break
            
        model_dir = os.path.join(base, "images", model_input)
        if os.path.exists(model_dir):
            model_dirs.append(model_dir)
            print(f"Added: {model_dir}")
        else:
            print(f"Directory not found: {model_dir}")
    
    if len(model_dirs) < 2:
        print("Need at least 2 model directories for comparison!")
        return
    
    # Collect all image files from all model directories
    all_image_files = []
    for model_dir in model_dirs:
        model_files = get_image_files(model_dir)
        all_image_files.extend(model_files)
    
    if not all_image_files:
        print("No image files found in the model directories!")
        return
    
    print(f"Found {len(all_image_files)} total images across {len(model_dirs)} models")
    
    # Get ground truth directory (optional)
    gt_dir = None
    if get_yes_no("Do you have ground truth images for comparison? [y/n]: "):
        gt_input = input("Enter path to ground truth directory (relative to images/ folder): ")
        gt_dir = os.path.join(base, "images", gt_input)
        if not os.path.exists(gt_dir):
            print(f"Ground truth directory not found: {gt_dir}")
            gt_dir = None
    
    # Get output directory
    output_dir = get_string("Enter output directory for comparison results [model_comparison]: ", 
                           "model_comparison")
    
    # Run model comparison
    print(f"\nComparing {len(model_dirs)} models...")
    try:
        results, df = evaluate_sr(
            input_data=all_image_files,
            gt_dir=gt_dir,
            output_dir=output_dir,
            save_report=True,
            verbose=True
        )
        
        if df is not None and len(df) > 0:
            print(f"\nModel comparison complete!")
            print(f"Results saved to {output_dir}/")
            
            # Print comparison summary
            print(f"\nComparison Summary:")
            print("-" * 40)
            
            models = df['model'].unique()
            for model in models:
                model_data = df[df['model'] == model]
                print(f"\n{model}:")
                print(f"  Images evaluated: {len(model_data)}")
                if 'PSNR' in df.columns:
                    print(f"  Average PSNR: {model_data['PSNR'].mean():.2f} ± {model_data['PSNR'].std():.2f} dB")
                if 'SSIM' in df.columns:
                    print(f"  Average SSIM: {model_data['SSIM'].mean():.4f} ± {model_data['SSIM'].std():.4f}")
            
            # Find best performing model
            if 'PSNR' in df.columns:
                best_psnr_model = df.groupby('model')['PSNR'].mean().idxmax()
                print(f"\nBest PSNR: {best_psnr_model}")
            if 'SSIM' in df.columns:
                best_ssim_model = df.groupby('model')['SSIM'].mean().idxmax()
                print(f"Best SSIM: {best_ssim_model}")
                
        else:
            print("No valid results obtained from model comparison.")
            
    except Exception as e:
        print(f"Model comparison failed: {str(e)}")

def main():
    print_welcome()
    gan_choice = get_gan_choice()
    
    if gan_choice == 0:
        # Handle evaluation
        handle_evaluation()
    else:
        # Handle SR model selection
        mode = get_mode(gan_choice.value)
        
        if gan_choice == GANType.SRGAN:
            handle_srgan(mode)
        elif gan_choice == GANType.REAL_ESRGAN:
            handle_real_esrgan(mode)
        elif gan_choice == GANType.SwinIR:
            handle_swinIR(mode)
        elif gan_choice == GANType.SinSR:
            handle_sinSR(mode)
    
    print("\nOperation completed!")

if __name__ == "__main__":
    main()