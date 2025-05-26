import os
import subprocess
import sys
from enum import Enum

class GANType(Enum):
    SRGAN = 1
    REAL_ESRGAN = 2
    SwinIR = 3
    DIPNet = 4
    SinSR = 5
    StableSR = 6

def print_welcome():
    print("\n" + "="*50)
    print("Super Resolution Comparison Tool".center(50))
    print("="*50)
    print("\nThis tool helps you compare different apporaches to super resolution (SR).")
    print("Available options:")
    print("1. SRGAN")
    print("2. Real-ESRGAN")
    print("3. SwinIR")
    print("4. DIPNet")
    print("5. SinSR")
    print("6. StableSR")
    print("\nPlease select which approach you'd like to use:")

def get_gan_choice():
    while True:
        try:
            choice = int(input("Enter a number (1-6): "))
            if choice in [1, 2, 3, 4, 5, 6]:
                return GANType(choice)
            print("Please enter a valid number (1-6)")
        except ValueError:
            print("Please enter a number")

def get_mode():
    while True:
        mode = input("\nDo you want to (t)rain or (i)nference? [t/i]: ").lower()
        if mode in ['t', 'train']:
            return 'train'
        elif mode in ['i', 'inference']:
            return 'inference'
        print("Please enter 't' for train or 'i' for inference")

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
    
    if mode == 'train':
        print("\nNote: SwinIR training requires specialized setup and large datasets.")
        print("For most users, we recommend using pretrained models for inference.")
        proceed = get_yes_no("Continue with training setup? [y/n]: ", False)
        if not proceed:
            return
        
        # Training parameters would go here
        # (SwinIR training is complex and typically requires config files)
        # print("\nSwinIR training requires manual configuration files.")
        # print("Please refer to the official SwinIR repository for training setup.")
        
    else:  # inference
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

def handle_zssr(mode):
    pass

def handle_dipNet(mode):
    pass

def handle_sinSR(mode):
    pass    

def handle_stableSR(mode):
    pass

def main():
    print_welcome()
    gan_choice = get_gan_choice()
    mode = get_mode()
    
    if gan_choice == GANType.SRGAN:
        handle_srgan(mode)
    elif gan_choice == GANType.REAL_ESRGAN:
        handle_real_esrgan(mode)
    elif gan_choice == GANType.SwinIR:
        handle_swinIR(mode)
    elif gan_choice == GANType.ZSSR:
        handle_dipNet(mode)
    elif gan_choice == GANType.SinSR:
        handle_sinSR(mode)
    elif gan_choice == GANType.StableSR:
        handle_stableSR(mode)
    
    print("\nOperation completed!")

if __name__ == "__main__":
    main()