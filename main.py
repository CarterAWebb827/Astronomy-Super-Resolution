import os
import subprocess
import sys
from enum import Enum

class GANType(Enum):
    SRGAN = 1
    REAL_ESRGAN = 2

def print_welcome():
    print("\n" + "="*50)
    print("GAN Comparison Tool".center(50))
    print("="*50)
    print("\nThis tool helps you compare different Generative Adversarial Networks")
    print("for super-resolution tasks. Available options:")
    print("1. SRGAN")
    print("2. Real-ESRGAN")
    print("3. Real-ESRGAN")
    print("4. Real-ESRGAN")
    print("5. Real-ESRGAN")
    print("6. Real-ESRGAN")
    print("7. Real-ESRGAN")
    print("8. Real-ESRGAN")
    print("\nPlease select which GAN you'd like to use:")

def get_gan_choice():
    while True:
        try:
            choice = int(input("Enter a number (1-8): "))
            if choice in [1, 2, 3, 4, 5, 6, 7, 8]:
                return GANType(choice)
            print("Please enter a valid number (1-8)")
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
            name = get_string("Model name [netG_epoch_4_100.pth]: ", "netG_epoch_4_100.pth")
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
        crop_size = get_integer("Crop size [88]: ", 88)
        batch_size = get_integer("Batch size [48]: ", 48) 
        upscale = get_integer("Upscale factor [4]: ", 4)
        # Num batches per epoch = total num images // batch_size
        warmup = get_integer("Warmup batches [400]: ", 400) # Num batches per epoch * desired num of epochs
        total_batches = get_integer("Total batches [711]: ", 711)
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
            "--warmup_batches", str(warmup),
            "--n_batches", str(total_batches),
            "--lr", str(lr),
            "--name", name
        ])
        
    else:  # inference
        input_path = get_input_path("Path to input image, directory, or video: ")
        model_name = get_string("Model name [generator_720.pth]: ", "generator_720.pth")
        test_mode = get_yes_no("Use GPU acceleration? [y/n]: ", True)
        
        # Supported image extensions
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        # Supported video extensions
        # video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
        
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
                    "--image_name", img_path,
                    "--model_name", model_name,
                    "--test_mode", "GPU" if test_mode else "CPU"
                ])
            
            print("\nFinished processing all images in directory")
        
        # elif input_path.lower().endswith(video_extensions):
        #     # Handle video case
        #     print("\nRunning Real-ESRGAN video processing with:")
        #     print(f"- Input video: {input_path}")
        #     print(f"- Model name: {model_name}")
        #     print(f"- GPU acceleration: {'Yes' if test_mode else 'No'}")
            
        #     print("\n[Processing Video]")
        #     # Note: You would need to implement or point to a video processing script
        #     print("Video processing not implemented in this example")
        
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
                "--image_name", input_path,
                "--model_name", model_name,
                "--test_mode", "GPU" if test_mode else "CPU"
            ])

def main():
    print_welcome()
    gan_choice = get_gan_choice()
    mode = get_mode()
    
    if gan_choice == GANType.SRGAN:
        handle_srgan(mode)
    elif gan_choice == GANType.REAL_ESRGAN:
        handle_real_esrgan(mode)
    
    print("\nOperation completed!")

if __name__ == "__main__":
    main()