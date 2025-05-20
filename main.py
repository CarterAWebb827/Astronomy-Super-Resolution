import os
import subprocess
import sys
from enum import Enum

class GANType(Enum):
    SRGAN = 1
    ESRGAN_PYTORCH = 2
    REAL_ESRGAN = 3

def print_welcome():
    print("\n" + "="*50)
    print("GAN Comparison Tool".center(50))
    print("="*50)
    print("\nThis tool helps you compare different Generative Adversarial Networks")
    print("for super-resolution tasks. Available options:")
    print("1. SRGAN")
    print("2. ESRGAN")
    print("3. Real-ESRGAN")
    print("\nPlease select which GAN you'd like to use:")

def get_gan_choice():
    while True:
        try:
            choice = int(input("Enter 1, 2, or 3: "))
            if choice in [1, 2, 3]:
                return GANType(choice)
            print("Please enter a valid number (1-3)")
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
        use_memory_save = get_string("Use memory save? [y/n]: ", "n")

        if use_memory_save == "y":
            mem_save = True
        else:
            mem_save = False

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
                    "--direc", direc,
                    "--use_memory_save", str(mem_save)
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
                # "--direc", direc,
                # "--use_memory_save", str(mem_save)
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
                "--direc", direc,
                "--use_memory_save", str(mem_save)
            ])

def handle_esrgan(mode):
    print("\nESRGAN (PyTorch implementation) selected")
    
    if mode == 'train':
        print("\nTraining ESRGAN requires the following:")
        train_path = get_input_path("Path to HR training images: ")
        val_path = get_input_path("Path to HR validation images: ")
        lr_path = get_input_path("Path to LR training images (leave empty to generate): ", "")
        epochs = get_integer("Number of epochs [500]: ", 500)
        crop_size = get_integer("Batch size [16]: ", 16)
        lr = get_float("Learning rate [0.0002]: ", 0.0002)
        
        print("\nStarting ESRGAN training with:")
        print(f"- HR training data: {train_path}")
        if lr_path:
            print(f"- LR training data: {lr_path}")
        else:
            print("- LR images will be generated automatically")
        print(f"- HR validation data: {val_path}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {crop_size}")
        print(f"- Learning rate: {lr}")
        
        # Here you would call the actual training script
        print("\n[This would launch the actual ESRGAN training]")
        
    else:  # inference
        input_path = get_input_path("Path to input image or directory: ")
        output_path = get_output_path("Path to save output: ")
        scale = get_integer("Scale factor (2, 4) [4]: ", 4)
        model_path = get_input_path("Path to pretrained model (leave empty for default): ", "")
        
        print("\nRunning ESRGAN inference with:")
        print(f"- Input: {input_path}")
        print(f"- Output directory: {output_path}")
        print(f"- Scale factor: {scale}")
        if model_path:
            print(f"- Custom model: {model_path}")
        else:
            print("- Using default model")
        
        # Here you would call the actual inference script
        print("\n[This would launch the actual ESRGAN inference]")

def handle_real_esrgan(mode):
    print("\nReal-ESRGAN selected (enhanced practical version)")
    
    if mode == 'train':
        print("\nTraining Real-ESRGAN is more complex and typically requires:")
        print("1. A dataset with diverse high-quality images")
        print("2. Significant computational resources")
        
        proceed = get_yes_no("\nDo you want to proceed with training? [y/n]: ")
        if not proceed:
            return
            
        train_path = get_input_path("Path to training images: ")
        val_path = get_input_path("Path to validation images: ")
        epochs = get_integer("Number of epochs [1000000]: ", 1000000)
        crop_size = get_integer("Batch size [16]: ", 16)
        lr = get_float("Learning rate [0.0001]: ", 0.0001)
        
        print("\nStarting Real-ESRGAN training with:")
        print(f"- Training data: {train_path}")
        print(f"- Validation data: {val_path}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {crop_size}")
        print(f"- Learning rate: {lr}")
        print("\nNote: Real-ESRGAN training typically requires days/weeks on high-end GPUs")
        
        # Here you would call the actual training script
        print("\n[This would launch the actual Real-ESRGAN training]")
        
    else:  # inference
        input_path = get_input_path("Path to input image or directory: ")
        output_path = get_output_path("Path to save output: ")
        scale = get_integer("Scale factor (2, 4) [4]: ", 4)
        face_enhance = get_yes_no("Use face enhancement? [y/n]: ", False)
        
        print("\nRunning Real-ESRGAN inference with:")
        print(f"- Input: {input_path}")
        print(f"- Output directory: {output_path}")
        print(f"- Scale factor: {scale}")
        print(f"- Face enhancement: {'Yes' if face_enhance else 'No'}")
        
        # Here you would call the actual inference script
        print("\n[This would launch the actual Real-ESRGAN inference]")

def main():
    print_welcome()
    gan_choice = get_gan_choice()
    mode = get_mode()
    
    if gan_choice == GANType.SRGAN:
        handle_srgan(mode)
    elif gan_choice == GANType.ESRGAN_PYTORCH:
        handle_esrgan(mode)
    elif gan_choice == GANType.REAL_ESRGAN:
        handle_real_esrgan(mode)
    
    print("\nOperation completed!")

if __name__ == "__main__":
    main()