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
    print("1. SRGAN (original implementation)")
    print("2. ESRGAN (PyTorch implementation)")
    print("3. Real-ESRGAN (enhanced practical version)")
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
        path = input(prompt)
        if os.path.exists(path):
            return path
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
        train_path = get_input_path("Path to training images: ")
        val_path = get_input_path("Path to validation images: ")
        epochs = get_integer("Number of epochs [100]: ", 100)
        batch_size = get_integer("Batch size [16]: ", 16)
        lr = get_float("Learning rate [0.0001]: ", 0.0001)
        
        print("\nStarting SRGAN training with:")
        print(f"- Training data: {train_path}")
        print(f"- Validation data: {val_path}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        print(f"- Learning rate: {lr}")
        
        # Here you would normally call the actual training script
        # For example:
        # subprocess.run(["python", "SRGAN/train.py", "--train", train_path, ...])
        print("\n[This would launch the actual SRGAN training]")
        
    else:  # inference
        input_path = get_input_path("Path to input image or directory: ")
        output_path = get_output_path("Path to save output: ")
        scale = get_integer("Scale factor (2, 4, or 8) [4]: ", 4)
        
        print("\nRunning SRGAN inference with:")
        print(f"- Input: {input_path}")
        print(f"- Output directory: {output_path}")
        print(f"- Scale factor: {scale}")
        
        # Here you would normally call the actual inference script
        print("\n[This would launch the actual SRGAN inference]")

def handle_esrgan_pytorch(mode):
    print("\nESRGAN (PyTorch implementation) selected")
    
    if mode == 'train':
        print("\nTraining ESRGAN requires the following:")
        train_path = get_input_path("Path to HR training images: ")
        val_path = get_input_path("Path to HR validation images: ")
        lr_path = get_input_path("Path to LR training images (leave empty to generate): ", "")
        epochs = get_integer("Number of epochs [500]: ", 500)
        batch_size = get_integer("Batch size [16]: ", 16)
        lr = get_float("Learning rate [0.0002]: ", 0.0002)
        
        print("\nStarting ESRGAN training with:")
        print(f"- HR training data: {train_path}")
        if lr_path:
            print(f"- LR training data: {lr_path}")
        else:
            print("- LR images will be generated automatically")
        print(f"- HR validation data: {val_path}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
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
        batch_size = get_integer("Batch size [16]: ", 16)
        lr = get_float("Learning rate [0.0001]: ", 0.0001)
        
        print("\nStarting Real-ESRGAN training with:")
        print(f"- Training data: {train_path}")
        print(f"- Validation data: {val_path}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
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
        handle_esrgan_pytorch(mode)
    elif gan_choice == GANType.REAL_ESRGAN:
        handle_real_esrgan(mode)
    
    print("\nOperation completed!")

if __name__ == "__main__":
    main()