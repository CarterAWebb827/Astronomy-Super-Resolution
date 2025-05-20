import argparse
import os
import gc

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from model import Generator

def process_video_frame_by_frame():
    parser = argparse.ArgumentParser(description='Test Single Video')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--video_name', type=str, required=True, help='test low resolution video name')
    parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
    parser.add_argument('--chunk_size', default=10, type=int, help='number of frames to process at once')
    parser.add_argument('--batch_size', default=4, type=int, help='sub-batch size for GPU processing')
    opt = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    # Load model
    model = Generator(opt.upscale_factor).eval().to(device)
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'models', 'srgan', 'epochs', opt.model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Video setup
    if not os.path.exists(opt.video_name):
        raise FileNotFoundError(f"Video file {opt.video_name} not found")
    
    cap = cv2.VideoCapture(opt.video_name)
    if not cap.isOpened():
        raise IOError(f"Could not open video {opt.video_name}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video writer
    sr_width, sr_height = width * opt.upscale_factor, height * opt.upscale_factor
    output_name = f'sr_{os.path.basename(opt.video_name)}'
    sr_writer = cv2.VideoWriter(
        output_name, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, 
        (sr_width, sr_height)
    )

    # Frame processing loop
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=frame_count, desc='Processing video')
    
    while True:
        frames = []
        for _ in range(opt.chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if not frames:
            break
            
        # Process frames in batch
        with torch.no_grad():
            # Convert frames to tensors properly
            input_tensors = []
            for frame in frames:
                # Convert BGR to RGB and to PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                tensor = ToTensor()(pil_image)
                input_tensors.append(tensor)
            
            inputs = torch.stack(input_tensors).to(device)
            
            # Process in sub-batches
            outputs = []
            for i in range(0, len(inputs), opt.batch_size):
                batch = inputs[i:i+opt.batch_size]
                out = model(batch)
                outputs.append(out.cpu())
                torch.cuda.empty_cache()
            
            sr_frames = torch.cat(outputs)
        
        # Save frames
        for frame_tensor in sr_frames:
            # Convert tensor to numpy array
            frame_np = frame_tensor.clamp(0, 1).numpy()
            frame_np = np.transpose(frame_np, (1, 2, 0)) * 255
            frame_np = frame_np.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            sr_writer.write(frame_bgr)
        
        progress.update(len(frames))
        del inputs, outputs, sr_frames, input_tensors
        gc.collect()
    
    # Cleanup
    cap.release()
    sr_writer.release()
    progress.close()
    print(f"Video processing complete. Saved to {output_name}")

if __name__ == "__main__":
    process_video_frame_by_frame()