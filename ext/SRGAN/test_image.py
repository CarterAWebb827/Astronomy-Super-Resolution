import argparse
import time
import os
from pathlib import Path

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

# Set memory optimization flags
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--memory', default=0.9, type=float, help='percentage of GPU usage by program')
parser.add_argument('--direc', default="", type=str, help='name of directory for model')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
MEMORY = opt.memory
DIREC = opt.direc

if TEST_MODE:
    if MEMORY > 1.0:
        MEMORY = 1.0
    elif MEMORY < 0.0:
        MEMORY = 0.01
    torch.cuda.set_per_process_memory_fraction(MEMORY)
    torch.cuda.empty_cache()

    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"ROCm available: {torch.version.hip is not None}")  # Should be True for AMD

base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if DIREC == "":
    output_dir = os.path.join(base, 'output', 'srgan', MODEL_NAME)
else:
    output_dir = os.path.join(base, 'output', 'srgan', DIREC, MODEL_NAME)
os.makedirs(output_dir, exist_ok=True)

# Get input image name and create output filename
input_path = Path(opt.image_name)
output_filename = f"out_srf_{opt.upscale_factor}_{input_path.stem}{input_path.suffix}"
output_path = os.path.join(output_dir, output_filename)

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    if DIREC == "":
        model.load_state_dict(torch.load(f'{base}/models/srgan/epochs/' + MODEL_NAME, weights_only=False))
    else:
        model.load_state_dict(torch.load(f'{base}/models/srgan/epochs/{DIREC}/' + MODEL_NAME, weights_only=False))
else:
    if DIREC == "":
        model.load_state_dict(torch.load(f'{base}/models/srgan/epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage, weights_only=False))
    else:
        model.load_state_dict(torch.load(f'{base}/models/srgan/epochs/{DIREC}/' + MODEL_NAME, map_location=lambda storage, loc: storage, weights_only=False))

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

def process_large_image(model, image_path, scale=4, tile_size=64, overlap=32):
    torch.backends.cudnn.benchmark = True

    # Process large images in tiles to avoid memory issues
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    tile_size = min(tile_size, width, height)
    
    # Create output canvas
    out_img = Image.new('RGB', (width*scale, height*scale))
    
    # Process each tile
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Get tile with overlap
            box = (x, y, 
                   min(x + tile_size, width), 
                   min(y + tile_size, height))
            tile = img.crop(box)
            
            # Process tile
            with torch.no_grad():
                input_tensor = ToTensor()(tile).unsqueeze(0)
                if opt.test_mode:
                    input_tensor = input_tensor.cuda()
                
                # Process with model
                output = model(input_tensor)
                output = torch.clamp(output, 0, 1)
                out_tile = ToPILImage()(output[0].cpu())
            
            # Paste into output image with overlap handling
            out_box = (x*scale, y*scale, 
                       (x + tile_size)*scale, 
                       (y + tile_size)*scale)
            out_img.paste(out_tile, (x*scale, y*scale))
            
            # Clean up
            del input_tensor, output, out_tile
            torch.cuda.empty_cache()
    
    torch.backends.cudnn.benchmark = False

    return out_img

try:
    print("Processing normally...")
    with torch.no_grad():
        img = Image.open(opt.image_name).convert('RGB')
        tensor = ToTensor()(img).unsqueeze(0)
        if opt.test_mode == 'GPU':
            tensor = tensor.cuda()
        result = ToPILImage()(model(tensor)[0].cpu().clamp(0, 1))
    
    # Save result
    output_path = output_dir + "/" + f"out_{Path(opt.image_name).name}"
    result.save(output_path)
    print(f"Saved result to {output_path}")
except RuntimeError as e:
    print(f"Memory error: {e}")
    print("Processing large image with tiling...")
    result = process_large_image(
        model, 
        opt.image_name,
        # tile_size=opt.tile_size,
        # overlap=opt.overlap
    )

    output_path = output_dir + "/" + f"out_{Path(opt.image_name).name}"
    result.save(output_path)
    print(f"Saved result to {output_path}")

# Time the processing
# start = time.time()
# out = model(image)
# elapsed = time.time() - start
# print(f'Processing time: {elapsed:.2f}s')

# Save output
# out_img = ToPILImage()(out[0].data.cpu())
# out_img.save(output_path)
# print(f"Upscaled image saved to: {output_path}")