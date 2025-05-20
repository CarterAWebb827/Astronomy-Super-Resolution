import argparse
import time
import os
from pathlib import Path

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--memory', default=0.9, type=float, help='percentage of GPU usage by program')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
MEMORY = opt.memory

if TEST_MODE:
    if MEMORY > 1.0:
        MEMORY = 1.0
    elif MEMORY < 0.0:
        MEMORY = 0.01
    torch.cuda.set_per_process_memory_fraction(MEMORY)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimizes CUDA operations
    torch.backends.cudnn.enabled = True

base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
output_dir = os.path.join(base, 'output', 'srgan', MODEL_NAME)
os.makedirs(output_dir, exist_ok=True)

# Get input image name and create output filename
input_path = Path(opt.image_name)
output_filename = f"out_srf_{opt.upscale_factor}_{input_path.stem}{input_path.suffix}"
output_path = os.path.join(output_dir, output_filename)

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(f'{base}/models/srgan/epochs/' + MODEL_NAME, weights_only=True))
else:
    model.load_state_dict(torch.load(f'{base}/models/srgan/epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage, weights_only=True))

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

# Time the processing
start = time.time()
out = model(image)
elapsed = time.time() - start
print(f'Processing time: {elapsed:.2f}s')

# Save output
out_img = ToPILImage()(out[0].data.cpu())
out_img.save(output_path)
print(f"Upscaled image saved to: {output_path}")