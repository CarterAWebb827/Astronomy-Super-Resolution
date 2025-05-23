import argparse
import os

import torch
from PIL import Image
from pathlib import Path
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor

from model import GeneratorRRDB

# Set memory optimization flags
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--test_mode', default='GPU', choices=['CPU', 'GPU'], help='Using CPU or GPU')
parser.add_argument('--n_batch', type=int, default=1200)
parser.add_argument('--image_name', required=True, help='Input image path')
parser.add_argument('--model_name', default='generator_720.pth', help='Generator model name')
parser.add_argument('--tile_size', type=int, default=512, help='Tile size for processing')
parser.add_argument('--overlap', type=int, default=32, help='Overlap between tiles')
args = parser.parse_args()

# Configuration
base = Path(__file__).parents[2]
N_BATCH = args.n_batch
model_path = base / "models" / "real-esrgan" / str(N_BATCH) / args.model_name
output_dir = base / "output" / "real-esrgan" / str(N_BATCH) / args.model_name
output_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = GeneratorRRDB(channels=3, num_res_blocks=23).eval()
if args.test_mode == 'GPU':
    model = model.cuda()
    state_dict = torch.load(model_path, weights_only=False)
else:
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict)

def process_large_image(model, image_path, scale=4, tile_size=512, overlap=32):
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
                if args.test_mode:
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
    
    return out_img

# Process image
try:
    if Path(args.image_name).stat().st_size > 50 * 1024 * 1024:  # >50MB
        print("Processing large image with tiling...")
        result = process_large_image(
            model, 
            args.image_name,
            tile_size=args.tile_size,
            overlap=args.overlap
        )
    else:
        print("Processing normally...")
        with torch.no_grad():
            img = Image.open(args.image_name).convert('RGB')
            tensor = ToTensor()(img).unsqueeze(0)
            if args.test_mode == 'GPU':
                tensor = tensor.cuda()
            result = ToPILImage()(model(tensor)[0].cpu().clamp(0, 1))
    
    # Save result
    output_path = output_dir / f"out_{Path(args.image_name).name}"
    result.save(output_path)
    print(f"Saved result to {output_path}")

except RuntimeError as e:
    print(f"Memory error: {e}")
    print("Try reducing tile size with --tile_size 256 or --tile_size 128")