import torch
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torch.utils.data import Dataset
from PIL import Image
from os import listdir
from os.path import join

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

class ImprovedTrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(ImprovedTrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor
        
    def __getitem__(self, index):
        # Load image
        try:
            hr_image = Image.open(self.image_filenames[index])
            
            # Convert grayscale to RGB if needed
            if hr_image.mode != "RGB":
                hr_image = hr_image.convert("RGB")
                
            # Apply high-resolution transform
            hr_image = self.hr_transform(hr_image)
            
            # Generate low-resolution image
            # Using a fixed downsampling method for consistency
            with torch.no_grad():
                lr_image = torch.nn.functional.interpolate(
                    hr_image.unsqueeze(0),
                    scale_factor=1.0/self.upscale_factor,
                    mode='bicubic',
                    align_corners=False
                ).squeeze(0)
                
                # Ensure values are in valid range [0,1]
                lr_image = torch.clamp(lr_image, 0, 1)
                hr_image = torch.clamp(hr_image, 0, 1)
            
            return lr_image, hr_image
            
        except Exception as e:
            print(f"Error loading image {self.image_filenames[index]}: {e}")
            # Return a small valid tensor pair as fallback
            return torch.zeros(3, 32, 32), torch.zeros(3, 32*self.upscale_factor, 32*self.upscale_factor)

    def __len__(self):
        return len(self.image_filenames)