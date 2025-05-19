import math
import torch
from torch import nn
from torchvision.models.vgg import vgg19
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from enum import Enum

class ModelType(Enum):
    SRGAN = "srgan"
    ESRGAN = "esrgan"

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()

        self.tv_loss_weight = tv_loss_weight
    
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()

        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class GeneratorLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(GeneratorLoss, self).__init__()

        # Content loss - perceptual loss using network
        vgg = vgg19(weights="DEFAULT")
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.to(device)
        
        # Pixel-wise loss
        self.mse_loss = nn.MSELoss()
        
        # L1 Loss
        self.l1_loss = nn.L1Loss()
        
        # TV loss for smoothness
        self.tv_loss = TVLoss()
        
        # Loss weights
        self.content_weight = 0.01 # 0.006 originally
        self.pixel_weight = 1.0
        self.adversarial_weight = 0.05 # 0.001 originally
        self.tv_weight = 2e-6
        self.feature_weight = 0.1

    def forward(self, out_labels, out_images, target_images):
        # Adversarial loss
        adversarial_loss = torch.mean(1 - out_labels)
        
        # Content/perceptual loss using VGG features
        out_features = self.loss_network(out_images)
        target_features = self.loss_network(target_images)
        content_loss = self.mse_loss(out_features, target_features)

        # Feature matching loss
        feature_loss = self.l1_loss(out_features, target_features)
        
        # Pixel-wise loss
        pixel_loss = self.mse_loss(out_images, target_images)
        
        # TV loss
        tv_loss = self.tv_loss(out_images)
        
        total_loss = (
            self.pixel_weight * pixel_loss +
            self.adversarial_weight * adversarial_loss +
            self.content_weight * content_loss +
            self.feature_weight * feature_loss +
            self.tv_weight * tv_loss
        )

        return total_loss

class ESRGAN_GeneratorLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(ESRGAN_GeneratorLoss, self).__init__()
        
        # VGG19 for perceptual loss (before activation)
        vgg = vgg19(weights="DEFAULT")
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.to(device)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Loss weights
        self.pixel_weight = 1.0
        self.perceptual_weight = 1.0
        self.adversarial_weight = 0.005
    
    def forward(self, out_labels, out_images, target_images):
        # Perceptual loss (before activation)
        out_features = self.loss_network(out_images)
        target_features = self.loss_network(target_images.detach())
        perceptual_loss = self.l1_loss(out_features, target_features)
        
        # Pixel loss
        pixel_loss = self.l1_loss(out_images, target_images)
        
        # Relativistic adversarial loss
        adversarial_loss = -torch.mean(torch.log(out_labels + 1e-8))
        
        # Total loss
        total_loss = (self.pixel_weight * pixel_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.adversarial_weight * adversarial_loss)
        
        return total_loss

def create_generator(scale_factor, model_type=ModelType.SRGAN):
    if model_type == ModelType.ESRGAN:
        return ESRGAN_Generator(scale_factor)
    return Generator(scale_factor)

def create_discriminator(model_type=ModelType.SRGAN, height=1, width=1):
    if model_type == ModelType.ESRGAN:
        # hr_shape = (opt.hr_height, opt.hr_width)
        return ESRGAN_Discriminator(input_shape=(3, height, width))
    return Discriminator()

def create_generator_loss(device="cuda", model_type=ModelType.SRGAN):
    if model_type == ModelType.ESRGAN:
        return ESRGAN_GeneratorLoss(device)
    return GeneratorLoss(device)

# SRGAN

class ResidualBlock(nn.Module):
    def __init__(self, channels=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()

        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size()[0]

        return torch.sigmoid(self.net(x).view(batch_size))

# ESRGAN

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    # Residual in Residual Dense Block
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class ESRGAN_Generator(nn.Module):
    def __init__(self, scale_factor, num_blocks=23, num_feat=64, num_grow_ch=32):
        super(ESRGAN_Generator, self).__init__()
        self.scale_factor = scale_factor
        
        # First convolution
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        
        # RRDB blocks
        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(RRDB(num_feat, num_grow_ch))
        
        # LR conv
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        if scale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif scale_factor == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif scale_factor == 8:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Final convolution
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        
    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = feat.clone()
        
        for block in self.body:
            body_feat = block(body_feat)
        
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        out = self.upsample(feat)
        out = self.conv_last(out)
        
        return torch.sigmoid(out)

class ESRGAN_Discriminator(nn.Module):
    def __init__(self):
        super(ESRGAN_Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(64, 64, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(64, 128, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(128, 128, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(128, 256, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(256, 256, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(256, 512, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(512, 512, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))