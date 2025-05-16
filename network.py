import math
import torch
from torch import nn
from torch import functional as F
from torchvision.models.vgg import vgg19

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
        
        # Color-specific loss
        # self.color_loss = nn.L1Loss()
        self.l1_loss = nn.L1Loss()
        
        # TV loss for smoothness
        self.tv_loss = TVLoss()
        
        # Loss weights
        self.content_weight = 0.01 # 0.006 originally
        self.pixel_weight = 1.0
        self.adversarial_weight = 0.05 # 0.001 originally
        self.tv_weight = 2e-
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
        
        # Color consistency loss (in RGB space)
        # Calculate mean color per channel and compare
        # out_mean = torch.mean(out_images, dim=[2, 3])
        # target_mean = torch.mean(target_images, dim=[2, 3])
        # color_loss = self.color_loss(out_mean, target_mean)
        
        # # Combined loss with weights
        # total_loss = (self.pixel_weight * pixel_loss + 
        #               self.adversarial_weight * adversarial_loss + 
        #               self.content_weight * content_loss + 
        #               self.tv_weight * tv_loss +
        #               self.color_weight * color_loss)
        
        total_loss = (
            self.pixel_weight * pixel_loss +
            self.adversarial_weight * adversarial_loss +
            self.content_weight * content_loss +
            self.feature_weight * feature_loss +
            self.tv_weight * tv_loss
        )

        return total_loss

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
        
        # Initial convolution to extract features - keeping RGB channels separate initially
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks with skip connections to maintain spatial information
        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(16)])
        
        # Post-residual convolution
        self.post_res = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling blocks
        upsampling = []
        for _ in range(upsample_block_num):
            upsampling.append(UpsampleBlock(64, 2))
        self.upsampling = nn.Sequential(*upsampling)
        
        # Final convolution to reconstruct RGB image
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Sigmoid()  # Using sigmoid to ensure output is in range [0,1]
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU-based layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store input for global skip connection
        initial_features = self.initial(x)
        
        # Pass through residual blocks
        x = initial_features
        for res_block in self.residual_blocks:
            x = res_block(x)
            
        # Post-residual convolution
        x = self.post_res(x)
        
        # Global skip connection
        x = x + initial_features
        
        # Upsampling
        x = self.upsampling(x)
        
        # Final convolution with sigmoid activation
        x = self.final(x)
        
        return x

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