import argparse
import os

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataset import *
from utils import *

from torchvision.utils import save_image

# os.makedirs('images/training', exist_ok=True)
# os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--batch_size', default=48, type=int, help='batch size of train dataset')
parser.add_argument('--warmup_batches', default=10_000, type=int, help='number of batches with pixel-wise loss only')
parser.add_argument('--n_batches', default=14_000, type=int, help='number of batches of training')
parser.add_argument('--residual_blocks', default=23, type=int, help='number of residual blocks in the generator')
parser.add_argument('--batch', default=0, type=int, help='batch to start training from')
parser.add_argument('--lr', default=0.0002, type=float, help='adam: learning rate')
parser.add_argument('--sample_interval', default=100, type=int, help='interval between saving image samples')
parser.add_argument('--name', default="", type=str, help='name of directory for training')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# print(device)

hr_shape = (opt.crop_size, opt.crop_size)
channels = 3

NAME = opt.name

base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# initialize generator and discriminator  
generator = GeneratorRRDB(channels, num_res_blocks=opt.residual_blocks).to(device)
discriminator = UNetDiscriminatorSN(channels).to(device)
feature_extractor = FeatureExtractor().to(device)

# set feature extractor to inference mode  
feature_extractor.eval()

# Losses  
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

models = os.path.join(base, "models", "real-esrgan", f"{opt.n_batches}")

if NAME == "":
    out_model = models
else:
    out_model = os.path.join(models, NAME)

if not os.path.exists(out_model):
    os.makedirs(out_model)

if opt.batch != 0:
    generator.load_state_dict(torch.load(f'{out_model}/generator_%d.pth' % opt.batch))
    discriminator.load_state_dict(torch.load(f'{out_model}/discriminator_%d.pth' % opt.batch))

# initialize optimzier 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

data = os.path.join(base, "data")

train_set = TrainDatasetFromFolder(f'{data}/train', crop_size=opt.crop_size, upscale_factor=opt.upscale_factor)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True)

# initialize ema  
ema_G = EMA(generator, 0.999)
ema_D = EMA(discriminator, 0.999)
ema_G.register()
ema_D.register()

# -----------------
# Training 
# -----------------

batch = opt.batch
while batch < opt.n_batches:
    for i, (data, target) in enumerate(train_loader):
        batches_done = batch + i

        imgs_lr = data.to(device)
        imgs_hr = target.to(device)

        valid = torch.ones((imgs_lr.size(0), 1, *imgs_hr.shape[-2:]), requires_grad=False).to(device)
        fake = torch.zeros((imgs_lr.size(0), 1, *imgs_hr.shape[-2:]), requires_grad=False).to(device)

        # ---------------------
        # Training Generator
        # ---------------------

        optimizer_G.zero_grad()

        gen_hr = generator(imgs_lr)

        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            loss_pixel.backward()
            optimizer_G.step()
            ema_G.update()
            print(
                '[Iteration %d/%d] [Batch %d/%d] [G pixel: %f]' % 
                (batches_done, opt.n_batches, i, len(train_loader), loss_pixel.item())
            )
            continue
        elif batches_done == opt.warmup_batches:
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)

        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        loss_GAN = (
            criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid) + 
            criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)
        ) / 2

        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        real_features = [real_f.detach() for real_f in real_features]
        loss_content = sum(criterion_content(gen_f, real_f) * w for gen_f, real_f, w in zip(gen_features, real_features, [0.1, 0.1, 1, 1, 1]))

        loss_G = loss_content + 0.1 * loss_GAN + loss_pixel

        loss_G.backward()
        optimizer_G.step()
        ema_G.update()

        # ---------------------
        # Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
        ema_D.update()

        # -------------------------
        # Log Progress
        # -------------------------

        print(
            '[Iteration %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]' % 
            (
                batches_done, 
                opt.n_batches, 
                i, 
                len(train_loader), 
                loss_D.item(), 
                loss_G.item(), 
                loss_content.item(), 
                loss_GAN.item(), 
                loss_pixel.item()
            )
        )

        images_out = os.path.join(base, "training_results", "real-esrgan", str(opt.n_batches))
        os.makedirs(images_out, exist_ok=True)

        if batches_done % opt.sample_interval == 0:
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4, mode='bicubic')
            img_grid = torch.clamp(torch.cat((imgs_lr, gen_hr, imgs_hr), -1), min=0, max=1)
            save_image(img_grid, f'{images_out}/%d.png' % batches_done, nrow=1, normalize=False)
        
    batch = batches_done + 1

    ema_G.apply_shadow()
    ema_D.apply_shadow()

    torch.save(generator.state_dict(), f'{out_model}/generator_%d.pth' % batch)
    torch.save(discriminator.state_dict(), f'{out_model}/discriminator_%d.pth' % batch)

    ema_G.restore()
    ema_D.restore()