import os
import opendatasets as od
import torch
import warnings
from torch import optim
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, ToPILImage
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.amp import GradScaler, autocast

from dataset import ImprovedTrainDatasetFromFolder
from network import ModelType, create_generator, create_discriminator, create_generator_loss

from enum import Enum

warnings.filterwarnings("ignore", message="Attempting to use hipBLASLt on an unsupported architecture!")

def set_model_type(model_type):
    for model in ModelType:
        if model.value == model_type:
            return model
    
    return ModelType.SRGAN

class DataDir(Enum):
    APOC = "astronomy-picture-of-the-day/APOC"
    APOC64 = "astronomy-picture-of-the-day/APOC64"
    WEBB = "webb-hubble-pictures"

def set_data_dir(data_dir):
    if data_dir == "apoc":
        DIR = "apoc"
        return DataDir.APOC
    elif data_dir == "64":
        DIR = "apoc64"
        return DataDir.APOC64
    else:
        DIR = "webb"
        return DataDir.WEBB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=UserWarning, message="MIOpen.*")

train_data_dir = "webb-hubble-pictures"
test_data_dir = "astronomy-picture-of-the-day/APOC"

Image.MAX_IMAGE_PIXELS = None
UPSCALE_FACTOR = 4
CROP_SIZE = UPSCALE_FACTOR * 32
N_EPOCHS = 150

def get_data():
    # Set the correct permissions for the file (Unix-based systems)
    os.chmod("kaggle.json", 0o600)

    # Create the .kaggle directory if it doesn"t exist
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # Check if the datasets already exist before downloading
    if not os.path.exists(test_data_dir):
        # Define the dataset
        od.download(
            "https://www.kaggle.com/datasets/melcore/astronomy-picture-of-the-day")
    else:
        print(f"\nThe dataset already exists in the {test_data_dir} folder.\nNo download needed.")

def compute_gradient_penalty(D, real_samples, fake_samples):
    # Calculates the gradient penalty loss for WGAN-GP"
    
    # Random weight term for interpolation
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(real_samples.device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Forward pass
    d_interpolates = D(interpolates)
    
    # Get gradient w.r.t. interpolates
    fake = torch.ones(d_interpolates.size()).to(real_samples.device)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def Train(N_EPOCHS, train_loader, netG, netD, optimizerG, optimizerD, generator_criterion, model_type, DIR):
    # Initialize history trackers
    history = {
        'epoch': [],
        'd_loss': [],
        'g_loss': [],
        'd_score': [],
        'g_score': []
    }

    best_g_loss = float("inf")
    epochs_without_improvement = 0
    patience = 10
    early_stop = False

    # Gradient accumulation steps
    accumulation_steps = 4
    d_accumulation_steps = 2
    
    # Mixed precision training
    scalerG = GradScaler()
    scalerD = GradScaler()

    # Learning rate schedulers
    schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerG, 'min', factor=0.5, patience=5, verbose=True)
    schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerD, 'min', factor=0.5, patience=5, verbose=True)

    # Training balance parameters
    d_update_freq = 1  # Start with updating D every step
    g_update_freq = 1  # Start with updating G every step
    min_d_update_freq = 1
    max_d_update_freq = 5
    min_g_update_freq = 1
    max_g_update_freq = 3

    # Adaptive training threshold
    d_score_target = 0.5  # Ideal discriminator accuracy
    d_score_tolerance = 0.2  # Allowed deviation from target
    
    # Check initial weights
    for name, param in netG.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in generator {name} at initialization!")
            param.data[torch.isnan(param.data)] = 0

    for epoch in range(1, N_EPOCHS + 1):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        train_bar = tqdm(train_loader)
        running_results = {
            "batch_sizes": 0,
            "d_loss": 0,
            "g_loss": 0,
            "d_score": 0,
            "g_score": 0
        }

        netG.train()
        netD.train()
        
        # Track gradient accumulation
        g_accum, d_accum = 0, 0

        for i, (data, target) in enumerate(train_bar):
            batch_size = data.size(0)
            running_results["batch_sizes"] += batch_size

            real_img = target.to(device)
            z = data.to(device)

            # --- Discriminator Update ---
            if i % d_update_freq == 0:  # Only update D according to current frequency
                optimizerD.zero_grad()
                
                with autocast("cuda"):
                    # Forward pass with stability checks
                    fake_img = netG(z).detach()
                    fake_img = torch.clamp(fake_img, 0, 1)  # Ensure valid range
                    
                    real_out = netD(real_img)
                    fake_out = netD(fake_img)
                    
                    # Stable loss calculation
                    if model_type == ModelType.SRGAN:
                        real_loss = torch.mean(torch.square(real_out - 1 + 1e-8))
                        fake_loss = torch.mean(torch.square(fake_out + 1e-8))
                        d_loss = (real_loss + fake_loss) / 2
                    else:
                        # ESRGAN with more stable relativistic loss
                        diff_real = real_out - torch.mean(fake_out) - 1
                        diff_fake = fake_out - torch.mean(real_out) + 1
                        d_loss_real = torch.mean(torch.square(diff_real.clamp(min=-10, max=10) + 1e-8))
                        d_loss_fake = torch.mean(torch.square(diff_fake.clamp(min=-10, max=10) + 1e-8))
                        d_loss = (d_loss_real + d_loss_fake) / 2
                
                # Scale loss and backpropagate
                scalerD.scale(d_loss / d_accumulation_steps).backward()
                d_accum += 1

                if d_accum % d_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # Gradient clipping with stability checks
                    scalerD.unscale_(optimizerD)
                    torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=0.5)  # Reduced from 1.0
                    
                    # Check for NaN gradients
                    for param in netD.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            param.grad[torch.isnan(param.grad)] = 0
                    
                    scalerD.step(optimizerD)
                    scalerD.update()
                    optimizerD.zero_grad()
                    d_accum = 0

            # --- Generator Update ---
            if i % g_update_freq == 0:  # Only update G according to current frequency
                optimizerG.zero_grad()
                
                with autocast("cuda"):
                    # Forward pass with stability checks
                    fake_img = netG(z)
                    fake_img = torch.clamp(fake_img, 0, 1)  # Ensure valid range
                    
                    fake_out = netD(fake_img)
                    
                    # Stable loss calculation
                    if model_type == ModelType.SRGAN:
                        g_loss = generator_criterion(fake_out, fake_img, real_img)
                    else:
                        with torch.no_grad():
                            real_out_for_g = netD(real_img)
                        diff_adv = fake_out - torch.mean(real_out_for_g) - 1
                        g_loss_adv = torch.mean(torch.square(diff_adv.clamp(min=-10, max=10) + 1e-8))
                        g_loss = generator_criterion(g_loss_adv, fake_img, real_img)
                
                # Scale loss and backpropagate
                scalerG.scale(g_loss / accumulation_steps).backward()
                g_accum += 1

                if g_accum % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # Gradient clipping with stability checks
                    scalerG.unscale_(optimizerG)
                    torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
                    
                    # Check for NaN gradients
                    for param in netG.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            param.grad[torch.isnan(param.grad)] = 0
                    
                    scalerG.step(optimizerG)
                    scalerG.update()
                    optimizerG.zero_grad()
                    g_accum = 0

            # Update running results with stability checks
            running_results["g_loss"] += g_loss.item() * batch_size if not torch.isnan(g_loss) else 0
            running_results["d_loss"] += d_loss.item() * batch_size if not torch.isnan(d_loss) else 0
            running_results["g_score"] += real_out.mean().item() * batch_size if not torch.isnan(real_out.mean()) else 0
            running_results["d_score"] += real_out.mean().item() * batch_size if not torch.isnan(real_out.mean()) else 0

            # Adaptive frequency adjustment
            current_d_score = running_results["d_score"] / (running_results["batch_sizes"] or 1)
            if current_d_score > (d_score_target + d_score_tolerance):  # D is too strong
                d_update_freq = min(d_update_freq + 1, max_d_update_freq)
                g_update_freq = max(g_update_freq - 1, min_g_update_freq)
                print(f"Adjusting: D too strong (score={current_d_score:.2f}). "
                      f"Now updating D every {d_update_freq} steps, G every {g_update_freq} steps")
            elif current_d_score < (d_score_target - d_score_tolerance):  # G is too strong
                d_update_freq = max(d_update_freq - 1, min_d_update_freq)
                g_update_freq = min(g_update_freq + 1, max_g_update_freq)
                print(f"Adjusting: G too strong (score={current_d_score:.2f}). "
                      f"Now updating D every {d_update_freq} steps, G every {g_update_freq} steps")

            # Update progress bar
            train_bar.set_description(desc="[%d/%d] Loss_D: %.4f, Loss_G: %.4f, D(x): %.4f, D(G(z)): %.4f" % (
                epoch, N_EPOCHS, 
                running_results["d_loss"] / (running_results["batch_sizes"] or 1),
                running_results["g_loss"] / (running_results["batch_sizes"] or 1),
                running_results["d_score"] / (running_results["batch_sizes"] or 1),
                running_results["g_score"] / (running_results["batch_sizes"] or 1)
            ))
            
            # Clear memory and check for NaNs
            if i % 50 == 0:
                torch.cuda.empty_cache()
                
                # Check model weights for NaNs
                for name, param in netG.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN detected in generator {name}")
                        param.data[torch.isnan(param.data)] = 0
                
                for name, param in netD.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN detected in discriminator {name}")
                        param.data[torch.isnan(param.data)] = 0
        
        # Calculate epoch metrics
        epoch_d_loss = running_results["d_loss"] / running_results["batch_sizes"]
        epoch_g_loss = running_results["g_loss"] / running_results["batch_sizes"]
        epoch_d_score = running_results["d_score"] / running_results["batch_sizes"]
        epoch_g_score = running_results["g_score"] / running_results["batch_sizes"]

        # Update LR schedulers
        schedulerG.step(epoch_g_loss)
        schedulerD.step(epoch_d_loss)

        # Update history
        history['epoch'].append(epoch)
        history['d_loss'].append(epoch_d_loss)
        history['g_loss'].append(epoch_g_loss)
        history['d_score'].append(epoch_d_score)
        history['g_score'].append(epoch_g_score)

        # Evaluate and save model
        netG.eval()
        
        # Create output directories
        out_path = os.path.join(f"training_results/{DIR}/{model_type.value}/", f"{N_EPOCHS}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        # Save checkpoint based on frequency
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            image_path = os.path.join(out_path, f"images/{epoch}")
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            model_save = os.path.join(out_path, f"models/{epoch}")
            if not os.path.exists(model_save):
                os.makedirs(model_save)
            
            # Generate and save sample images
            with torch.no_grad():
                for i, (lr, hr) in enumerate(train_loader):
                    if i > 5:  # Limit to a few samples
                        break
                    lr = lr.to(device)
                    hr = hr.to(device)
                    sr = netG(lr)
                    
                    # Save individual images
                    save_image(sr.cpu(), os.path.join(image_path, f"SR_{i}_epoch_{epoch}.png"))
                    save_image(hr.cpu(), os.path.join(image_path, f"HR_{i}_epoch_{epoch}.png"))
                    save_image(lr.cpu(), os.path.join(image_path, f"LR_{i}_epoch_{epoch}.png"))
                    
                    # Create composite image (LR, SR, HR) side by side
                    composite = torch.cat([
                        torch.nn.functional.interpolate(lr.cpu(), scale_factor=UPSCALE_FACTOR, mode='nearest'),
                        sr.cpu(),
                        hr.cpu()
                    ], dim=3)  # Concatenate along width
                    save_image(composite, os.path.join(image_path, f"comparison_{i}_epoch_{epoch}.png"))

            # Save model checkpoints
            torch.save(netG.state_dict(), os.path.join(model_save, f"generator_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(model_save, f"discriminator_epoch_{epoch}.pth"))

            # Plot and save training curves
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.plot(history['epoch'], history['d_loss'], label='Discriminator Loss')
            plt.plot(history['epoch'], history['g_loss'], label='Generator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['epoch'], history['d_score'], label='D(x)')
            plt.plot(history['epoch'], history['g_score'], label='D(G(z))')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Discriminator Scores')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(image_path, f"training_curves_epoch_{epoch}.png"))
            plt.close()

        # Save training log
        results = {
            "epoch": epoch,
            "d_loss": epoch_d_loss,
            "g_loss": epoch_g_loss,
            "d_score": epoch_d_score,
            "g_score": epoch_g_score
        }
        df = pd.DataFrame([results])
        df.to_csv(os.path.join(out_path, "training_log.csv"), 
                mode="a", 
                header=not os.path.exists(os.path.join(out_path, "training_log.csv")))
        
        # Check for generator improvement
        if epoch_g_loss < best_g_loss:
            best_g_loss = epoch_g_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(netG.state_dict(), os.path.join(out_path, "generator_best.pth"))
            torch.save(netD.state_dict(), os.path.join(out_path, "discriminator_best.pth"))
        else:
            epochs_without_improvement += 1
            
        # Check for potential discriminator collapse
        if epoch_d_loss < 0.01 and epoch > 20:
            print(f"Warning: Possible discriminator collapse (Loss_D: {epoch_d_loss:.4f})")
            
        # Check for significant imbalance
        if epoch_d_loss < 0.01 and epoch_g_loss > 10:
            print(f"Early stopping: Severe imbalance detected (D: {epoch_d_loss:.4f}, G: {epoch_g_loss:.4f})")
            early_stop = True
            
        # Check patience period
        if epochs_without_improvement >= patience:
            print(f"Early stopping: No improvement for {patience} epochs")
            early_stop = True

def Test(upscale_factor=4, model_type=ModelType.SRGAN):
    discriminator_path = f"training_results/{model_type.value}/discriminator_best.pth"
    netD = create_discriminator().to(device)
    netD.load_state_dict(torch.load(discriminator_path, weights_only=True))

    netD.eval()

    # Load your generator (if testing fake images)
    generator_path = f"training_results/{model_type.value}/generator_best.pth"
    netG = create_generator(UPSCALE_FACTOR).to(device)
    netG.load_state_dict(torch.load(generator_path, weights_only=True))
    netG.eval()

    # Create a test dataset (adjust paths as needed)
    test_dataset = ImprovedTrainDatasetFromFolder("astronomy-picture-of-the-day/APOC", 
                                        crop_size=CROP_SIZE, 
                                        upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    real_preds = []
    fake_preds = []

    with torch.no_grad():
        # Test on real images
        for data, target in test_loader:
            real_images = target.to(device)
            real_preds.append(netD(real_images).cpu().numpy())
        
        # Test on fake images
        for data, _ in test_loader:
            lr_images = data.to(device)
            fake_images = netG(lr_images)
            fake_preds.append(netD(fake_images).cpu().numpy())

    # Concatenate predictions
    real_preds = np.concatenate(real_preds)
    fake_preds = np.concatenate(fake_preds)

    # Ground truth labels (1 for real, 0 for fake)
    y_true = np.concatenate([np.ones_like(real_preds), np.zeros_like(fake_preds)])
    y_scores = np.concatenate([real_preds, fake_preds])

    # Compute metrics
    auc = roc_auc_score(y_true, y_scores)
    precision = precision_score(y_true, y_scores > 0.5)
    recall = recall_score(y_true, y_scores > 0.5)

    print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # def plot_discriminator_response(image, label):
    #     with torch.no_grad():
    #         output = netD(image.unsqueeze(0).to(device)).item()
        
    #     display_image = image.detach().cpu().permute(1, 2, 0).numpy()
    #     plt.imshow(display_image)
    #     plt.title(f"Label: {'Real' if label == 1 else 'Fake'}\nD(x) = {output:.4f}")
    #     plt.axis('off')
    #     plt.show()

    # # Test on a real and fake image
    # real_image, _ = next(iter(test_loader))
    # fake_image = netG(real_image.to(device)).cpu()

    # plot_discriminator_response(real_image[0], label=1)
    # plot_discriminator_response(fake_image[0], label=0)

def Upscale(input_path, output_path, model_path, upscale_factor=4, model_type=ModelType.SRGAN, up_dir=DataDir.APOC):
    if up_dir == DataDir.APOC64:
        input_path = input_path[1]
    else:
        input_path = input_path[0]

    # Load model
    netG = create_generator(upscale_factor, model_type).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    netG.eval()
    
    # Preprocess
    img = Image.open(input_path)
    # Convert grayscale to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img.save(os.path.join(output_path, f"ORIGINAL.png"))

    bicubic_upscaled = img.resize(
        (img.width * UPSCALE_FACTOR, img.height * UPSCALE_FACTOR),
        Image.BICUBIC
    )
    bicubic_upscaled.save(os.path.join(output_path, f"BICUBIC.png"))
    print(f"Bicubic image saved to {output_path}")
    
    # Convert to tensor and normalize
    transform = Compose([
        ToTensor(), # Normalizes to [0, 1]
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Upscale
    with torch.no_grad():
        output_tensor = netG(input_tensor)
        
        # Ensure output is in valid range [0,1]
        output_tensor = torch.clamp(output_tensor, 0, 1)

    # Convert back to PIL Image
    output_image = ToPILImage()(output_tensor.squeeze(0).cpu())
    
    # Save
    output_image.save(os.path.join(output_path, f"GAN.png"))
    print(f"GAN image saved to {output_path}")

def find_file_by_name(search_dir, target_filename, case_sensitive=False):
    matches = []
    
    if not case_sensitive:
        target_filename = target_filename.lower()
    
    for root, dirs, files in os.walk(search_dir):
        for filename in files:
            compare_name = filename if case_sensitive else filename.lower()
            if compare_name == target_filename:
                matches.append(os.path.join(root, filename))
    
    return matches

def main():
    get_data()

    # UPSCALE_FACTOR = int(input("\nHow much would you like to upscale an image by?: "))
    # CROP_SIZE = UPSCALE_FACTOR * 22
    N_EPOCHS = int(input("\nHow many epochs would you like to have?: "))
    model_type = input("\nWhich model would you like to train? (srgan/esrgan): ").lower()
    data_dir = input("\nWhich data directory would you like to use for training? (apoc, 64, webb): ").lower()
    up_dir = input("\nWhich data directory would you like to use for upscaling? (apoc, 64): ").lower()
    model_type = set_model_type(model_type)
    data_dir = set_data_dir(data_dir)
    up_dir = set_data_dir(up_dir)

    train_data = ImprovedTrainDatasetFromFolder(data_dir.value, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(train_data, batch_size=64, num_workers=0, shuffle=True)

    netG = create_generator(UPSCALE_FACTOR, model_type).to(device)
    netD = create_discriminator(model_type).to(device)

    generator_criterion = create_generator_loss(device, model_type).to(device)

    # In your main() function:
    if model_type == ModelType.ESRGAN:
        # More stable settings for ESRGAN
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.99))
        optimizerD = optim.Adam(netD.parameters(), lr=4e-4, betas=(0.9, 0.99))
        
        # Smaller batch sizes work better
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    else:
        optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.9, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.9, 0.999))

    results = {
        "d_loss": [],
        "g_loss": [],
        "d_score": [],
        "g_score": []
    }

    if data_dir == DataDir.APOC64:
        DIR = "apoc64"
    else:
        if data_dir == DataDir.APOC:
            DIR = "apoc"
        else:
            DIR = "webb"

    Train(N_EPOCHS, train_loader, netG, netD, optimizerG, optimizerD, generator_criterion, model_type, DIR)

    # Test()

    inp = input("\nWhat is the name of the image you would like to upscale?: ")
    input_file = find_file_by_name(os.path.abspath(""), inp)
    if up_dir == DataDir.APOC64:
        UP_DIR = "apoc64"
    else:
        if up_dir == DataDir.APOC:
            UP_DIR = "apoc"
        else:
            UP_DIR = "webb"
    out = os.path.join("output/", f"{model_type.value}/{UP_DIR}/{inp}")
    if not os.path.exists(out):
        os.makedirs(out)

    Upscale(input_file, out, f"training_results/{DIR}/{model_type.value}/{N_EPOCHS}/generator_best.pth", model_type=model_type, up_dir=up_dir)

if __name__ == "__main__":
    main()