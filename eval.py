import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import tempfile
from scipy.stats import pearsonr, kendalltau
import torch
from torch import device
from lpips import LPIPS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SREvaluator:
    def __init__(self, gt_dir=None):
        self.results = {}
        self.gt_dir = gt_dir
        self.metrics = {
            'PSNR': self.calculate_psnr,
            'SSIM': self.calculate_ssim,
            'MSE': self.calculate_mse,
            'VMAF': self.calculate_vmaf  # Requires FFmpeg
        }
    
    def load_image(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    def calculate_psnr(self, img1, img2):
        return psnr(img1, img2)
    
    def calculate_ssim(self, img1, img2):
        return ssim(img1, img2, multichannel=True, channel_axis=2)
    
    def calculate_mse(self, img1, img2):
        return np.mean((img1 - img2) ** 2)
    
    def calculate_vmaf(self, img1, img2, vmaf_path="vmaf", vmaf_model="vmaf_v0.6.1.json"):
        with tempfile.NamedTemporaryFile(suffix='.yuv') as ref_file, \
             tempfile.NamedTemporaryFile(suffix='.yuv') as dis_file:
            
            # Convert images to YUV420p format
            height, width = img1.shape[:2]
            ref_yuv = self._convert_to_yuv420(img1)
            dis_yuv = self._convert_to_yuv420(img2)
            
            # Write temporary files
            ref_file.write(ref_yuv.tobytes())
            dis_file.write(dis_yuv.tobytes())
            ref_file.flush()
            dis_file.flush()
            
            # Run VMAF calculation
            cmd = [
                vmaf_path,
                "-r", ref_file.name,
                "-d", dis_file.name,
                "-w", str(width),
                "-h", str(height),
                "-p", "420",
                "-b", "8",
                "--model", f"path={vmaf_model}",
                "--json"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Parse JSON output to get VMAF score
                import json
                vmaf_json = json.loads(result.stdout)
                return vmaf_json['frames'][0]['metrics']['vmaf']
            except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
                print(f"VMAF calculation failed: {str(e)}")
                return None
    
    def highlight_differences(self, img1, img2):
        # Generate heatmap of significant differences
        diff = cv2.absdiff(img1, img2)
        diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        return heatmap

    def _convert_to_yuv420(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)

    def model_consistency_analysis(self, df):
        # Analyze which models agree/disagree in their outputs
        # Get unique models and images
        models = df['model'].unique()
        images = df['image'].unique()
        
        # Initialize consistency metrics
        consistency_metrics = {
            'pairwise_psnr_corr': {},
            'pairwise_ssim_corr': {},
            'ranking_consistency': {}
        }
        
        # Calculate pairwise correlations between models
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                # Get common scores
                scores1 = df[df['model'] == model1].sort_values('image')
                scores2 = df[df['model'] == model2].sort_values('image')
                
                # Pearson correlation for PSNR and SSIM
                psnr_corr, _ = pearsonr(scores1['PSNR'], scores2['PSNR'])
                ssim_corr, _ = pearsonr(scores1['SSIM'], scores2['SSIM'])
                
                # Kendall's tau for ranking consistency
                kendall_tau, _ = kendalltau(scores1['PSNR'], scores2['PSNR'])
                
                key = f"{model1}_vs_{model2}"
                consistency_metrics['pairwise_psnr_corr'][key] = psnr_corr
                consistency_metrics['pairwise_ssim_corr'][key] = ssim_corr
                consistency_metrics['ranking_consistency'][key] = kendall_tau
        
        # Calculate overall consistency metrics
        all_psnr = [df[df['model'] == m]['PSNR'] for m in models]
        all_ssim = [df[df['model'] == m]['SSIM'] for m in models]
        
        consistency_metrics['mean_psnr_std'] = np.mean(np.std(np.array(all_psnr), axis=0))
        consistency_metrics['mean_ssim_std'] = np.mean(np.std(np.array(all_ssim), axis=0))
        
        return consistency_metrics

    def perceptual_analysis(self, img):
        # Use deep learning models to assess perceptual quality
        # Convert numpy array to torch tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        results = {}
        
        try:
            # Initialize perceptual metrics
            lpips = LPIPS().to(device)
            
            # For LPIPS and PieAPP we need a reference (use blurred version as pseudo-reference)
            with torch.no_grad():
                blurred = torch.nn.functional.avg_pool2d(img_tensor, kernel_size=3, stride=1, padding=1)
                
                # LPIPS (Learned Perceptual Image Patch Similarity)
                results['lpips'] = lpips(img_tensor, blurred).item()
                
        except Exception as e:
            print(f"Perceptual analysis failed: {str(e)}")
            results['lpips'] = None
        
        # Add NIQE (No-Reference quality metric)
        results['niqe'] = self.calculate_niqe(img)
        
        return results
    
    def calculate_niqe(self, img):
        # Calculate NIQE (Natural Image Quality Evaluator) score
        try:
            from sewar.full_ref import niqe
            return niqe(img)
        except ImportError:
            print("NIQE calculation requires sewar package")
            return None
        except Exception as e:
            print(f"NIQE calculation failed: {str(e)}")
            return None

    def evaluate_single(self, sr_path, model_name, image_name):
        sr_img = self.load_image(sr_path)
        
        if self.gt_dir:
            gt_path = os.path.join(self.gt_dir, f"{image_name}.png")  # Adjust extension as needed
            gt_img = self.load_image(gt_path)
        else:
            gt_img = None
        
        result = {'model': model_name, 'image': image_name}
        
        if gt_img is not None:
            for metric_name, metric_func in self.metrics.items():
                try:
                    result[metric_name] = metric_func(gt_img, sr_img)
                except Exception as e:
                    print(f"Error calculating {metric_name} for {image_name}: {str(e)}")
                    result[metric_name] = None
        
        # Store additional analysis
        result.update(self.visual_analysis(sr_img, gt_img))
        return result
    
    def visual_analysis(self, sr_img, gt_img=None):
        analysis = {}
        
        # Edge detection comparison
        sr_edges = cv2.Canny(cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY), 100, 200)
        analysis['edge_intensity'] = np.mean(sr_edges)
        
        if gt_img is not None:
            gt_edges = cv2.Canny(cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY), 100, 200)
            analysis['edge_similarity'] = ssim(sr_edges, gt_edges)
        
        # Color histogram analysis
        analysis['color_std'] = [np.std(sr_img[:,:,i]) for i in range(3)]
        
        # Texture analysis
        analysis['texture_score'] = self.calculate_texture(sr_img)
        
        return analysis
    
    def calculate_texture(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        return np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    def add_results(self, model_results):
        for result in model_results:
            key = (result['image'], result['model'])
            self.results[key] = result
    
    def generate_report(self, output_dir):
        # Create DataFrame from results
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Comparative analysis
        comparative = df.groupby('image').agg({
            'PSNR': ['mean', 'std', 'max'],
            'SSIM': ['mean', 'std', 'max'],
            'edge_intensity': ['mean', 'std'],
            'texture_score': ['mean', 'std']
        })
        
        # Save reports
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'detailed_results.csv'))
        comparative.to_csv(os.path.join(output_dir, 'comparative_analysis.csv'))
        
        # Generate visual comparison for each image
        self.generate_visual_comparisons(df, output_dir)
        
        return df, comparative
    
    def generate_visual_comparisons(self, df, output_dir):
        images = df['image'].unique()
        
        for img_name in images:
            img_results = df[df['image'] == img_name]
            best_model = img_results.loc[img_results['PSNR'].idxmax()]
            
            # Create comparison figure
            fig, axes = plt.subplots(1, len(img_results)+1, figsize=(20, 5))
            
            # Show original if available
            if self.gt_dir:
                gt_img = self.load_image(os.path.join(self.gt_dir, f"{img_name}.png"))
                axes[0].imshow(gt_img)
                axes[0].set_title("Ground Truth")
                axes[0].axis('off')
                start_idx = 1
            else:
                start_idx = 0
            
            # Show all SR results
            for idx, (_, row) in enumerate(img_results.iterrows(), start_idx):
                sr_img = self.load_image(f"output/{row['model']}/{img_name}_SwinIR.png")
                axes[idx].imshow(sr_img)
                title = f"{row['model']}\nPSNR: {row['PSNR']:.2f}, SSIM: {row['SSIM']:.4f}"
                if row['model'] == best_model['model']:
                    title = "★ " + title
                axes[idx].set_title(title)
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{img_name}_comparison.png"))
            plt.close()