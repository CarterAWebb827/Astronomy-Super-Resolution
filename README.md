# Super Resolution Comparison Tool

A comprehensive tool for comparing different super resolution (SR) approaches including SRGAN, Real-ESRGAN, SwinIR, and SinSR. This tool provides training, inference, and evaluation capabilities with extensive metrics reporting. This tool also aims to focus on astronomy-based images, where accuracy and fidelity is key when using super resolution.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Interpretation](#results-interpretation)

## Features

- **Multiple SR Approaches**: Compare SRGAN, Real-ESRGAN, SwinIR, and SinSR
- **Training & Inference**: Supports both training and inference modes for SRGAN-based models. Supports inference mode for SwinIR and SinSR.
- **Comprehensive Evaluation**:
  - Traditional metrics (PSNR, SSIM, MSE)
  - Perceptual metrics (LPIPS, NIQE)
  - Visual analysis (edge intensity, texture score)
  - Artifact detection (blocking, ringing artifacts)
- **Batch Processing**: Evaluate multiple images at once
- **Model Comparison**: Compare performance across different models
- **Detailed Reporting**: CSV exports and visualizations

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (recommended) with CUDA 11.3+
- Linux (tested on Ubuntu 22.04)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/CarterAWebb827/Astronomy-Super-Resolution.git
cd Astronomy-Super-Resolution
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# or 
venv\Scripts\activate     # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install additional dependencies:
```bash
# For LPIPS metric
pip install lpips

# For NIQE metric (if available)
pip install sewar

# For Real-ESRGAN
pip install basicsr

# For SwinIR
pip install timm
```

## Usage

### Running the Tool

```bash
python main.py
```

You'll see the main menu:
```
==================================================
       Super Resolution Comparison Tool        
==================================================

1. SRGAN
2. Real-ESRGAN
3. SwinIR
4. SinSR
0. Evaluate SR Results
```

### Basic Workflows

1. **Run Inference with a Model**:
   - Select a model (1-4)
   - Choose inference mode
   - Provide input path (image, directory, or video)
   - Configure model parameters during selection

2. **Evaluate Results**:
   - Select option 0 (Evaluate SR Results)
   - Provide paths to SR results and ground truth (if available)

### Example Commands

**Single Image Inference with SRGAN**:
```bash
python main.py
# Select 1 (SRGAN)
# Select inference
# Enter image path: images/input/test.jpg
# Use default parameters
```

**Evaluation**:
```bash
python main.py
# Select 0 (Evaluate SR Results)
# Enter SR results directory: output/srgan_results/
# Enter ground truth directory (if available): images/ground_truth/
```

## Evaluation Metrics

The tool calculates several metrics:

### Traditional Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **MSE**: Mean Squared Error (lower is better)
- **VMAF**: Video Multi-Method Assessment Fusion (0-100, higher is better)

### Perceptual Metrics
- **LPIPS**: Learned Perceptual Image Patch Similarity (0-1, lower is better)
- **NIQE**: Natural Image Quality Evaluator (lower is better)

### Visual Analysis
- **Edge Intensity**: Measures sharpness
- **Texture Score**: Evaluates texture preservation
- **Sharpness (Laplacian)**: Quantifies image sharpness
- **Artifact Detection**: Identifies blocking and ringing artifacts

## Results Interpretation

After evaluation, you'll find these files in your output or batch_evaluation directory:

### CSV Reports
- `combined_model_summary.csv`: Contains all summary metrics for each model
- `model_summary.csv`: Summary statistics for each model
- `detailed_results.csv`: Full statistics for each model

### Visualization
- `performance_plots_XX.png`: Performance image plots

### Compare models - using the summary statistics
|Model Name          |edge_intensity|edge_intensity|edge_intensity|edge_intensity|texture_score|texture_score|texture_score|texture_score|lpips |lpips |lpips |lpips |mean_brightness|mean_brightness|mean_brightness|mean_brightness|std_brightness|std_brightness|std_brightness|std_brightness|contrast|contrast|contrast|contrast|sharpness_laplacian|sharpness_laplacian|sharpness_laplacian|sharpness_laplacian|blocking_artifacts|blocking_artifacts|blocking_artifacts|blocking_artifacts|ringing_artifacts|ringing_artifacts|ringing_artifacts|ringing_artifacts|
|--------------------|--------------|--------------|--------------|--------------|-------------|-------------|-------------|-------------|------|------|------|------|---------------|---------------|---------------|---------------|--------------|--------------|--------------|--------------|--------|--------|--------|--------|-------------------|-------------------|-------------------|-------------------|------------------|------------------|------------------|------------------|-----------------|-----------------|-----------------|-----------------|
|                    |mean          |std           |min           |max           |mean         |std          |min          |max          |mean  |std   |min   |max   |mean           |std            |min            |max            |mean          |std           |min           |max           |mean    |std     |min     |max     |mean               |std                |min                |max                |mean              |std               |min               |max               |mean             |std              |min              |max              |
|netG_epoch_4_250.pth|6.8965        |7.7752        |0.652         |31.4872       |421.6014     |283.3227     |107.5805     |1095.4412    |0.0383|0.0122|0.0126|0.0544|52.3385        |24.78          |9.5793         |90.8944        |51.4446       |17.3354       |21.3747       |81.5475       |50.4673 |17.975  |16.5037 |79.4431 |65.4917            |67.8756            |11.1501            |303.1057           |71.7618           |26.3059           |24.8816           |118.5443          |13.7912          |6.2117           |6.7367           |31.4832          |
|generator_28050.pth |9.2273        |14.7951       |0.6951        |68.9174       |406.9509     |290.9884     |105.5305     |1238.381     |0.0784|0.0529|0.0185|0.2476|52.2575        |25.1395        |9.7353         |91.8816        |51.805        |17.6839       |20.4083       |82.5822       |50.8816 |18.0154 |17.3048 |80.5971 |185.5896           |332.4347           |18.9172            |1585.0735          |76.2028           |28.4628           |24.9381           |125.15            |18.6774          |11.0836          |7.6097           |57.8113          |
|DFO_PSNR            |5.786         |5.9899        |0.6173        |24.0119       |350.1271     |312.9352     |72.0221      |1258.6761    |0.0186|0.0113|0.0067|0.051 |52.3721        |25.6225        |9.2702         |92.5821        |53.6681       |18.3488       |21.3203       |86.0309       |52.5804 |18.6364 |17.9118 |83.7501 |77.9874            |105.4748           |11.2227            |377.0494           |49.0225           |22.5055           |15.2553           |98.0906           |14.1065          |8.2762           |6.5099           |38.4682          |
|DFO_GAN             |7.4267        |6.7269        |0.6898        |24.0644       |347.7819     |298.9178     |77.2         |1160.871     |0.055 |0.0261|0.0151|0.0978|51.4152        |25.9304        |6.6241         |92.5321        |53.1506       |19.0838       |18.9569       |86.1885       |52.1421 |19.3716 |17.6242 |84.1373 |206.4889           |227.7963           |30.9914            |836.1832           |64.3967           |23.195            |19.8249           |102.2201          |20.7615          |9.2082           |9.3458           |41.293           |
|DFOWMFC_GAN         |7.5952        |6.3599        |0.8369        |23.4174       |353.7162     |280.3943     |81.3883      |1099.1411    |0.0526|0.0244|0.011 |0.0981|51.9062        |25.6985        |7.2024         |92.3086        |53.2281       |18.8798       |19.8526       |86.3552       |52.1703 |19.1354 |17.7633 |84.1148 |169.1567           |144.929            |28.3283            |665.3937           |63.6927           |24.0372           |22.2297           |106.6596          |20.088           |7.7577           |9.5181           |39.3012          |
|DFOWMFC_PSNR        |5.6376        |5.875         |0.5553        |23.5927       |346.3694     |298.1925     |72.0816      |1184.5099    |0.0171|0.0091|0.0065|0.0376|52.4901        |25.5346        |9.2872         |92.5647        |53.412        |18.2587       |21.3099       |85.715        |52.3405 |18.5669 |17.9293 |83.4866 |59.2548            |64.8854            |7.8481             |232.5098           |50.2212           |23.7423           |14.9345           |102.5577          |12.9752          |6.497            |5.5882           |27.817           |
|single-step         |9.7289        |8.042         |1.8706        |23.2107       |304.4038     |129.0354     |158.6052     |494.2762     |0.1248|0.0611|0.0441|0.2286|51.3118        |29.3716        |13.7414        |93.568         |56.7515       |18.6035       |33.113        |76.9046       |55.8716 |19.1877 |33.0088 |76.7382 |653.1445           |642.1976           |135.2595           |1854.1793          |86.4631           |12.8222           |62.9734           |98.8165           |30.4095          |13.1351          |17.3289          |51.8098          |
|non-single-step     |9.947         |8.4955        |1.839         |24.5516       |305.4771     |131.1986     |160.9348     |502.7327     |0.1248|0.0616|0.046 |0.2332|51.3347        |29.3363        |13.8092        |93.5531        |56.7424       |18.6155       |33.1411       |76.8955       |55.869  |19.1993 |33.0351 |76.7322 |658.9804           |686.4103           |145.6755           |1968.55            |86.5076           |12.5518           |63.7628           |99.4722           |30.5018          |13.5006          |17.9024          |53.1133          |

- **Edge Intensity**: For edge intensity, the Real-ESRGAN and SinSR approaches seem to have the best values, while not being excesively high. This indicates sharper, more defined edges.
- **Texture Score**: For texture score, the SRGAN and Real-ESRGAN models have the best values, with the SRGAN model performing the best overall. This measures richness of texture details, where higher values indicate better preservation of natural textures.
- **LPIPS**: For LPIPS, it seems the SwinIR small model of DFO with PSNR and DFO with MFC and PSNR perform quite well. LPIPS helps to measures perceptual similarity to a reference. Lower values display more natural results.
- **Mean Brightness**: For mean brightness, it seems no models deviate excessively, meaning they are all similar to the original image. 
- **Std Brightness**: Similarly for standard deviation of the brightness, it seems no models deviate excessively, meaning they are all similar to the original image.
- **Contrast**: Again, with contrast, each image preserves the contrast, which helps to reserves an images "pop".
- **Sharpness (Laplacian)**: Laplacian sharpness shows how sharp the image is that a given model produces. Values that are too high can show over-sharpening. This can be seen with the SinSR models. A more appropriate value seems to be with the Real-ESRGAN model or the SwinSR model that utilizes DFO and GAN.
- **Blocking Artifacts**: Block artifacts are grid-like compression artifacts. We prefer a lower value in this category. Some of the lowest values come from DFO with PSNR and DFO with MFC and PSNR from the SwinIR approach.
- **Ringing Artifacts**: Ringing artifacts are halos and/or oscillations near the edges of an image. We also prefer a lower value, which we get from the DFO with MFC and PSNR, SRGAN, and DFO with PSNR approaches.

### Check consistency - good models should perform well across all metrics
Looking through the models, it seems the SRGAN is the most average model. It doesn't perform incredibly well, but it doesn't perform poorly either. Similarly, the Real-ESRGAN approach is just an overall well-performing model. It has good performance in most of the categories and even performs the best in a handful of them. In my opinion, the best overall approach comes from the SwinIR transformer, specifically the DFO with MFC and PSNR. It performs exceedingly well in many categories and when it isnt performing the best, it usually performs well. Also, the SinSR diffusion approach seemed to do well, but it hallucinates too many objects and doesn't do well when considering the halos generated from it.