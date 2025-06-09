# Super Resolution Comparison Tool

A comprehensive tool for comparing different super resolution (SR) approaches including SRGAN, Real-ESRGAN, SwinIR, and SinSR. This tool provides training, inference, and evaluation capabilities with extensive metrics reporting. This tool also aims to focus on astronomy-based images, where accuracy and fidelity is key when using super resolution.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Interpretation](#results-interpretation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

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

### Traditional Metrics (require ground truth)
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

After evaluation, you'll find these files in your output directory:

### CSV Reports
- `detailed_results.csv`: Contains all metrics for each image
- `model_summary.csv`: Summary statistics for each model

Example CSV output:
```csv
image,model,PSNR,SSIM,LPIPS,sharpness_laplacian
test1.png,SRGAN,28.45,0.892,0.12,45.23
test1.png,ESRGAN,29.12,0.901,0.09,48.76
```

### Visualizations
- `psnr_vs_ssim.png`: Scatter plot comparing PSNR and SSIM
- `performance_plots_0.png`: Box plots of metric distributions
- Visual comparison images showing SR results vs ground truth

### Interpreting Results
1. **Compare models** using the summary statistics
2. **Check consistency** - good models should perform well across all metrics
3. **Examine tradeoffs** - some models may excel in PSNR but have higher artifacts
4. **Review visual comparisons** - metrics don't always capture perceptual quality

## Troubleshooting

### Common Issues

**1. Missing Dependencies**
```
ModuleNotFoundError: No module named 'lpips'
```
Solution: Install missing packages with `pip install lpips`

**2. CUDA Errors**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce batch size or use smaller images

**3. Missing Ground Truth**
```
"Column(s) ['PSNR', 'SSIM'] do not exist"
```
Solution: Provide ground truth images or use perceptual metrics only

**4. File Not Found**
```
FileNotFoundError: [Errno 2] No such file or directory
```
Solution: Verify all paths are correct and relative to the project root

### Getting Help

For additional support, please [open an issue](https://github.com/yourusername/super-resolution-comparison/issues) on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.