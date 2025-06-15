# PET-Attenuation-Correction-with-GANs

This repository contains code and data for PET image attenuation correction using Generative Adversarial Networks (GANs). We compare a vanilla Pix2Pix model against a WGAN-GP variant to generate attenuation-corrected (AC) images from non-corrected (NAC) PET scans without the need for CT. Source detection on NAC and GEN (WGAN-GP) images with YOLOv8n is examined.

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Model Training and Testing](#model-training-and-testing)
* [Results](#results)
* [License](#license)
* [Contact](#contact)

## Overview

**Attenuation correction** is essential in PET imaging because photon loss inside the body can lead to inaccurate intensity values. Traditional correction uses CT images, increasing radiation dose and cost. In this work, we:

1. **Simulate** NAC/AC image pairs using GATE (Monte Carlo toolkit).
2. **Train** two GANs:

   * **Pix2Pix** (U-Net + PatchGAN) with BCE + L1 loss.
   * **WGAN-GP** (U-Net + Wasserstein loss + gradient penalty).
3. **Evaluate** image quality (PSNR, SSIM) and downstream object detection (YOLOv8).

## Repository Structure

```
PET-Attenuation-Correction-with-GANs/
├── vanilla_pix2pix/       # Pix2Pix model code
│   ├── generator.py
│   ├── discriminator.py
│   ├── training.py
│   └── utils.py
│   └── config.py
│   └── dataset.py
├── wgan_gp/               # WGAN-GP model code
│   ├── generator.py
│   ├── critic.py
│   ├── training.py
│   └── utils.py
│   └── config.py
│   └── dataset.py
├── YOLOv8n/             
│   └── YOLOv8_detection.ipynb   # Fine-tune and results of soyrce detection with YOLOv8n
│   └── find_parameters.py   #Extract YOLO parameters for fine tuning
├── GATE_sims/              #Contains macro files for GATE simulations and .shell files for automation
│   ├── PET_AC_single.mac
│   ├── PET_AC_double.mac
│   └── PET_NAC_single.mac
│   └── PET_NAC_double.mac
│   └── run_all_single.sh
│   └── run_all_double.sh
├── recon/             #Contains reconstruction code files for single and double sources
│   └── iter_recon_single.py
│   └── iter_recon_double.py
└── split_data.py              # Code for splitting data to train/val/test 
└── Report.pdf              # Report of the project
└── Presentation.ppt              # Presentation of the project
└── README.md              # This file
```

### Requirements

* Python 3.8+
* PyTorch 1.12+
* torchvision
* GATE (for simulation)
* Uproot, numpy, scikit-image, matplotlib, albumentations
* Ultralytics YOLOv8 (for object detection)


## Installation

1. Clone this repository:
```
git clone [https://github.com/iraklisspyrou/PET-Attenuation-Correction-with-GANs.git](https://github.com/iraklisspyrou/PET-Attenuation-Correction-with-GANs.git)
cd PET-Attenuation-Correction-with-GANs
```


2. Install Python packages from the above mentioned requirements.


3. Install vGATE (Virtual Box version of GATE) for data simulations.

## Data Preparation

1. Use GATE to simulate PET phantoms and generate paired NAC/AC images:
        Run the two .sh files in the GATE_sims folder
2. Reconstruct images using reconstruction python files in the recon folder

## Model Training and testing

### Pix2Pix

```
python vanilla_pix2pix/training.py \
```

### WGAN-GP

```
python wgan_gp/training.py \

```


### Downstream detection with YOLOv8:

   ```
  find_parameters.py, \
  YOLOv8_detection.ipynb \
  ```

## Results
- **Quantitative Metrics**: WGAN-GP outperforms Pix2Pix by +1.6 dB PSNR and has smoother training with much less spikes in the loss curves
- **Detection**: YOLOv8 mAP@50 improved from 0.932 (NAC) to 0.991 (generated).

See the `Report.pdf` for images, metrics, and plots.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or contributions, please open an issue or contact:
- **Iraklis Spyrou** (iraklis.spyrou@gmail.com)
-**Alexandros Ntagiantas** (alexisnt13@gmail.com)

