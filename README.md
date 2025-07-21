# Shola
 
![text-to-image](https://img.shields.io/badge/text--to--image-blue)
![deep-learning](https://img.shields.io/badge/deep--learning-green)
![image-generation](https://img.shields.io/badge/image--generation-purple)
![character-consistency](https://img.shields.io/badge/character--consistency-orange)
![unet](https://img.shields.io/badge/unet-red)
![pytorch](https://img.shields.io/badge/pytorch-lightgrey)
![replication-guide](https://img.shields.io/badge/replication--guide-brightgreen)

Shola is a text-to-image deep learning pipeline designed to generate images with consistent character appearances across different prompts and views. The project leverages PyTorch and UNet-based architectures, with a focus on robust preprocessing, hyperparameter tuning, fine-tuning, and inference.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Hyperparameter Tuning](#2-hyperparameter-tuning)
  - [3. Fine-tuning](#3-fine-tuning)
  - [4. Continue Fine-tuning](#4-continue-fine-tuning)
  - [5. Inference](#5-inference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Text-to-Image Generation:** Generate images from text prompts with character appearance consistency.
- **Custom Dataset Handling:** Automated caption generation and preprocessing for structured image datasets.
- **Augmentation & Normalization:** Built-in image augmentation and normalization for robust training.
- **Hyperparameter Tuning:** Scripted support for optimizing model parameters.
- **Fine-tuning & Inference:** Easy-to-use scripts for model training and image generation.

---

## Project Structure

```
Shola/
  ├── print_unet_submodules.py
  ├── README.md
  └── src/
      ├── 01_preprocessing.py
      ├── 02_hyperparameter_tuning.py
      ├── 03_finetuning.py
      ├── 04_continue_finetuning.py
      └── 05_inference.py
```

---

## Dependencies

Install the following Python packages (preferably in a virtual environment):

```bash
pip install torch torchvision pandas pillow
```

**Full list:**
- torch
- torchvision
- pandas
- pillow

If you are running on Kaggle or Colab, these may already be installed.

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Shola
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision pandas pillow
   ```

---

## Data Preparation

Organize your dataset as follows:

```
image_input/
  ├── happy/
  │   ├── front_view/
  │   │   ├── img1.png
  │   │   └── ...
  │   └── side_view/
  │       ├── img2.png
  │       └── ...
  └── sad/
      ├── front_view/
      └── side_view/
```

Each emotion (e.g., `happy`, `sad`) is a folder containing `front_view` and `side_view` subfolders with images.

---

## Step-by-Step Usage Guide

### 1. Preprocessing

**Script:** `src/01_preprocessing.py`

- **Purpose:** Preprocess images, apply augmentations, generate captions, and save processed images and a CSV file with captions.
- **How to Run:**

  ```bash
  python src/01_preprocessing.py
  ```

- **Inputs:** Expects images in `/kaggle/input/image-input-main/image_input` (modify the path in the script if needed).
- **Outputs:** Processed images in `/kaggle/working/processed_images` and captions in `/kaggle/working/captions.csv`.

**To use your own data path:**  
Edit the `input_dir` and `output_dir` variables in `01_preprocessing.py`.

---

### 2. Hyperparameter Tuning

**Script:** `src/02_hyperparameter_tuning.py`

- **Purpose:** Tune model hyperparameters for optimal performance.
- **How to Run:**

  ```bash
  python src/02_hyperparameter_tuning.py
  ```

- **Inputs:** Processed images and captions from the previous step.
- **Outputs:** Best hyperparameters (check script output or logs).

---

### 3. Fine-tuning

**Script:** `src/03_finetuning.py`

- **Purpose:** Fine-tune the model using the best hyperparameters.
- **How to Run:**

  ```bash
  python src/03_finetuning.py
  ```

- **Inputs:** Processed data and hyperparameters.
- **Outputs:** Fine-tuned model weights (check script for output path).

---

### 4. Continue Fine-tuning

**Script:** `src/04_continue_finetuning.py`

- **Purpose:** Resume fine-tuning from a checkpoint.
- **How to Run:**

  ```bash
  python src/04_continue_finetuning.py
  ```

- **Inputs:** Checkpoint from previous fine-tuning.
- **Outputs:** Updated model weights.

---

### 5. Inference

**Script:** `src/05_inference.py`

- **Purpose:** Generate images from text prompts using the trained model.
- **How to Run:**

  ```bash
  python src/05_inference.py
  ```

- **Inputs:** Trained model weights, text prompts.
- **Outputs:** Generated images (check script for output directory).

---

## Troubleshooting

- **Module Not Found:** Ensure all dependencies are installed and you are using the correct Python environment.
- **File Paths:** Double-check input/output paths in each script.
- **CUDA Errors:** If using GPU, ensure CUDA is properly installed and available.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License.

---

**Happy experimenting! If you have questions or need help, feel free to open an issue.**