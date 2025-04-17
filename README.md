# 3D Monocular Depth Estimation

**3D Monocular Depth Estimation** is a deep learning project designed to predict depth maps from 2D images. By leveraging advanced models, the project achieves high accuracy in depth prediction, enabling better understanding of 3D structures in various applications.

## Overview

This project integrates cutting-edge machine learning models, including:
- **DenseNet**
- **Deep3DBox**
- **EfficientNet**
- **RegNet**
- **MiDaS**

These models contribute to robust depth estimation and ensure precise generation of depth maps from single RGB images.

## Features

- **State-of-the-art Models**: Utilizes advanced deep learning architectures for superior performance.
- **High Accuracy**: Optimized for precise depth prediction across diverse scenarios.
- **Scalable and Flexible**: Easily customizable for different datasets and applications.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/unajmieh/3DMonocular-Depth-estimation
    cd 3d-monocular-depth-estimation
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the necessary dataset and pre-trained weights. Refer to the [Dataset](#dataset) section for details.

## Usage

To run the depth estimation:
```bash
python Models.py --input path_to_input_image --output path_to_output

