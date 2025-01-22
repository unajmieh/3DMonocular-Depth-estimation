# 3DMonocular-Depth-estimation  

## Overview  

**3D Monocular Depth estimation** is a deep learning project designed to estimate depth maps from 2D images, transforming them into 3D representations. This project utilizes various advanced models, including DenseNet, Deep3DBox, EfficientNet, RegNet, and the MiDaS model, to achieve high accuracy in depth estimation.  

## Table of Contents  

- [Features](#features)  
- [Models Used](#models-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

## Features  

- Depth estimation from single 2D images.  
- Implementation of multiple state-of-the-art models for comparison.  
- High-quality depth maps that enhance 3D visualizations.  
- Support for various input image formats.  

## Models Used  

1. **DenseNet**: A convolutional neural network that connects each layer to every other layer in a feed-forward manner.  
2. **Deep3DBox**: A model tailored for 3D box localization and depth estimation.  
3. **EfficientNet**: A scalable convolutional neural network model that balances accuracy and efficiency.  
4. **RegNet**: A model focusing on regularized networks for optimal performance.  
5. **MiDaS**: An advanced model specifically designed for monocular depth estimation, known for its impressive results on various datasets.  

## Installation  

To get started with the project, clone the repository and install the required dependencies:  

bash  
git clone https://github.com/your_username/3DMonocular-Depth-estimation.git  
cd 3DMonocular-Depth-estimation  
pip install -r requirements.txt


## Usage  

To estimate depth from a 2D image, you can use the following command:  

```
bash  
python depth_estimation.py --input <path_to_image> --model <model_name>
```

Example
```
bash
python depth_estimation.py --input image.jpg --model MiDaS  
```


## Results
The estimated depth maps will be output in the specified directory. You can visualize the results using standard image viewers or additional scripts provided in this repository for enhanced visualizations.

## Contributing
Contributions are welcome! If you want to contribute, please fork the repository and submit a pull request. For significant changes, please open an issue first to discuss what you want to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
MiDaS for their groundbreaking work on depth estimation.
The authors of DenseNet, Deep3DBox, EfficientNet, and RegNet for their contributions to neural network architecture development.
