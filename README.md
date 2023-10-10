# SSY340 Deep Machine Learning Project

SSY340 Deep Machine Learning project, a part of the Chalmers University of Technology course.

## Project Overview
Our project focuses on leveraging semantic segmentation techniques to detect tumors in brain MRI images. Specifically, we will train a U-Net model using a curated dataset and explore the impact of incorporating attention gates into the architecture. This comparative analysis will allow us to assess the performance of the original U-Net model against an enhanced version with attention mechanisms. U-Net is a convolutional neural network (CNN) architecture uniquely designed for biomedical image segmentation tasks. It features an encoder-decoder structure with skip connections, making it particularly effective for tasks such as tumor detection.

## Repository Structure
- **Input**: This folder contains the dataset sourced from Kaggle. You can access the dataset here: [Kaggle Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

- **brain_dataset**: This section of the repository houses various functions related to data preprocessing, augmentation, and manipulation.

- **unet_architecture**: Here, you'll find the U-Net architecture implementation, which serves as the foundation of our project.

- **attention_gate**: This directory contains the implementation of the attention gate architecture. We will use this to compare the performance of the original U-Net model.

- **project.ipynb**: The main project file, located in this notebook, is where we execute and document the code.



