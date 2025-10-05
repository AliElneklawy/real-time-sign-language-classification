# Real Time Sign Language Classification

This project aims to classify sign language letters using machine learning. This README provides an overview of the project and instructions for usage.

## Table of Contents
- [Demo](#Demo)
- [Project Structure](#ProjectStructure)
- [Usage](#Usage)

## Demo

<p align="center">
  <img src="https://github.com/AliElneklawy/real-time-sign-language-classification/blob/main/assets/demo.gif" alt="demo">
</p>

## Project Structure

The project consists of five main files:

  `augmentation.py`: This file contains code for data augmentation, which is a technique used to increase the size and diversity of a dataset.
  
  `collect_images.py`: This file uses OpenCV to collect images of sign language letters from the user's camera.
  
  `create_data.ipynb`: This Jupyter Notebook uses MediaPipe to track the hand in the collected images and extract the keypoint features. The keypoint features are then used to create a dataset for training the random forest classifier.
  
  `train_model.py`: This file trains the random forest classifier on the created dataset.

  `inference.py`: This file is used to classify sign language letters in real-time.

## Usage

You only need two files to run the project; `utils/model.pkl` and `inference.py`. Here are the main usage instructions:

  - Run the real-time sign langugae detection script: `python inference.py`

  - A live video stream will open, and the application will start detecting hands and their status.

  - To quit the application, press the 'q' key.

