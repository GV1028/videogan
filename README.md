# Generating Videos with Scene Dynamics

## Introduction
This repository contains an implementation of "Generating Videos with Scene Dynamics" in Tensorflow. The paper can be found here (http://carlvondrick.com/tinyvideo/paper.pdf). The model learns to generate a video by upsampling from some latent space, using adversarial training.

## Requirements
For running this code and reproducing the results, you need the following packages. Python 2.7 has been used.

Packages:
* TensorFlow
* NumPy
* cv2
* scikit-video
* scikit-image


## VideoGAN - Architecture and Working
Attached below is the architecture used in the paper [paper](http://carlvondrick.com/tinyvideo/paper.pdf).<br />
![Video_GAN](images/videogan.png)

## Usage  
Place the videos inside a folder called "trainvideos".<br />
Run main.py with the required values for each flag variable.

## Results
Below are some of the results on the model trained on MPII Cooking Activities dataset.<br />
![Sample Train Video](images/true_video.gif)<br/>
![Sample Generated Video](images/generated_video.gif)
