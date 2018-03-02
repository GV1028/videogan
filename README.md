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
<table><tr><td>
<strong>Real videos</strong><br>
<img src='images/true1.gif'>
<img src='images/true2.gif'>
<img src='images/true3.gif'><br>
<img src='images/true4.gif'>
<img src='images/true5.gif'>
<img src='images/true6.gif'><br>
<img src='images/true7.gif'>
<img src='images/true8.gif'>
<img src='images/true9.gif'>
</td><td>
<strong>Generated videos</strong><br>
<img src='images/gen1.gif'>
<img src='images/gen2.gif'>
<img src='images/gen3.gif'><br>
<img src='images/gen4.gif'>
<img src='images/gen5.gif'>
<img src='images/gen6.gif'><br>
<img src='images/gen7.gif'>
<img src='images/gen8.gif'>
<img src='images/gen9.gif'>
</td></tr></table>

## Acknowledgements
* [Generating Videos With Scene Dynamics](http://carlvondrick.com/tinyvideo/paper.pdf) - Carl Vondrick et al.

