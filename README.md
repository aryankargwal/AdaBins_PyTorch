# AdaBins: A PyTorch Implementation

## Introduction
An easy to use implementation of the [Adabins](https://arxiv.org/abs/2011.14141) by Bhat et al.
This effort was undertaken under [SRM-MIC](https://github.com/srm-mic)'s 'ResCon' event.

## The prompt
The problem addressed over here is the estimation of the Depth map of an environment from a single RGB image so as to aid automative vehicles/robots and hopefully replace stereo cameras and LIDAR which are being used right now.<br>

<img src="https://www.controleng.com/wp-content/uploads/sites/2/2019/04/CTL1904_WEB_IMG_Cornell_LiDAR_Stereo.jpg"><br>

## The Approach
This prompt has been one of the classic Computer Vision task with a vast number of architectures trying to tackle this. The architectures however have a drawback which being that the global analysis of the output values takes place when the tensors reach a very small spatial resolution or are at the bottleneck layer. To deal with this very problem the Authors propose a new architecture building block known as <b><i>AdaBins</i></b>.<br>
<img src="images/1.png">
The <b><i>AdaBins</i></b> module performs a global statistical analysis of the output from a tradi-tional encoder-decoder architecture and tries to refine the output Depth map. 

## The Dataset
For our implementation we decided to go for the [NYU Depth Dataset v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). The dataset consists of over 1400 Densely Labeled indoor images with RGB and depth images pairs.<br>
<img src="https://cs.nyu.edu/~silberman/images/nyu_depth_v2_labeled.jpg"> 

## The Results
Paper Results<br>
<img src="examples/paper_image.png" style="height:400px"><br>
Our Results<br>
<img src="examples/detected_image.png" style="height:400px"><br>

You can access the demo notebook [here](https://nbviewer.jupyter.org/github/aryankargwal/AdaBins_PyTorch/blob/main/Adabins_Inference_example.ipynb) if the [GitHub](https://github.com/aryankargwal/AdaBins_PyTorch/blob/main/Adabins_Inference_example.ipynb) does not open.
