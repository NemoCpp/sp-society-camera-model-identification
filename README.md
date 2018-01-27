
# Kaggle Competition: IEEE's Signal Processing Society - Camera Model Identification

## Introduction

This project is used for the Kaggle Competition: IEEE's Signal Processing Society - Camera Model Identification

The original work is from Andres Torrubia. He has made significant changes from the code base I forked to create this project. You can get his work at:  https://github.com/antorsae/sp-society-camera-model-identification

My specific hardware consists of a desktop withan i7 for the CPU and a single GTX1080Ti for the GPU. I believe the reliability problems I was having were caused by the parallel processing code. So for now the changes in this project will make it run slower.

The goals of this fork are:

* Stability and robustness
* Break into smaller files separated by function
* Project structure for a more generic framework
* Fully documented via inline comments within the source code

## Input images

The dataset images is shared across multiple projects. As a result there is a central dataset directories and the projects simply contain symbolic links.

The test and train directories are the standard ones from the data provided by Kaggle.

The flickr_images and val_images directories are based on the extra data as proposed by Gleb Posobin. For more information visit: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235

Note that downloading the flickr images can take several attempts. Also, any file ending in upper case (ie. JPG) needs to be renamed to lower case (ie. jpg)

## Running

Running for a DenseNet201 network:

python train.py -g 1 -b 4 -cs 512 -cm ResNet50 -x -l 1e-4

Running for a DenseNet201 network:

python train.py -g 1 -b 4 -cs 512 -cm DenseNet201 -x -l 1e-4


## License

I started to work on this when the original code had no license attached. Since then Andres has released his code with a GPLv3 license.

I am keeping my code and contributions under my own copyright and releasing them under the more permissive MIT license.

A different project will be released later on with a more generic code base fully under an MIT license.

## Future

Work on this code tree will continue to the extent that I need to make submissions to the competition.
