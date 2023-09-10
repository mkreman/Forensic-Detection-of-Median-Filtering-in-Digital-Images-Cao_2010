# Forensic Detection of Median Filtering in Digital Images
This GitHub project presents the Python code for detecting median filtering in images based on the [study](https://ieeexplore.ieee.org/abstract/document/5583869) done by Gang Cao, Yao Zhao, Rongrong Ni, Lifang Yu, Huawei Tian. I used the [UCID Dataset](https://www.researchgate.net/publication/220979862_UCID_An_uncompressed_color_image_database) for training the model.
The study aims to detect the tempering process on image documents done by forgery makers.

# Overview
This project follows the following steps
- Grayscale images are being used by converting the images into grayscale images in the standard manner.
- The UCID dataset is considered as negative images (original uncompressed images) for the model.
- Positive images are created by applying median filtering on the same dataset images.
- Hyperparameters are tunned using cross-validation.
- ROC curves are created and the results given in the paper are verified.

# Data Preparing
The model takes original uncompressed images and various filtered (Gaussian, average, jpeg55,..., etc.) images of the UCID dataset as negative values. And their median filtered images are taken as positive values.

# Model Architecture
Since the median filter makes the neighbouring pixel same. Model counts the zeros of row and column difference of images. Look for the zeros on the textured region and compare them with a threshold to identify the image as a median-filtered or non-median-filtered one.

# Results
For classification between original and median filtered images, the model performed well using the parameters d (the width of the square statistical region) = 7, tau (a threshold to determine whether a region is textured or not) = 100, and threshold (to determine whether the image is median filtered or not) = 0.22 with 98.5% recall for class 1 (median filtered images) and 99% recall for class 0 (original images).

If the images are post processed (jpeg compression) after applying median filter then the model yeild a lower recall of $72.24\%$.

# Limitations and Future Work
The model works well for uncompressed images but does not work well for other filtered images like (Gaussian, average, and JPEG compressed images).

Since computing a feature to determine median filtering is a very difficult task and is affected by various manipulations. And affacted by the post processing of positive images. Therefore there is a need for a technique that automatically computes the feature and classifies the median filtering. ML algorithm is the way to go. I explored a such technique in [this GitHub repository](https://github.com/nagar-mayank/Median-Filtering-Forensics-Based-on-Convlutional-Neural-Network.git).