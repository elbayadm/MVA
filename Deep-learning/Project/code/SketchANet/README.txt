Author: Yongxin Yang
Last Revised: 23/07/2015

Here is an instruction file for the code, datasets, and pre-trained models that are used in the paper titled,

Sketch-a-Net that Beats Humans (BMVC'15) http://arxiv.org/pdf/1501.07873v3.pdf

1. Dataset files

In total, 10 dataset files are given. They are basically a re-wrap of the PNG files in TU-Berlin Sketch Dataset.

http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/

The sketch dataset is licensed under a Creative Commons Attribution 4.0 International License. (http://creativecommons.org/licenses/by/4.0/)

Nothing is changed except they are saved in MATLAB data format, besides the naming variables follows the needs of Matconvnet.

There are two factors that make the difference: (i) with/without order information (ii) resolution 

Note, the size of each image is always 256x256, but some of those are oversampled from a lower resolution (e.g., 192x192 -> 256x256). The resolution here refers to the original value (e.g., 192x192). 

You can tell the difference from the names of files.

dataset_with_order_info_64.mat

dataset_with_order_info_128.mat

dataset_with_order_info_192.mat

dataset_with_order_info_224.mat

dataset_with_order_info_256.mat

dataset_without_order_info_64.mat

dataset_without_order_info_128.mat

dataset_without_order_info_192.mat

dataset_without_order_info_224.mat

dataset_without_order_info_256.mat

2. Pre-trained model files

Corresponding to the ten dataset files above, we also provide the pre-trained model files. You can tell which model is associated with which dataset file by their naming format.

model_with_order_info_64.mat

model_with_order_info_128.mat

model_with_order_info_192.mat

model_with_order_info_224.mat

model_with_order_info_256.mat

model_without_order_info_64.mat

model_without_order_info_128.mat

model_without_order_info_192.mat

model_without_order_info_224.mat

model_without_order_info_256.mat

3. Code

There is an AIO file named "cnn_sketch.m" by which you can (re-)train the CNN model.

To run the code, you need to change three lines as follow,

Line 3

--> run('../matlab/vl_setupnn.m') ;

Change this line to the path of your installed Matconvnet.

Line 5

--> opts.expDir = fullfile('sketch_model') ;

This line assigns a folder that stores all the intermediate files generated during training.

Line 6

--> opts.imdbPath = fullfile('dataset_with_order_info_256.mat');

You can find the available dataset files in the first section. 

A small piece of code to illustrate testing can be found in "cnn_test.m".

It crops each testing image into 10 sub-samples, and predicts the label by average voting.
