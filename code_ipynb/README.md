# What is this repo ?

This repository is for the final project of APMA4990

Using GAN for image Deblurring. 

Original paper [DeBlur GANs](https://arxiv.org/pdf/1711.07064.pdf). 


By:
Yilin Lyu	 *yl3832*

Yiyi Zhang *yz3280*

Liangliang Huang *lh2863*

Shan Guan  *sg3506*

## What is inside this folder?

This file contains the **dataset organization**，**model building**, **model training**, **comparations and evaluations**, and **model testing** and all baisc components of the project. 

You could find step-by-step instructions and results in `main.ipynb` 

Detailed about implementation of another model could be found in `keras_basic_GAN`

## Dataset:

Get the [GOPRO dataset](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing), and extract it in the working directory. The directory name of dataset should be `GOPRO_Large`.

## Use the dataset:
```
python organize_dataset.py --dir_in=GOPRO_Large --dir_out=images
```

## Deblur customized images:

Save the images you want to deblur in a folder called `own` inside the `images` folder

```
python main.py --customized=1 --is_train=0 --model_name=./model/your_model_name
```

## Train a GAN model:
```
python main.py 
```

## Test model and evaluation 
```
python main.py --is_train=0 --testing_image=-1 --model_name=./model/your_model_name
```

## Pre-trained model
We saved two models that we pre-trained for about 20 epoches. 

Get the [model](https://drive.google.com/drive/folders/1kkcD8GRtkKO720eh9nFNFHD4UBb0vBBG?usp=sharing).

Details about the two models can be found in `model_log.txt`. 

Note that you need to log in your **LionMail@Columbia** to get access to the models.

The directory name of models should be `model`.

## Example of Result
From left to right: Sharp, blurred, deblurred

![image](https://github.com/yl3829/deblur_tf/blob/master/examples/7200_0.png)

See examples in `examples`.


## Statement of contributions:

Yilin Lyu: Understanding the novel Deblur GAN model and implemented in Tensorflow, trained and tested. 

Yiyi Zhang:  Model training testing and evaluating, hyperparameters searching, web application development. 

LIangliang Huang: Understanding the other basic GAN model for deblurring and implemented in Keras,  SQL utilization.    

Shan Guan: Understanding the other basic GAN model for deblurring and implemented in Keras ,SQL utilization.  
