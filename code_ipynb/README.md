# APMAE4990 PROJECT - Deblur Images  
The jupyter notebook [**`main.ipynb`**](main.ipynb) contains the implementation of the other approach introduced in [DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks](https://arxiv.org/pdf/1711.07064.pdf) and the evaluation of the two approaches.  

## Dataset:

Get the [GOPRO dataset](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing), and extract it in the working directory. The directory name of dataset should be `GOPRO_Large`.

## Use the dataset:
```
python organize_dataset.py --dir_in=GOPRO_Large --dir_out=images
```

## Deblur customized images:

Save the images you want to deblur in the [`images/own`](images/own) folder.

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

See examples in [`examples`](examples).
