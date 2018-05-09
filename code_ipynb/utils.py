import tensorflow as tf 
import numpy as np
import os
from PIL import Image
import math

def res_block(input, filters, training, use_dropout=False):
    
    padded_input = tf.pad( input, [ [0, 0], [1, 1], [1, 1], [0, 0] ], mode="REFLECT" )
    
    _out = tf.layers.conv2d(padded_input, filters=filters, kernel_size=(3, 3), strides=(1, 1), padding = 'VALID')
    _out = tf.layers.batch_normalization(_out, training=training)
    _out = tf.nn.relu(_out)
    if use_dropout:
        _out = tf.nn.dropout( _out, keep_prob = 0.5 )
    
    _out = tf.pad( _out, [ [0, 0], [1, 1], [1, 1], [0, 0] ], mode="REFLECT" )
    _out = tf.layers.conv2d(_out, filters=filters, kernel_size=(3, 3), strides=(1, 1), padding = 'VALID')
    _out = tf.layers.batch_normalization(_out, training=training)
    
    result = tf.add( _out, input )
    
    return result


def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def list_image_files(directory):
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]

def load_image(path):
    img = Image.open(path)
    return img

def preprocess_image(cv_img, reshape=True):
    if reshape:
        RESHAPE=(256,256)
        cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')

def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)
    
def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }

def load_own_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    all_paths= list_image_files(path)
    images=[]
    for path_A in all_paths:
        img=load_image(path_A)
        images.append(preprocess_image(img,reshape=False))
        if len(images) > n_images - 1: break

    return np.array(images)

def PSNR(img1, img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



