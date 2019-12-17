''' Predict script for multi-input phase retrieval'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
import glob
from tensorflow.keras.layers import BatchNormalization, Conv2D, ReLU, Conv2DTranspose, add, concatenate
from scipy.io import loadmat
import numpy as np
from mobilev3 import MobileNetV3Large
from vgg_multi_input import VGG_multi
from tensorflow.keras.callbacks import TensorBoard
import logging
import cv2

# 参数配置
img_size = (299,299)
# batch_size = 4
num_label = 20
total_epoch = 100
exp_thresh = [0.1e4,0.5e4,3e4]
# 编号
# case_num = 10

os.chdir(os.getcwd())

# train_img_list_1 = sorted(glob.glob('../dataset/exp_train/intensity_1/*.mat'))
# train_img_list_2 = sorted(glob.glob('../dataset/exp_train/intensity_2/*.mat'))
# train_img_list_3 = sorted(glob.glob('../dataset/exp_train/intensity_3/*.mat'))
# train_label_list = sorted(glob.glob('../dataset/train/phase/*.txt'))
# val_img_list_1 = sorted(glob.glob('../dataset/validate/intensity/*.mat'))
# val_label_list = sorted(glob.glob('../dataset/validate/phase/*.txt'))
#无噪声
test_img_list_1 = sorted(glob.glob('../dataset/exp_test/intensity_1/*.mat'))
test_img_list_2 = sorted(glob.glob('../dataset/exp_test/intensity_2/*.mat'))
test_img_list_3 = sorted(glob.glob('../dataset/exp_test/intensity_3/*.mat'))
test_label_list = sorted(glob.glob('../dataset/exp_test/phase/*.txt'))
ckpt_path = '../checkpoints_multi/notes/11/VGG_multi-{epoch}.ckpt'.format(epoch=40)
result_path = '../checkpoints_multi/notes/11/'
# 加噪声
# test_img_list_1 = sorted(glob.glob('../dataset/noise_test/intensity_1/*.mat'))
# test_img_list_2 = sorted(glob.glob('../dataset/noise_test/intensity_2/*.mat'))
# test_img_list_3 = sorted(glob.glob('../dataset/noise_test/intensity_3/*.mat'))
# test_label_list = sorted(glob.glob('../dataset/noise_test/phase/*.txt'))
# ckpt_path = '../checkpoints_multi/VGG_multi-{epoch}.ckpt'.format(epoch=99)
# result_path = '../result/noise_multi-epoch40+100-experi11/'
if not os.path.exists(result_path):
    os.mkdir(result_path)

# read data
####################### 利用tf.data高级API进行数据读取 ########################
def read_img(filename, exp_thresh, suffix):
    # print(filename)
    image_dict = loadmat(filename.decode('utf-8'), verify_compressed_data_integrity=False)
    # process 3 降采样，可以提升batch_size
    # exp_thresh = 1e4
    image_decoded = image_dict['Iz_{}'.format(suffix.decode('utf-8'))]
    image_decoded = cv2.resize(image_decoded, img_size, interpolation=cv2.INTER_AREA)
    # image_decoded[image_decoded>exp_thresh] = exp_thresh
    # image_decoded /= exp_thresh
    image_resized = np.float32(np.expand_dims(image_decoded, axis=-1))
    # image_resized = tf.convert_to_tensor(image_resized)
    return image_resized

def read_label(filename):
    label = open(filename).read()
    # print(label)
    label = label.strip().split(' ')
    label = [np.float32(i) for i in label if i!='']
    label = np.reshape(label, [1,1,-1])
    label = np.array(label) + 0.5  # 归一化
    # label = tf.convert_to_tensorf(label)
    return label

def parse_function(image_filename1, image_filename2, image_filename3, label_filename):
    # print('--------------{}-------------------'.format(tf.as_string(image_filename))) # filename变成tensor了？
    # img1, img2, img3 = tf.numpy_function(read_multi_imgs, [image_filename1, image_filename2, image_filename3], tf.float32)
    img1 = tf.numpy_function(read_img, [image_filename1, exp_thresh[0], 'exp1'], tf.float32)
    img2 = tf.numpy_function(read_img, [image_filename2, exp_thresh[1], 'exp2'], tf.float32)
    img3 = tf.numpy_function(read_img, [image_filename3, exp_thresh[2], 'exp3'], tf.float32)
    label = tf.numpy_function(read_label, [label_filename], tf.float32)
    return img1, img2, img3, label


def predict():
    p_rmse = tf.metrics.Mean(name="prediction_rmse")
    logging.basicConfig(level=logging.INFO)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img_list_1,test_img_list_2,test_img_list_3, test_label_list))
    test_dataset = test_dataset.map(parse_function,3).batch(1)
    model = VGG_multi(num_classes=num_label)
    model.load_weights(ckpt_path)
    logging.info("model loaded")
    predictions = []
    for img1, img2, img3, labels in test_dataset:
        prediction = model(img1, img2, img3, training=False)
        predictions.append(prediction)
        if len(prediction.shape) == 2:
            pred = tf.reshape(prediction,[-1, 1, 1, num_label])
        print("predicted results: {}".format(prediction))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(pred-labels)))
        p_rmse(rmse)
    print("Average rmse: {}".format(p_rmse.result()))
    np.savetxt(os.path.join(result_path, 'results.txt'), np.squeeze(np.array(predictions)), fmt='%.8f', delimiter=' ')

if __name__ == "__main__":
    predict()
