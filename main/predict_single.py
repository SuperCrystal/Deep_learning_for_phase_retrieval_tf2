''' Predict script for single-input phase retrieval'''

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
# from mobilev3 import MobileNetV3Large
from vgg_pr import VGG_PR
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

test_img_list = sorted(glob.glob('../dataset/test/intensity/*.mat'))
test_label_list = sorted(glob.glob('../dataset/test/phase/*.txt'))
ckpt_path = '../checkpoints/notes/8/VGG-{epoch}.ckpt'.format(epoch=12)
result_path = '../result/single/'
if not os.path.exists(result_path):
    os.mkdir(result_path)

# read data
####################### 利用tf.data高级API进行数据读取 ########################
def read_img(filename):
    # print(filename)
    image_dict = loadmat(filename.decode('utf-8'))
    # process 3 降采样，可以提升batch_size
    exp_thresh = 1e4
    image_decoded = image_dict['Iz']
    image_decoded = cv2.resize(image_decoded, img_size, interpolation=cv2.INTER_AREA)
    image_decoded[image_decoded>exp_thresh] = exp_thresh
    image_decoded /= exp_thresh
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

def parse_function(image_filename, label_filename):
    # print('--------------{}-------------------'.format(tf.as_string(image_filename))) # filename变成tensor了？
    # img1, img2, img3 = tf.numpy_function(read_multi_imgs, [image_filename1, image_filename2, image_filename3], tf.float32)
    img = tf.numpy_function(read_img, [image_filename], tf.float32)
    label = tf.numpy_function(read_label, [label_filename], tf.float32)
    return img, label


def predict():
    p_rmse = tf.metrics.Mean(name="prediction_rmse")
    logging.basicConfig(level=logging.INFO)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img_list, test_label_list))
    test_dataset = test_dataset.map(parse_function,3).batch(1)
    model = VGG_PR(num_classes=num_label)
    model.load_weights(ckpt_path)
    logging.info("model loaded")
    predictions = []
    for img, labels in test_dataset:
        prediction = model(img, training=False)
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
