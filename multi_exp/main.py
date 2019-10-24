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
from tensorflow.keras.callbacks import TensorBoard
import logging

# 参数配置
img_size = (299,299)
batch_size = 2
num_label = 20
initial_lr = 0.00001
total_epoch = 20
# 编号
case_num = 3

train_img_list = glob.glob('E:/00_PhaseRetrieval/PhENN/dataset/train/intensity/*.mat')
train_label_list = glob.glob('E:/00_PhaseRetrieval/PhENN/dataset/train/phase/*.txt')
val_img_list = glob.glob('E:/00_PhaseRetrieval/PhENN/dataset/validate/intensity/*.mat')
val_label_list = glob.glob('E:/00_PhaseRetrieval/PhENN/dataset/validate/phase/*.txt')
ckpt_path = 'E:/00_PhaseRetrieval/PhENN/checkpoints/IVR2-{epoch}.ckpt'
# tb_path = 'E:/00_PhaseRetrieval/PhENN/tensorboard/'
log_path = 'E:/00_PhaseRetrieval/PhENN/log/{}/'
if not os.path.exists(log_path.format(case_num)):
    os.mkdir(log_path.format(case_num))

# read data
####################### 利用tf.data高级API进行数据读取 ########################
def read_img(filename):
    image_dict = loadmat(filename.decode('utf-8'))
    # process 1 原图归一化
    # image_decoded = image_dict['Iz'] /4e-4  # 归一化
    # image_resized = np.float32(np.expand_dims(image_decoded, axis=-1))
    ### process 2 过曝图归一化
    image_decoded = image_dict['Iz']
    image_decoded[image_decoded>2e-4] = 2e-4
    image_decoded /= 2e-4
    image_resized = np.float32(np.expand_dims(image_decoded, axis=-1))
    # process 3 降采样，可以提升batch_size
    image_decoded = image_dict['Iz']
    image_decoded = cv2.resize(image_decoded, img_size, interpolation=cv2.INTER_AREA)
    image_decoded[image_decoded>2e-4] = 2e-4
    image_decoded /= 2e-4
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

# plt.figure()
# img = read_img('E:/00_PhaseRetrieval/PhENN/dataset/train/intensity/image000010.mat')
# print(tf.reduce_max(img))
# plt.imshow(tf.squeeze(np.log(img)), cmap='gray')
# plt.show()
# print(read_label('E:/00_PhaseRetrieval/PhENN/dataset/train/phase/image000010.txt'))

# 这个函数将作为map的输入，因此尽管label看似输入后没有用到，但也必须写进来
def parse_function(image_filename, label_filename):
    # print('--------------{}-------------------'.format(tf.as_string(image_filename))) # filename变成tensor了？
    img = tf.numpy_function(read_img, [image_filename], tf.float32)
    label = tf.numpy_function(read_label, [label_filename], tf.float32)
    return img, label

def train():
    # tf.keras.backend.set_learning_phase(True)
    # tf.logging.set_verbosity(tf.logging.INFO) #设置输出的最低的信息级别
    logging.basicConfig(level=logging.INFO)
    tdataset = tf.data.Dataset.from_tensor_slices((train_img_list, train_label_list))
    tdataset = tdataset.map(parse_function, 3).shuffle(buffer_size=5).batch(batch_size)
    vdataset = tf.data.Dataset.from_tensor_slices((val_img_list, val_label_list))
    vdataset = vdataset.map(parse_function, 3).batch(batch_size)

    # base_model = tf.keras.applications.MobileNetV2
    # base_model = MobileNetV3Large(classes=num_label)
    # model = base_model
    base_model = tf.keras.applications.InceptionResNetV2(weights=None, include_top=False, input_shape=(512,512,1))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    predictions = layers.Dense(num_label, activation='linear')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    # tf.keras.utils.plot_model(base_model,'mb3_model.png',show_shapes=True) # 无法打印？
    logging.info('Model loaded')
    # print('model loaded')

    start_epoch = 0
    # 该函数返回 the full path to the latest checkpoint，是string
    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        logging.info('training from scratch since weights no there')

    ######## 用 fit 进行训练 ########
    # model.compile(
    #     optimizer = 'adam',
    #     loss = 'mean_squared_error',
    #     metrics = [])
    # # 下面这种方法使我们在ctrl+C终止程序时能够自动保存一次参数
    # try:
    #     model.fit(
    #         tdataset,
    #         epochs=50,
    #         steps_per_epoch=200)
    # except KeyboardInterrupt:
    #     model.save_weights(ckpt_path.format(epoch=0))
    #     logging.info('keras fit model saved')
    # model.save_weights(ckpt_path.format(epoch=0)) # fit好像无法获取到进行到多少epoch？

    ######## 用自定义loop进行训练 ########
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    train_loss = tf.metrics.Mean(name='train_loss') # 表示对所有训练损失求平均
    val_loss = tf.metrics.Mean(name='val_loss')
    writer = tf.summary.create_file_writer(log_path.format(case_num))

    with writer.as_default():
        for epoch in range(start_epoch, total_epoch):
            print('start training')
            try:
                for batch, data in enumerate(tdataset):
                    images, labels = data
                    with tf.GradientTape() as tape:
                        pred = model(images, training=True)
                        # pred = tf.squeeze(pred)
                        loss = loss_object(labels, pred)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    train_loss(loss)
                    # print('-----------pred: {}--------------'.format(pred))
                    # print(pred.shape)
                    # print(labels.shape)

                    if batch % 20 ==0:
                        # result() computes and returns the metric value tensor.
                        logging.info('Epoch: {}, iter: {}, loss:{}'.format(epoch, batch, train_loss.result()))
                    tf.summary.scalar('train_loss', train_loss.result(), step=epoch*5000+batch)
                    tf.summary.text('Zernike_coe_pred', tf.as_string(tf.squeeze(pred)), step=epoch*5000+batch)
                    tf.summary.text('Zernike_coe_gt', tf.as_string(tf.squeeze(labels)), step=epoch*5000+batch)
                    # tf.summary.image('input_intensity', tf.math.log(images), step=epoch*5000+batch, max_outputs=1)
                    writer.flush()
                model.save_weights(ckpt_path.format(epoch=epoch))
            except KeyboardInterrupt:
                logging.info('interrupted.')
                model.save_weights(ckpt_path.format(epoch=epoch))
                logging.info('model saved into {}'.format(ckpt_path.format(epoch=epoch)))
                exit(0) # 无异常退出
            for images, labels in vdataset:
                val_pred = model(images, training=False)
                v_loss = loss_object(labels, val_pred)
                val_loss(v_loss)
            logging.info('Epoch: {}, val_loss: {}'.format(epoch, val_loss.result()))
            tf.summary.scalar('val_loss', val_loss.result(), step = epoch)
            writer.flush()
        model.save_weights(ckpt_path.format(epoch=epoch))
        # print('------------------ this is the end ------------------')

if __name__ == "__main__":
    train()
