"""
This is the implementation of multi-exposure DL enhanced phase retrieval network based on VGG.
"""
import tensorflow as tf

# ------------------------------- Layers part -------------------------------
class ConvBnAct(tf.keras.layers.Layer):
    def __init__(
            self,
            filters=64,
            kernel_size=(3,3),
            activation='relu',
            padding='same',
            name='conv'):
        super().__init__(name="ConvBnAct")

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            name=name)
        # self.norm = BatchNormalization()
        self.norm = tf.keras.layers.BatchNormalization(name='BatchNorm')

    def call(self, input, training):
        x = self.conv(input)
        x = self.norm(x,training=training)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, suffix):
        super(Encoder, self).__init__()
        self.conv1 = ConvBnAct(32, name='{}_conv1'.format(suffix))
        self.conv2 = ConvBnAct(64, name='{}_conv2'.format(suffix))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
        # self.conv3 = ConvBnAct(128, name='{}_conv3'.format(suffix))
        # self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

    def call(self, input, training):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        x = self.pool1(x)
        # x = self.conv3(x, training)
        # x = self.pool2(x)
        return x


class Block_1(tf.keras.layers.Layer):
    def __init__(
            self):
        super(Block_1, self).__init__(name="Block_1")
        self.conv1 = ConvBnAct(64,name='block1_conv1')
        self.conv2 = ConvBnAct(64,name='block1_conv2')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    def call(self, input,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.pool(x)
        return x

class Block_2(tf.keras.layers.Layer):
    def __init__(
            self):
        super(Block_2, self).__init__(name="Block_2")
        self.conv1 = ConvBnAct(128,name='block2_conv1')
        self.conv2 = ConvBnAct(128,name='block2_conv2')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    def call(self, input,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.pool(x)
        return x

class Block_3(tf.keras.layers.Layer):
    def __init__(
            self):
        super(Block_3, self).__init__(name="Block_3")
        self.conv1 = ConvBnAct(256,name='block3_conv1')
        self.conv2 = ConvBnAct(256,name='block3_conv2')
        self.conv3 = ConvBnAct(256,name='block3_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

    def call(self, input ,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class Block_4(tf.keras.layers.Layer):
    def __init__(
            self):
        super(Block_4, self).__init__(name="Block_4")
        self.conv1 = ConvBnAct(512,name='block4_conv1')
        self.conv2 = ConvBnAct(512,name='block4_conv2')
        self.conv3 = ConvBnAct(512,name='block4_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

    def call(self, input,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class Block_5(tf.keras.layers.Layer):
    def __init__(
            self):
        super(Block_5, self).__init__(name="Block_5")
        self.conv1 = ConvBnAct(512,name='block5_conv1')
        self.conv2 = ConvBnAct(512,name='block5_conv2')
        self.conv3 = ConvBnAct(512,name='block5_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

    def call(self, input,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class VGG_PR(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(VGG_PR, self).__init__()
        self.block1 = Block_1()
        self.block2 = Block_2()
        self.block3 = Block_3()
        self.block4 = Block_4()
        self.block5 = Block_5()
        self.avg = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', name='fc1')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', name='fc2')
        self.fc3 = tf.keras.layers.Dense(num_classes,activation='linear',name='predictions')

    def call(self, input, training):
        x = self.block1(input,training)
        x = self.block2(x,training)
        x = self.block3(x,training)
        x = self.block4(x,training)
        x = self.block5(x,training)
        x = self.avg(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class VGG_multi(tf.keras.Model):
    def __init__(self, num_classes):
        super(VGG_multi, self).__init__()
        self.e1 = Encoder("encode1")
        self.e2 = Encoder("encode2")
        self.e3 = Encoder("encode3")
        self.vgg_base = VGG_PR(num_classes)

    def call(self, input1, input2, input3, training=False):
        x1 = self.e1(input1, training)
        x2 = self.e2(input2, training)
        x3 = self.e3(input3, training)
        x = tf.keras.layers.concatenate([x1, x2, x3])
        x =  self.vgg_base(x, training)
        # print("---------------output shape:{}".format(x.shape))
        return x
