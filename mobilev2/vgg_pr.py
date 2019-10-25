"""
Implementation of vgg based pr method using Tensorflow 2.0
"""
import tensorflow as tf

# ------------------------------- Layers part -------------------------------
class BatchNormalization(tf.keras.layers.Layer):
    """All our convolutional layers use batch-normalization
    layers with average decay of 0.99.
    """

    def __init__(self):
        super().__init__(name="BatchNormalization")
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=0.99,
            name="BatchNorm")

    def call(self, input, training):
        return self.bn(input, training)

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
        self.norm = BatchNormalization()

    def call(self, input, training):
        x = self.conv(input)
        x = self.norm(x,training=training)
        return x

class Block_1(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_1")
        self.conv1 = ConvBnAct(64,name='block1_conv1')
        self.conv2 = ConvBnAct(64,name='block1_conv2')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    def call(self, input,training):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.pool(x)
        return x

class Block_2(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_2")
        self.conv1 = ConvBnAct(128,name='block2_conv1')
        self.conv2 = ConvBnAct(128,name='block2_conv2')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    def call(self, input,training):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.pool(x)
        return x

class Block_3(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_3")
        self.conv1 = ConvBnAct(256,name='block3_conv1')
        self.conv2 = ConvBnAct(256,name='block3_conv2')
        self.conv3 = ConvBnAct(256,name='block3_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

    def call(self, input ,training):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class Block_4(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_4")
        self.conv1 = ConvBnAct(512,name='block4_conv1')
        self.conv2 = ConvBnAct(512,name='block4_conv2')
        self.conv3 = ConvBnAct(512,name='block4_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

    def call(self, input,training):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class Block_5(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_5")
        self.conv1 = ConvBnAct(512,name='block5_conv1')
        self.conv2 = ConvBnAct(512,name='block5_conv2')
        self.conv3 = ConvBnAct(512,name='block5_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

    def call(self, input,training):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class VGG_PR(tf.keras.Model):
    def __init__(self,num_classes):
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
        # print("output1:{}".format(x))
        x = self.fc1(x)
        # print("output2:{}".format(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x
