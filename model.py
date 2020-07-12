#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 
import numpy as np 
import tensorflow as tf
import tensorflow.keras.layers as layers

class DepthConv(layers.Layer):
	def __init__(self, pointwise_conv_filters, alpha=1.0,
                     depth_multiplier=1, strides=(1, 1), block_id=1, idx="channels_last"):
            super(DepthConv, self).__init__()
            self.strides = strides
            self.block_id = block_id
            self.depth_multiplier = depth_multiplier
            self.pointwise_conv_filters = pointwise_conv_filters
            self.idx = idx
            self.alpha = alpha

		
	def call(self, inputs):
            self.channel_axis = 1 if self.idx == 'channels_first' else -1
            self.pointwise_conv_filters = int(self.pointwise_conv_filters * self.alpha)

            if self.strides == (1, 1):
                x = inputs
            else:
                x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                         name='conv_pad_%d' % self.block_id)(inputs)

            x = layers.DepthwiseConv2D((3, 3),
                                       padding='same' if self.strides == (1, 1) else 'valid',
                                       depth_multiplier=self.depth_multiplier,
                                       strides=self.strides,
                                       use_bias=False,
                                       name='conv_dw_%d' % self.block_id)(x)
            x = layers.BatchNormalization(
                    axis=self.channel_axis, name='conv_dw_%d_bn' % self.block_id)(x)
            x = layers.ReLU(6., name='conv_dw_%d_relu' % self.block_id)(x)

            x = layers.Conv2D(self.pointwise_conv_filters, (1, 1),
                                              padding='same',
                                              use_bias=False,
                                              strides=(1, 1),
                                              name='conv_pw_%d' % self.block_id)(x)
            x = layers.BatchNormalization(axis=self.channel_axis,
                                                                      name='conv_pw_%d_bn' % self.block_id)(x)
            return layers.ReLU(6., name='conv_pw_%d_relu' % self.block_id)(x)

class Conv(layers.Layer):
	def __init__(self, filters, alpha, channel_idx, kernel=(3, 3), strides=(1, 1)):
            super(Conv, self).__init__()
            self.filters = filters
            self.alpha = alpha
            self.kernel = kernel
            self.strides = strides
            self.channel_idx = channel_idx

	def call(self, inputs):
            # if self.channel_idx == "channels_first":
            # 	self.channel_axis = 1
            # else:
            # 	self.channel_axis = -1

            self.channel_axis = -1
            filters = int(self.filters * self.alpha)
            print(f"Received Input with shape: {inputs.shape}")
            x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
            x = layers.Conv2D(filters=self.filters, 
                              kernel_size=self.kernel,
                              padding='valid',
                              use_bias=False,
                              strides=self.strides,
                              name='conv1')(x)
            x = layers.BatchNormalization(axis=self.channel_axis, name='conv1_bn')(x)
            return layers.ReLU(6., name='conv1_relu')(x)


class MobileNet(tf.keras.Model):
	"""Implementation of MobileNet in Python"""
	def __init__(self,
                    input_shape=None,
                    alpha=1.0,
                    depth_multiplier=1,
                    dropout=1e-3,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=26,
                    strides=(1,1),
                    idx="channels_last",
                    **kwargs):		
            super(MobileNet, self).__init__()

            # self.input_shape = input_shape
            # self.strides = strides
            self.alpha = alpha            
            self.depth_multiplier = depth_multiplier 
            self.idx=idx
            self.dropout = dropout
            # self.include_top = include_top
            # self.weights = weights
            # self.input_tensor = input_tensor
            # self.pooling = pooling
            self.classes = classes


	def call(self, img_input):
            x = Conv(32, self.alpha, strides=(2, 2), channel_idx=self.idx)(img_input)
            x = DepthConv(64, self.alpha, self.depth_multiplier, block_id=1)(x)

            x = DepthConv(128, self.alpha, self.depth_multiplier,
                                                              strides=(2, 2), block_id=2)(x)
            x = DepthConv(128, self.alpha, self.depth_multiplier, block_id=3)(x)

            x = DepthConv(256, self.alpha, self.depth_multiplier,
                                                              strides=(2, 2), block_id=4)(x)
            x = DepthConv(256, self.alpha, self.depth_multiplier, block_id=5)(x)

            x = DepthConv(512, self.alpha, self.depth_multiplier,
                                                              strides=(2, 2), block_id=6)(x)
            x = DepthConv(512, self.alpha, self.depth_multiplier, block_id=7)(x)
            x = DepthConv(512, self.alpha, self.depth_multiplier, block_id=8)(x)
            x = DepthConv(512, self.alpha, self.depth_multiplier, block_id=9)(x)
            x = DepthConv(512, self.alpha, self.depth_multiplier, block_id=10)(x)
            x = DepthConv(512, self.alpha, self.depth_multiplier, block_id=11)(x)

            x = DepthConv(1024, self.alpha, self.depth_multiplier,
                                                              strides=(2, 2), block_id=12)(x)
            x = DepthConv(1024, self.alpha, self.depth_multiplier, block_id=13)(x)

            if self.idx == 'channels_first':
                shape = (int(1024 * self.alpha), 1, 1)
            else:
                shape = (1, 1, int(1024 * self.alpha))

            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Reshape(shape, name='reshape_1')(x)
            x = layers.Dropout(self.dropout, name='dropout')(x)
            x = layers.Conv2D(self.classes, (1, 1),
                              padding='same',
                              name='conv_preds')(x)
            x = layers.Reshape((self.classes,), name='reshape_2')(x)
            x = layers.Activation('softmax', name='act_softmax')(x)

            return x





### Testing Model ###

if __name__ == "__main__":
    import cv2
    model = MobileNet(classes=26, idx="channels_last")
    model.compile(optimizer='adam', loss="categorical_crossentropy")
    img = cv2.imread('yeet.png')
    imgs = np.array([img])/255.0
    print(imgs.shape)
    pred = np.argmax(model.predict(imgs))
    print(f"Prediction: {pred}")
