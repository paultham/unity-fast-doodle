#%%
from keras.utils.data_utils import get_file
from models import *
from pipeline import *
import h5py

class VGG19Weights:
    def __init__(self):
        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.filename = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')

    def __enter__(self):
        self.open_file = h5py.File(self.filename, 'r')
        self.layer_names = self.open_file.attrs['layer_names']
        return self

    def __exit__(self, *args):
        if hasattr(self.open_file, 'close'):
            self.open_file.close()

    def get_weights(self, layer_id):
        #b'input_2', b'block1_conv1', b'block1_conv2', b'block1_pool',
        #b'block2_conv1', b'block2_conv2', b'block2_pool', 
        #b'block3_conv1', b'block3_conv2', b'block3_conv3', b'block3_conv4', b'block3_pool',
        #b'block4_conv1', b'block4_conv2', b'block4_conv3', b'block4_conv4', b'block4_pool', 
        #b'block5_conv1', b'block5_conv2', b'block5_conv3', b'block5_conv4', b'block5_pool'
        layer = self.open_file[self.layer_names[layer_id]]
        weight_names = layer.attrs['weight_names']
        W = layer[weight_names[0]].value
        B = layer[weight_names[1]].value
        return (W, B)

class VGG19:
    def __init__(self, X, M, name):
        self.style_layers = []
        with VGG19Weights() as w:
            with tf.name_scope(name):
                # M = tf.expand_dims(M, 0 )
                X = tf.reverse(X, [3])
                X = X - tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                with tf.name_scope('b1'):
                    X = relu(conv(X, 3, 64, weights=w.get_weights(1), bias=True))
                    self.style_layers.append(self.resize_m(X, M))
                    X = relu(conv(X, 3, 64, weights=w.get_weights(2), bias=True))
                    X = max_pool(X)
                with tf.name_scope('b2'):
                    X = relu(conv(X, 3, 128, weights=w.get_weights(4), bias=True))
                    self.style_layers.append(self.resize_m(X, M))
                    X = relu(conv(X, 3, 128, weights=w.get_weights(5), bias=True))
                    self.content_layer=X
                    X = max_pool(X)
                with tf.name_scope('b3'):
                    X = relu(conv(X, 3, 256, weights=w.get_weights(7), bias=True))
                    self.style_layers.append(self.resize_m(X, M))
                    X = relu(conv(X, 3, 256, weights=w.get_weights(8), bias=True))
                    X = relu(conv(X, 3, 256, weights=w.get_weights(9), bias=True))
                    X = relu(conv(X, 3, 256, weights=w.get_weights(10), bias=True))
                    X = max_pool(X)
                with tf.name_scope('b5'):
                    X = relu(conv(X, 3, 512, weights=w.get_weights(12), bias=True))
                    self.style_layers.append(self.resize_m(X, M))
                    X = relu(conv(X, 3, 512, weights=w.get_weights(13), bias=True))
                    X = relu(conv(X, 3, 512, weights=w.get_weights(14), bias=True))
                    X = relu(conv(X, 3, 512, weights=w.get_weights(15), bias=True))
                    X = max_pool(X)
                # with tf.name_scope('b6'):
                #     X = relu(conv(X, 3, 512, weights=w.get_weights(17), bias=True))
                #     X = relu(conv(X, 3, 512, weights=w.get_weights(18), bias=True))
                #     X = relu(conv(X, 3, 512, weights=w.get_weights(19), bias=True))
                #     X = relu(conv(X, 3, 512, weights=w.get_weights(20), bias=True))
                #     X = max_pool(X)

    def resize_m(self, X, M):
        _, h, w, _ = X.get_shape()
        # print(M.get_shape())
        M = tf.image.resize_images(M, [h, w])
        # I = tf.ones(shape=X.get_shape())
        # return tf.concat([X, I*M], 0)
        return tf.concat([X, M], 3)