#%%
import tensorflow as tf
import numpy as np
from vgg import VGG19
from pipeline import *

# def content_loss(vggTrain, vggRef, weight):
#     with tf.variable_scope('content_loss'):
#         Y = vggRef.content_layer
#         X = vggTrain.content_layer
#         size = tf.size(X)
#         return weight * tf.nn.l2_loss(X - Y) * 2 / tf.to_float(size)

def gram(X):
    with tf.variable_scope('gram'):
        print(X.get_shape())
        m, h, w, c = X.get_shape().as_list()
        m = 1 # wtf!
        X = tf.reshape(X, tf.stack([m, -1, c]))
        return tf.matmul(X, X, transpose_a=True) / tf.to_float(w*h*c)

def style_loss(vggTrain, style_grams, weight):
    with tf.variable_scope('style_loss'):
        loss = 0
        for i in range(len(vggTrain.style_layers)):
            with tf.variable_scope('style_loss_layer_'+str(i)):
                X = vggTrain.style_layers[i]
                size = tf.size(X)
                Y = style_grams[i]
                loss += tf.nn.l2_loss(gram(X) - Y) * 2 / tf.to_float(size)
    return loss * weight

def total_loss(vggTrain, style_grams, params):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.variable_scope('losses'):
        # J_content = content_loss(vggTrain, vggRef, params.content_weight)
        J_style = style_loss(vggTrain, style_grams, params.style_weight)
        # J_tv = tv_loss(generator.output, params.tv_weight)
        #with tf.variable_scope('total_loss'):
        total_loss =  J_style #+ J_content + J_tv
        with tf.variable_scope('optimizer'):
            train_step = tf.train.AdamOptimizer(params.learn_rate).minimize(total_loss, global_step=global_step)
        return total_loss, train_step, J_style, global_step #J_content, J_style, 

def eval_style(params):
    with tf.Session() as sess:
        with tf.variable_scope('eval_style'):
            M = process_mask(params.mask_path, params.num_colors)
            M = tf.constant(M, dtype=tf.float32, name='style_mask')
            h, w, c = M.get_shape()
            
            X = process_img(params.style_path, (h, w, 3))
            X = tf.expand_dims(X, 0)
            M = tf.stack([M])
            vggRef = VGG19(X, M, 'style_vgg')
            style_layers = [gram(l) for l in vggRef.style_layers]
            # return X, sess.run(style_layers), (h, w)
            return style_layers, (h, w)


# from tests import *
# from params import TrainingParams
# from models import SpriteGenerator

# def test_model(sess):

#     # tf.reset_default_graph()

#     params = TrainingParams()
#     Y, style_grams, input_shape = eval_style(params)

#     # tf.reset_default_graph()

#     # M is an example of a doodle
#     # for training, it is randomly generated using diamond square
#     M = tf.constant(generate_mask(params.num_colors, shape=input_shape), name='random_map', dtype=tf.float32)
#     R = tf.stack([M]) # batch them
#     # the randomly generated M is then given to the generator network to be transformed into the 
#     # stylized artwork
#     generator = SpriteGenerator(R,'Gen')
    
#     # the output of the generator is then graded by the VGG19
#     train = VGG19(generator.output, M, 'train')

#     with tf.variable_scope('losses'):
#         loss = style_loss(train, style_grams, 1.0)

# summarize(test_model)  
# test_model(None)