#%%

import tensorflow as tf
from pipeline import *
from models import *
from vgg import *
from losses import *  
from params import TrainingParams

def train(params, start_new=False):

    tf.reset_default_graph()

    params = TrainingParams()
    style_grams, input_shape = eval_style(params)
    
    # tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # M is an example of a doodle
    # for training, it is randomly generated using diamond square
    M = tf.constant(generate_mask(params.num_colors, shape=input_shape), name='random_map', dtype=tf.float32)
    R = tf.stack([M]) # batch them
    # the randomly generated M is then given to the generator network to be transformed into the 
    # stylized artwork
    generator = SpriteGenerator(R,'Gen')
    
    # the output of the generator is then graded by the VGG19
    train = VGG19(generator.output, M, 'train')

    # the total loss
    J, train_step, J_style, global_step = total_loss(train, style_grams, params)

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', J)
        tf.summary.scalar('style_loss', J_style)
    
    if start_new:
        print('Starting...')
        sess.run(tf.global_variables_initializer())
    else:
        print('Continuing...')

    # with tf.train.MonitoredTrainingSession(
    #     checkpoint_dir=params.save_path, 
    #     log_step_count_steps=params.log_step,
    #     save_summaries_steps=params.summary_step
    #     ) as sess:
    #     while not sess.should_stop():
    _, total_cost, style_cost = sess.run([train_step, J, J_style])

    print('Done...')

train(TrainingParams(), True)