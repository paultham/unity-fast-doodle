#%%

import tensorflow as tf
from pipeline import *
from models import *
from vgg import *
from losses import *  
from params import TrainingParams

class TrainStep:
    def __init__(self, gs, tl, sl):
        self.train_step = gs
        self.total_loss = tl
        self.style_loss = sl
        self.step_count = 0

    def run_step(self, step_context):
        # if(self.step_count > 10):
        #     step_context.request_stop()
        # self.step_count += 1
        
        print('step...')
        return step_context.run_with_hooks([self.train_step, self.total_loss, self.style_loss])

def train(params, start_new=False):

    tf.reset_default_graph()

    params = TrainingParams()
    style_grams, input_shape = eval_style(params)
    
    # tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # M is an example of a doodle
    # for training, it is randomly generated using diamond square
    # M = tf.constant(generate_mask(params.num_colors, shape=input_shape), name='random_map', dtype=tf.float32)
    # R = tf.stack([M]) # batch them
    R = create_tf_pipeline(params)
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

    trainer = TrainStep(train_step, J, J_style)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=params.save_path, 
        log_step_count_steps=params.log_step,
        save_summaries_steps=params.summary_step
        ) as sess:
        while not sess.should_stop():
            _, total_cost, style_cost = sess.run_step_fn(trainer.run_step)
    
    print('Done...')

train(TrainingParams(), False)