#%%
import tensorflow as tf

def summarize(test_fn):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    test_fn(sess)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('summaries', sess.graph)
    sess.run(tf.global_variables_initializer())
    print('Done summarizing')