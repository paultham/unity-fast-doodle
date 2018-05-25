#%%
import tensorflow as tf
import imageio
import numpy as np
import diamond as DS
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from params import *

def process_map_tf(path, colors):
    img = tf.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    shape = img.get_shape()
    img = tf.reshape(img, [-1,3])
    km = tf.contrib.factorization.KMeansClustering(num_clusters=colors)
    img = km.train(lambda:img).predict(img)
    img = tf.one_hot(img, colors)
    img = tf.reshape(img, [-1, img[0], img[1]])
    return img

def process_mask(path, colors):
    img = imageio.imread(path)[..., :3] # ignore alpha
    shape = img.shape # remember original shape
    img = np.reshape(img, [-1, 3]) # flatten w and h
    kmeans = KMeans(n_clusters=colors).fit(img)
    img = kmeans.predict(img)[:, None] # cluster to labels
    img = img.reshape([shape[0], shape[1]]) # back to hxw
    img = OneHotEncoder(n_values=colors, sparse=False).fit_transform(img) # separate
    img = img.reshape([shape[0], shape[1], colors])
    # img = img.transpose([0, 2, 1])
    return img

def process_img(path, shape=None, crop=False):
    img = tf.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    if shape is not None:
        img.set_shape(shape)
    return tf.to_float(img)

def generate_mask(colors, shape = (256,256)):
    hmap = np.stack([np.array(DS.diamond_square(shape, -1, 1, 0.35)),
            np.array(DS.diamond_square(shape, -1, 1, 0.55)),
            np.array(DS.diamond_square(shape, -1, 1, 0.75))],
            axis=2)
    hmap = np.sum(hmap, axis=2)
    hmap = np.array(hmap).ravel()[:, None]
    kmeans = KMeans(n_clusters=colors).fit(hmap)
    hmap = kmeans.predict(hmap)[:,None]
    hmap = OneHotEncoder(n_values=colors, sparse=False).fit_transform(hmap) # separate
    hmap = hmap.reshape([shape[0], shape[1], colors])
    return hmap

def _img_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def make_one_example(img):
    feature={'img':_img_feature(img)}
    return tf.train.Example(features=tf.train.Features(feature=feature))

def make_all_examples(params, filecount=100, count=100):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    for f in range(filecount):

        writer = None
        for i in range(count):
            if writer is None:
                path = "%s%i" % (params.train_path, f)
                writer = tf.python_io.TFRecordWriter(path)

            print('Mask %i  of %i' % (i, count))
            mask = generate_mask(params.num_colors, params.input_shape[:2])
            writer.write(make_one_example(mask).SerializeToString())

        writer.close()
        writer = None

def process_tf(x, num_colors, shape=None):
    parsed_features = tf.parse_single_example(x, features={
        'img':tf.FixedLenFeature([shape[0]*shape[1]*num_colors], dtype=tf.float32)
    })
    imgs = parsed_features['img']
    imgs = tf.reshape(imgs, shape + [num_colors])
    return imgs

def create_tf_pipeline(params):
    file_paths = ["%s%i" % (params.train_path, f) for f in range(params.num_train_files) ]
    files = tf.data.TFRecordDataset(file_paths)
    files = files.map(lambda x: process_tf(x, params.num_colors, params.input_shape[0:2]), num_parallel_calls=params.read_thread)
#     files = files.shuffle(params.total_train_sample)
    files = files.take(params.total_train_sample)
    files = files.batch(params.batch_size)
    files = files.repeat(params.num_epoch)
    files_iterator = files.make_one_shot_iterator()
    next_files = files_iterator.get_next()
    return next_files

#make_all_examples(TrainingParams(), filecount=10, count=10) 

# tf.reset_default_graph()
# sess = tf.InteractiveSession()
# masks = create_tf_pipeline(TrainingParams())
# for i in range(10):
#     value = sess.run(masks)
#     print(value.shape)

# tf.reset_default_graph()
# # img = process_mask('data/style_mask.jpg', 4)
# # print(img.shape)

# hmap = generate_mask(4)
# imageio.imwrite('data/output.jpg', mask)
# mask = np.sum(mask, axis=2)
# mask = np.array(mask).ravel()[:, None]
# print(hmap.shape)