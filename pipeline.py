#%%
import tensorflow as tf
import imageio
import numpy as np
import diamond as DS
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# def process_map_tf(path, colors):
#     img = tf.read_file(path)
#     img = tf.image.decode_png(img, channels=3)
#     shape = img.get_shape()
#     img = tf.reshape(img, [-1,3])
#     km = tf.contrib.factorization.KMeansClustering(num_clusters=colors)
#     img = km.train(lambda:img).predict(img)
#     img = tf.one_hot(img, colors)
#     img = tf.reshape(img, [-1, img[0], img[1]])
#     return img

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
    # hmap = hmap.transpose([0, 2, 1])
    return hmap

# tf.reset_default_graph()
# # img = process_mask('data/style_mask.jpg', 4)
# # print(img.shape)

# hmap = generate_mask(4)
# imageio.imwrite('data/output.jpg', mask)
# mask = np.sum(mask, axis=2)
# mask = np.array(mask).ravel()[:, None]
# print(hmap.shape)