import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pix2pix import Pix2pix

# plots complete testset

A = np.load('dataset_y_test.npy') # Architectural labels
B = np.load('dataset_x_test.npy') # Photo

with tf.Graph().as_default() as graph:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        model = Pix2pix(256, 256, ichan=3, ochan=3) # creating net
        saver = tf.train.Saver()
        saver.restore(sess, "models/model.ckpt") # loading last checkpoint
        for i in range(len(A)):
            pred = model.sample_generator(sess, np.expand_dims(A[i], axis=0))[0]
            combined = np.concatenate([A[i], B[i], pred], axis=1)
            plt.imshow(combined)
            plt.axis('off')
            plt.savefig('images/pred/img_%d_combined.jpg' % i)
            plt.close()

            plt.imshow(pred)
            plt.axis('off')
            plt.savefig('images/pred/img_%d.jpg' % i)
            plt.close()