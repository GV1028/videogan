"""
Author: Vignesh Gokul
Code structure inspired from https://github.com/carpedm20/DCGAN-tensorflow

"""
import os
import scipy.misc
import numpy as np

from model import VideoGAN

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train [25]")
flags.DEFINE_integer("video_dim", [32,64,64,3], "The dimension of each video, must be of shape [frames, height, width, channels]")
flags.DEFINE_integer("zdim", 100, "The dimension of latent vector")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("checkpoint_file", None, "The checkpoint file name")
FLAGS = flags.FLAGS

def main(_):
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./genvideos"):
        os.makedirs("./genvideos")
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto(    )
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        videogan = VideoGAN(sess,video_dim = FLAGS.video_dim,zdim = FLAGS.zdim,batch_size = FLAGS.batch_size,epochs=FLAGS.epoch,checkpoint_file = FLAGS.checkpoint_file)
        videogan.build_model()
        videogan.train()

if __name__ == '__main__':
    tf.app.run()
