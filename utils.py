import tensorflow as tf
import cv2
import skvideo.io
import skimage.transform
import numpy as np

def save_gen(generated_images, n_ex = 36, epoch = 0, iter = 0):
    for i in range(generated_images.shape[0]):
        cv2.imwrite('/root/code/Video_Generation/gen_images/image_' + str(epoch) + '_' + str(iter) + '_' + str(i) + '.jpg', generated_images[i, :, :, :])

def process_and_write_image(images,name):
    images = np.array(images)
    images = (images + 1)*127.5
    for i in range(images.shape[0]):
        cv2.imwrite("./genvideos/" + name + ".jpg",images[i,0,:,:,:])
def read_and_process_video(files,size,nof):
    videos = np.zeros((size,nof,64,64,3))
    counter = 0
    for file in files:
        vid = skvideo.io.vreader(file)
        curr_frames = []
        i = 0
        for frame in vid:
            ## Considering first 10 frames for now.
            frame = skimage.transform.resize(frame,[64,64])
            #if len(frame.shape)<3:
            #    frame = np.repeat(frame,3).reshape([64,64,3])
            curr_frames.append(frame)
            i = i + 1
        curr_frames = np.array(curr_frames)
        curr_frames = curr_frames*255.0
        curr_frames = curr_frames/127.5 - 1
    #    print "Shape of frames: {0}".format(curr_frames.shape)
        videos[counter,:,:,:,:] = curr_frames
        counter = counter + 1
        #idx = map(int,np.linspace(0,len(curr_frames)-1,32))
        #curr_frames = curr_frames[idx,:,:,:]
        #print "Captured 80 frames: {0}".format(curr_frames.shape)
    return videos

def process_and_write_video(videos,name):
    videos =np.array(videos)
    videos = np.reshape(videos,[-1,32,64,64,3])
    vidwrite = np.zeros((32,64,64,3))
    for i in range(videos.shape[0]):
        vid = videos[i,:,:,:,:]
        vid = (vid + 1)*127.5
        for j in range(vid.shape[0]):
            frame = vid[j,:,:,:]
            #frame = (frame+1)*127.5
            vidwrite[j,:,:,:] = frame
        skvideo.io.vwrite("./genvideos/" +name + ".mp4",vidwrite)

def conv2d(input_, output_dim,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv

def deconv2d(input_, output_shape,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="deconv2d", with_w=False):
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
          deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
          deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
          return deconv, w, biases
        else:
            return deconv

def conv3d(input_, output_dim,k_h=5, k_w=5,k_z =5, d_h=2, d_w=2,d_z=2, stddev=0.02,name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w,k_z, input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w,d_z, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv

def deconv3d(input_, output_shape,k_h=5, k_w=5,k_z=5, d_h=2, d_w=2,d_z=2, stddev=0.02,name="deconv2d", with_w=False):
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w,k_z, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
          deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w,d_z, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
          deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w,d_z, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
          return deconv, w, biases
        else:
            return deconv

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
