#
# model.py
#
# ==============================================================================
"""Simple GCN model with tf dense tensor"""
import os

import numpy as np
import tensorflow as tf

from config import lr, input_feature_len, num_classes

gpu_num = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num


def uniform(shape, half_interval=0.05, name=None):
  """Return a tf.Variable with truncated uniform distribution."""
  init_value = tf.random_uniform(shape,
                                 minval=-half_interval,
                                 maxval=half_interval,
                                 dtype=tf.float32)
  return tf.Variable(init_value, name=name)

def glorot(shape, name=None):
  half_interval = np.sqrt(6.0/(shape[0]+shape[1]))
  return uniform(shape, half_interval, name=name)

def graph_conv_layer(lapl_mat, input_x, feature_shape, name, is_last=False):
  """Graph conv layer."""

  with tf.variable_scope(name+"_vars"):
    weight = glorot(feature_shape, name="weight")
    bias = uniform([feature_shape[-1]], name="bias")

    out = tf.matmul(lapl_mat,
                    tf.add(tf.matmul(input_x, weight), bias),
                    name="out")

    if is_last:
      act = out
      return act
    act = tf.nn.relu(out, name="act")
    return act

def build_model():
  """Build simple gcn network"""
  plhdr_input = tf.placeholder(tf.float32, shape=(None, 3), name='input')
  plhdr_label = tf.placeholder(tf.float32, shape=(None, 2), name='label')
  plhdr_lapl = tf.placeholder(tf.float32, shape=(None, None), name='adj')

  graph_layer_1 = graph_conv_layer(plhdr_lapl,
                                   plhdr_input,
                                   [3, 2],
                                   name="gc1")
  graph_layer_2 = graph_conv_layer(plhdr_lapl,
                                   graph_layer_1,
                                   [2, 3],
                                   name="gc2")
  graph_layer_3 = graph_conv_layer(plhdr_lapl,
                                   graph_layer_2,
                                   [3, 2],
                                   name="gc3",
                                   is_last=True)

  # Cross entropy with softmax as loss function
  with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=graph_layer_3,
                                                                           labels=plhdr_label),
                                   name="xent")
  # AdamOptimizer for train network
  with tf.name_scope("train"):
    train = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

  # Accuracy
  with tf.name_scope("accuracy"):
    prediction = tf.argmax(graph_layer_3, 1)
    correction = tf.argmax(plhdr_label, 1)
    # Returns the truth value (Integer) of (x == y) element-wise;
    correct_prediction = tf.cast(tf.equal(prediction, correction), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

  return accuracy, cross_entropy, train

def build_model_3():
  """Build simple gcn network"""
  plhdr_input = tf.placeholder(tf.float32,
                               shape=(None, input_feature_len),
                               name='input')
  plhdr_label = tf.placeholder(tf.float32,
                               shape=(None, num_classes),
                               name='label')
  plhdr_lapl = tf.placeholder(tf.float32,
                              shape=(None, None),
                              name='adj')

  graph_layer_1 = graph_conv_layer(plhdr_lapl,
                                   plhdr_input,
                                   [input_feature_len, 30],
                                   name="gc1")
  graph_layer_2 = graph_conv_layer(plhdr_lapl,
                                   graph_layer_1,
                                   [30, 30],
                                   name="gc2")
  graph_layer_3 = graph_conv_layer(plhdr_lapl,
                                   graph_layer_2,
                                   [30, num_classes],
                                   name="gc3",
                                   is_last=True)

  # Cross entropy with softmax as loss function
  with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=graph_layer_3,
                                                                           labels=plhdr_label),
                                   name="xent")
  # AdamOptimizer for train network
  with tf.name_scope("train"):
    train = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

  # Accuracy
  with tf.name_scope("accuracy"):
    prediction = tf.argmax(graph_layer_3, 1)
    correction = tf.argmax(plhdr_label, 1)
    # Returns the truth value (Integer) of (x == y) element-wise;
    correct_prediction = tf.cast(tf.equal(prediction, correction), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

  return accuracy, cross_entropy, train
