#
# utils.py
# ==============================================================================
"""Utils"""
import numpy as np

from config import input_feature_len, num_classes


def load_data():
  """Generate very f***in simple gcn training data N=5"""

  # num_nodes = 5

  # N x N
  # bitcoin, seoul, crypto, current, money
  adj_mat = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 0, 1, 0],
                      [0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0]])

  # Normalized laplacian matrix
  # lapl_mat = adj_mat + np.eye(adj_mat.shape[0])
  # lapl_mat = np.eye(adj_mat.shape[0])
  lapl_mat = adj_mat

  # N x (num_features)
  # num_features = 3 # (search frequency, time remaining, average search hours)
  # average search hours => (24 h = 1). when do users search such words?
  features = np.array([[0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5]])
  # features = np.array([[0.8, 0.4, 0.9],
  #                      [0.1, 0.1, 0.25],
  #                      [0.8, 0.5, 0.84],
  #                      [0.85, 0.3, 0.3],
  #                      [0.92, 0.5, 0.1]])

  labels = np.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [1, 0]])
  # labels = np.array([[1, 0],
  #                    [0, 1],
  #                    [1, 0],
  #                    [1, 0],
  #                    [0, 1]])

  return lapl_mat, features, labels

def load_data_2():
  """Generate very f***in simple gcn training data N=4"""

  # num_nodes = 5

  # N x N
  # bitcoin, seoul, crypto, current, money
  adj_mat = np.array([[0, 0, 1, 0],
                      [0, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 1, 0]])

  # Normalized laplacian matrix
  # lapl_mat = adj_mat + np.eye(adj_mat.shape[0])
  # lapl_mat = np.eye(adj_mat.shape[0])
  lapl_mat = adj_mat

  # N x (num_features)
  # num_features = 3 # (search frequency, time remaining, average search hours)
  # average search hours => (24 h = 1). when do users search such words?
  features = np.array([[0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5]])
  # features = np.array([[0.8, 0.4, 0.9],
  #                      [0.1, 0.1, 0.25],
  #                      [0.8, 0.5, 0.84],
  #                      [0.85, 0.3, 0.3]])

  labels = np.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1]])
  # labels = np.array([[1, 0],
  #                    [0, 1],
  #                    [1, 0],
  #                    [1, 0]])

  return lapl_mat, features, labels

def load_data_3():
  """Generate very f***in simple gcn training data N=1000"""

  num_nodes = 1000

  # N x N
  adj_mat = np.ones((num_nodes, num_nodes))

  # Normalized laplacian matrix
  lapl_mat = adj_mat + np.eye(adj_mat.shape[0])
  # lapl_mat = np.eye(adj_mat.shape[0])
  # lapl_mat = adj_mat

  features = np.ones((num_nodes, input_feature_len))

  labels = np.ones((num_nodes, num_classes))

  return lapl_mat, features, labels

def print_acc_loss(sess, ops, feed_dict, epoch):
  acc, loss = sess.run(ops, feed_dict=feed_dict)
  print("Epoch: {}  Accuracy: {:.3}  Loss: {:.4}".format(epoch, acc, loss))

def run_train(sess, ops, feed_dict):
  sess.run(ops, feed_dict=feed_dict)
