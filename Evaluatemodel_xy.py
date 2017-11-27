import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy as sp
'''
meanX = tf.constant([-0.012983715, 0.176911511, 7.389664058, 2.231791453, 0.089528897, -0.08092153])
meanY = tf.constant([-0.004594611, 0.175259144, 10.67893221, -2.192553039])
stdX = tf.constant([0.148645532, 0.077239627, 38.28027042, 32.22820913, 155.8045396, 155.8624478])
stdY = tf.constant([0.151827128, 0.08044371,  39.73674448, 34.49124595])
'''
'''
meanX = tf.constant([-0.012983715, 0.176911511, 3.694832029, 1.115895726, -0.121276589,  -0.362678419])
meanY = tf.constant([-0.009013613, 0.177060652, 4.577344593, 0.073810308])
stdX = tf.constant([0.148645532, 0.077239627, 19.14013521, 16.11410456, 155.8245209, 155.9038717])
stdY = tf.constant([0.149931339, 0.077736906, 20.17017772, 17.38957652])
'''
meanX = tf.constant([0.009187305,  0.181890982, 3.913660832, 0.588540817, 0.169099644, 0.117121659])
meanY = tf.constant([0.013237291,  0.181458486, 4.588948668, -0.628719787])
stdX = tf.constant([0.153151136, 0.0752278, 19.47154709, 16.36464396, 155.8866265, 155.7958725])
stdY = tf.constant([0.154047981, 0.076453883, 20.5097364,  17.64558234])

NumState = 2
NumInput = 2

trainX = tf.placeholder("float", [None, 2*NumState+NumInput], name = 'trainX')
trainY = tf.placeholder("float", [None, 2*NumState], name = 'trainY')
trainOne = tf.placeholder("float", [None, 1], name = 'trainOne')
hidden = 36

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('./modelNoBais_xy3/ANN_xy.meta', clear_devices=True)
saver.restore(sess, './modelNoBais_xy3/ANN_xy')

W1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W1')[0]
W2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W2')[0]

W1_arr = W1.eval()
W2_arr = W2.eval()

def read_my_file_format(filename_queue):
  reader = tf.TextLineReader()
  key, record_string = reader.read(filename_queue)
  record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
  col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(record_string, record_defaults=record_defaults)
  features = tf.pack(([col2, col3, col4, col5, col6, col7]-meanX)/stdX)
  Label = tf.pack(([col8, col9, col10, col11]-meanY)/stdY)
  return features, Label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=False)
  example, label = read_my_file_format(filename_queue)
  min_after_dequeue = 8
  capacity = min_after_dequeue + batch_size
  example_batch, label_batch = tf.train.batch(
      [example, label], batch_size=batch_size, capacity=capacity)
  return example_batch, label_batch

def GetIndex(list):
    return [idx for idx, h in enumerate(list) if h==0]

def ReduceMatrix(x, index_list, axis):
    x = np.delete(x, index_list, axis)
    size = hidden-len(index_list)
    return x, size

def main():
    
    DataIndex = 10000
    h = tf.nn.relu(tf.matmul(trainX, W1), name="h")
    dynamic_next = tf.matmul(h, W2, name = 'y')

    trainFile = ["AutoencodeTrainData\TrainAll_one_step_xy3.csv"]
    Example_batch,Label_batch = input_pipeline(trainFile, DataIndex)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    feature_batch, label_batch = sess.run([Example_batch, Label_batch])

    next_x = sess.run(dynamic_next, feed_dict={trainX:feature_batch, trainY:label_batch})
    hiden_pattern = sess.run(h, feed_dict={trainX:feature_batch, trainY:label_batch})
    
    coord.request_stop()
    coord.join(threads)
    sess.close()

    plt.figure(1)
    Ax1 = plt.subplot(411)
    Ax2 = plt.subplot(412)
    Ax3 = plt.subplot(413)
    Ax4 = plt.subplot(414)
    x = np.linspace(0, DataIndex, DataIndex)
    sample_arr = np.array(next_x)
    label_arr = np.array(label_batch)
    
    plt.sca(Ax1)
    plt.title('X')
    plt.plot(x, label_arr[:, 0], color = "red")
    plt.plot(x, sample_arr[:, 0], color = "blue")
    plt.sca(Ax2)
    plt.title('Y')
    plt.plot(x, label_arr[:, 1], color = "red")
    plt.plot(x, sample_arr[:, 1], color = "blue")
    plt.sca(Ax3)
    plt.title('dotX')
    plt.plot(x, label_arr[:, 2], color = "red")
    plt.plot(x, sample_arr[:, 2], color = "blue")
    plt.sca(Ax4)
    plt.title('dotY')
    plt.plot(x, label_arr[:, 3], color = "red")
    plt.plot(x, sample_arr[:, 3], color = "blue")
    plt.legend()
    plt.show()

main()
print("Finish!")