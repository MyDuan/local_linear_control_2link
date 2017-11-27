import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy as sp

meanX = tf.constant([1.060287521, 1.060287521, -30, -30, 0.044834633, -0.026438883, 0])
meanY = tf.constant([1.030287521, 1.030287521, -11.29412641, -91.16077679])
stdX = tf.constant([0.676768727, 0.676768727, 172.3377411, 172.3377411, 155.8123305, 155.5320226, 1])
stdY = tf.constant([0.698366814, 0.698366814, 170.1032079, 212.5436693])

NumState = 2
NumInput = 2

trainX = tf.placeholder("float", [None, 2*NumState+NumInput+1], name = 'trainX')
trainY = tf.placeholder("float", [None, 2*NumState], name = 'trainY')
trainOne = tf.placeholder("float", [None, 1], name = 'trainOne')
hidden = 36

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('./modelANNNoBais/ANN.meta', clear_devices=True)
saver.restore(sess, './modelANNNoBais/ANN')

W1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W1')[0]
W2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W2')[0]

W1_arr = W1.eval()
W2_arr = W2.eval()

def read_my_file_format(filename_queue):
  reader = tf.TextLineReader()
  key, record_string = reader.read(filename_queue)
  record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
  col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(record_string, record_defaults=record_defaults)
  features = tf.pack(([col2, col3, col4, col5, col6, col7, col1]-meanX)/stdX)
  Label = tf.pack(([col8, col9, col10, col11]-meanY)/stdY)
  One = tf.pack([col1])
  return features, Label, One

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=False)
  example, label, one = read_my_file_format(filename_queue)
  min_after_dequeue = 8
  capacity = min_after_dequeue + batch_size
  example_batch, label_batch, one_batch = tf.train.batch(
      [example, label, one], batch_size=batch_size, capacity=capacity)
  return example_batch, label_batch, one_batch

def GetIndex(list):
    return [idx for idx, h in enumerate(list) if h==0]

def ReduceMatrix(x, index_list, axis):
    x = np.delete(x, index_list, axis)
    size = hidden-len(index_list)
    return x, size

def main():
    
    DataIndex = 10000
    h = tf.nn.relu(tf.matmul(trainX, W1), name="h")
    H = tf.concat(1, [h, trainOne])
    dynamic_next = tf.matmul(H, W2, name = 'y')

    trainFile = ["AutoencodeTrainData\TrainAll_one_step.csv"]
    Example_batch,Label_batch,One_batch = input_pipeline(trainFile, DataIndex)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    feature_batch, label_batch, one_batch = sess.run([Example_batch, Label_batch, One_batch])

    next_x = sess.run(dynamic_next, feed_dict={trainX:feature_batch, trainY:label_batch, trainOne:one_batch})
    hiden_pattern = sess.run(h, feed_dict={trainX:feature_batch, trainY:label_batch, trainOne:one_batch})
    #print(hiden_pattern)

    #with open('AutoencodeTrainData\AllPattern.csv','a+', newline='') as f:
    #    writer = csv.writer(f)
    #    for i in range(DataIndex):
    #        line = []
    #        line = GetIndex(list(hiden_pattern[i,:]))
            #W1_r, W1_size = ReduceMatrix(W1_arr, line, 1)
            #W2_r, W2_size = ReduceMatrix(W2_arr, line, 0)
            #print(np.shape(W1_r))
            #print(np.shape(W2_r[0:W2_size, :]))
            #W_r = np.dot(W1_r, W2_r[0:W2_size, :])# X(t+1) = AX(t) + BU(t) + C
            #A = W_r[0:2*NumState, :]
            #B = W_r[2*NumState:2*NumState+NumInput, :]
            #C = W_r[2*NumState+NumInput:2*NumState+NumInput+1, :]
            #print(C)

            #return
            #line.append(np.count_nonzero(hiden_pattern[i,:]))
    #        writer.writerow(line)

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
    #data_arr = np.array(180*(next_x-next_xde)/3.14)
    
    plt.sca(Ax1)
    plt.title('theta1')
    plt.plot(x, label_arr[:, 0], color = "red")
    plt.plot(x, sample_arr[:, 0], color = "blue")
    plt.sca(Ax2)
    plt.title('theta2')
    plt.plot(x, label_arr[:, 1], color = "red")
    plt.plot(x, sample_arr[:, 1], color = "blue")
    plt.sca(Ax3)
    plt.title('omega1')
    plt.plot(x, label_arr[:, 2], color = "red")
    plt.plot(x, sample_arr[:, 2], color = "blue")
    plt.sca(Ax4)
    plt.title('omega2')
    plt.plot(x, label_arr[:, 3], color = "red")
    plt.plot(x, sample_arr[:, 3], color = "blue")
    plt.legend()
    plt.show()

main()
print("Finish!")