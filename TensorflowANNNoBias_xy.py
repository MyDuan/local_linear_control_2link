import tensorflow as tf
import numpy as np
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

def read_my_file_format(filename_queue):
  reader = tf.TextLineReader()
  key, record_string = reader.read(filename_queue)
  record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
  col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(record_string, record_defaults=record_defaults)
  features = tf.pack(([col2, col3, col4, col5, col6, col7]-meanX)/stdX)
  Label = tf.pack(([col8, col9, col10, col11]-meanY)/stdY)
  One = tf.pack([col1])
  return features, Label, One

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label, one = read_my_file_format(filename_queue)
  min_after_dequeue = 50000
  capacity = min_after_dequeue + batch_size
  example_batch, label_batch, one_batch = tf.train.shuffle_batch(
      [example, label, one], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch, one_batch

trainX = tf.placeholder("float", [None, 6], name = 'trainX')
trainY = tf.placeholder("float", [None, 4], name = 'trainY')
trainOne = tf.placeholder("float", [None, 1], name = 'trainOne')
hidden = 36

W1 = tf.Variable(tf.truncated_normal([6, hidden], stddev=0.1), name = 'W1')
h = tf.nn.relu(tf.matmul(trainX, W1), name="h")
H = tf.concat(1, [h, trainOne])
W2 = tf.Variable(tf.truncated_normal([hidden, 4], stddev=0.1), name = 'W2')
y = tf.matmul(h, W2, name = 'y')
loss = tf.reduce_mean(tf.square(trainY - tf.cast(y, tf.float32)), name = 'loss')
#global_step = tf.Variable(0)
#learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.96, staircase=True)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, name = 'train_step')
train_step = tf.train.AdamOptimizer(0.002).minimize(loss, name = 'train_step')

trainFile = ["AutoencodeTrainData\TrainAll_one_step_xy3.csv"]

#trainFile = ["allStateSpaceData\Traintest.csv"]
Example_batch,Label_batch,One_batch = input_pipeline(trainFile, 500000)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
for j in range(0, 2):
    feature_batch, label_batch, one_batch = sess.run([Example_batch, Label_batch, One_batch])
    print(feature_batch)
    for i in range(1200):
        train_step.run({trainX:feature_batch, trainY:label_batch, trainOne:one_batch})
        print(j, i, loss.eval({trainX:feature_batch, trainY:label_batch, trainOne:one_batch}))

saver = tf.train.Saver()
saver.save(sess, './modelNoBais_xy3/ANN_xy')
coord.request_stop()
coord.join(threads)
sess.close()