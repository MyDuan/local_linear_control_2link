import tensorflow as tf
import numpy as np

meanX = tf.constant([1.060287521, 1.060287521, -30, -30, 0.044834633, -0.026438883])
meanY = tf.constant([1.030287521, 1.030287521, -11.29412641, -91.16077679])
stdX = tf.constant([0.676768727, 0.676768727, 172.3377411, 172.3377411, 155.8123305, 155.5320226])
stdY = tf.constant([0.698366814, 0.698366814, 170.1032079, 212.5436693])

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
train_step = tf.train.AdamOptimizer(0.001).minimize(loss, name = 'train_step')

trainFile = ["AutoencodeTrainData\TrainAll_one_step.csv"]

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
saver.save(sess, './modelNoBais/ANN')
coord.request_stop()
coord.join(threads)
sess.close()