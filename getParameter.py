import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import linalg

NumState = 2
NumInput = 2

hidden = 36

q = np.diag((1, 1, 0, 0))
r = np.diag((0.1, 0.1))

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('./modelANN1/ANN.meta', clear_devices=True)
saver.restore(sess, './modelANN1/ANN')

W1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W1')[0]
W2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W2')[0]

W1_arr = W1.eval()
W2_arr = W2.eval()

def ReduceMatrix(x, index_list, axis):
    x = np.delete(x, index_list, axis)
    size = hidden-len(index_list)
    return x, size

def encode(index_list):
    Key = 0
    for x in index_list:
        temp = 1
        Key += temp<<x
    return Key

def main():
    
    DataIndex = 68000

    with open('AutoencodeTrainData\Test.csv','a+', newline='') as f:
        writer = csv.writer(f)
        all_pattern_reader = csv.reader(open('AutoencodeTrainData\AllPattern.csv', encoding='utf-8'))
        for row in all_pattern_reader:
            line = []
            while '' in row:
                row.remove('')
            line = list(map(int, row))
            Key = encode(line)
            W1_r, W1_size = ReduceMatrix(W1_arr, line, 1)
            W2_r, W2_size = ReduceMatrix(W2_arr, line, 0)
            #print(np.shape(W1_r))
            #print(np.shape(W2_r[0:W2_size, :]))
            W_r = np.dot(W1_r, W2_r[0:W2_size, :])# X(t+1) = AX(t) + BU(t) + C
            A = W_r[0:2*NumState, :].T
            B = W_r[2*NumState:2*NumState+NumInput, :].T
            #C = W_r[2*NumState+NumInput:2*NumState+NumInput+1, :]
            P = linalg.solve_continuous_are(A, B, q, r)
            K = np.dot(np.dot(linalg.inv(r), B.T), P)
            K_list = [Key]
            for i in range(2):
              for j in range(4):
                K_list.append(K[i][j])
            writer.writerow(K_list)


    sess.close()

main()
print("Finish!")