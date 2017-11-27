import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import linalg
import random
import types
import csv

meanX = np.array([1.060287521, 1.060287521, -30, -30, 0.044834633, -0.026438883])
meanY = np.array([1.030287521, 1.030287521, -11.29412641, -91.16077679])
stdX = np.array([0.676768727, 0.676768727, 172.3377411, 172.3377411, 155.8123305, 155.5320226])
stdY = np.array([0.698366814, 0.698366814, 170.1032079, 212.5436693])

hidden = 36
NumState = 2
NumInput = 2

#train model data
#q = np.diag((0.01, 0.01, 0.01, 0.01))
#r = np.diag((0.01, 0.01))

#real model data
#q = np.diag((0.00001, 0.00001, 0.00001, 0.00001))
#r = np.diag((1, 1))

q = np.diag((0.1, 0.1, 0.1, 0.1))
r = np.diag((0.001, 0.001))

'''
def Init_K_Dictionary():

    K_dictionary = {}
    all_pattern_reader = csv.reader(open('AutoencodeTrainData\Test.csv', encoding='utf-8'))
    for row in all_pattern_reader:

        line = list(map(float, row))
        K = np.zeros((2, 4))
        K[0] = line[1:5]
        K[1] = line[5:9]
        K_dictionary[int(line[0])] = K
    return K_dictionary
'''
def encode(index_list):
    Key = 0
    for x in index_list:
        temp = 1
        Key += temp<<x
    return Key

def GetIndex(list):
    return [idx for idx, h in enumerate(list) if h==0]

def ReduceMatrix(x, index_list, axis):
    x = np.delete(x, index_list, axis)
    size = hidden-len(index_list)
    return x, size

def TrainDynamic(A, B, X, U, target):
    X = np.dot(A, (X-target))+ np.dot(B, U)+target
    return X

def ArmDynamic(X, U):

    derutaT = 0.001
    nextX = np.zeros(2*NumState)
    m1 = 0.5
    m2 = 0.5
    l1 = 0.15
    l2 = 0.15
    lg1 = (1/2)*l1 
    lg2 = (1/2)*l2 
    g = 9.8
    I1 = (1.0/3)*m1*l1*l1
    I2 = (1.0/12)*m2*l2*l2

    [theta1, theta2, omega1, omega2] = [X[0], X[1], X[2], X[3]]

    u1 = U[0]
    u2 = U[1]

    M = np.zeros((2,2))
    h = np.zeros(2)
    G = np.zeros(2)
    f = np.zeros(2)

    M[0][0] = m1*np.power(lg1, 2)+m2*np.power(l1, 2)+m2*np.power(lg2, 2)+I1+I2+2*m2*l1*lg2*np.cos(theta2)
    M[0][1] = m2*np.power(lg2, 2)+I2+m2*l1*lg2*np.cos(theta2)
    M[1][0] = M[0][1]
    M[1][1] = m2*np.power(lg2, 2)+I2
    _M = np.linalg.inv(M)
    h[0] = -m2*l1*lg2*(2*omega1+omega2)*omega2*np.sin(theta2)
    h[1] = m2*l1*lg2*np.power(omega1, 2)*np.sin(theta2)
    G[0] = (g*l1*m2+g*lg1*m1)*np.cos(theta1)+m2*g*lg2*np.cos(theta1+theta2)
    G[1] = m2*g*lg2*np.cos(theta1+theta2)
    f[0] = _M[0][0]*(-h[0]-G[0]+u1)+_M[0][1]*(-h[1]-G[1]+u2)
    f[1] = _M[1][0]*(-h[0]-G[0]+u1)+_M[1][1]*(-h[1]-G[1]+u2)

    nextX[0] = X[0]+derutaT*omega1
    nextX[1] = X[1]+derutaT*omega2
    nextX[2] = X[2]+derutaT*f[0]
    nextX[3] = X[3]+derutaT*f[1]
    return nextX

def main():

    step = 10
    period = 1000

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('./modelNoBais/ANN.meta', clear_devices=True)
    saver.restore(sess, './modelNoBais/ANN')

    W1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W1')[0]
    W2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W2')[0]

    W1_arr = W1.eval()
    W2_arr = W2.eval()
    '''
    K_dictionary = {}
    K_dictionary = Init_K_Dictionary()
    sort_key = sorted(K_dictionary.keys())
    '''
    K = np.zeros(((step, 2, 4)))
    
    theta1d = 15
    theta2d = 15
    omega1 = 0
    omega2 = 0
    theta1 = 1.0*theta1d*np.pi/180
    theta2 = 1.0*theta2d*np.pi/180
    u1 = random.uniform(-270, 270)
    u2 = random.uniform(-270, 270)
    
    Xd1 = 50
    Xd2 = 80
    targetTheta1 = 1.0*Xd1*np.pi/180
    targetTheta2 = 1.0*Xd2*np.pi/180

    target = (np.array([targetTheta1, targetTheta2, 0, 0])-meanX[0:4])*(1/stdX[0:4])

    CX = np.zeros((2*NumState, (period+1)))
    CU = np.zeros((NumInput, (period)))
    
    CK = np.zeros(((period, 2, 4)))
    CA = np.zeros(((period, 4, 4)))
    CB = np.zeros(((period, 4, 2)))
    
    TK = []
    Tk_index = 0

    U = np.zeros(NumInput)

    [CX[0][0], CX[1][0], CX[2][0], CX[3][0]] = [theta1, theta2, omega1, omega2]
    #oldKey = 0

    for k in range(0, period):
 
        oldKey = 0
        [theta1, theta2, omega1, omega2] = [CX[0][k], CX[1][k], CX[2][k], CX[3][k]]
        for j in range(0, step):

            Input = [[theta1], [theta2], [omega1], [omega2], [u1], [u2]]
            Input_arr = (np.array(Input).reshape(6)-meanX)*(1/stdX)
            h = np.maximum(np.dot(W1_arr.T, Input_arr), 0)
            h_index = GetIndex(h)
            W1_r, W1_size = ReduceMatrix(W1_arr, h_index, 1)
            W2_r, W2_size = ReduceMatrix(W2_arr, h_index, 0)
            search_key = encode(h_index)
            key = search_key
            '''
            if search_key in K_dictionary.keys():
                key = search_key
            else:
            	for i in range(len(sort_key)):
            		if search_key < sort_key[i]:
            			key = np.maximum(sort_key[i]-1, sort_key[i])
            '''
            W_r = np.dot(W1_r, W2_r[0:W2_size, :])# X(t+1) = AX(t) + BU(t) + C
            A = W_r[0:2*NumState, :].T
            B = W_r[2*NumState:2*NumState+NumInput, :].T
     
            #C = W_r[2*NumState+NumInput:2*NumState+NumInput+1, :].reshape(4)

            P = linalg.solve_discrete_are(A, B, q, r)
            K_now = np.dot(linalg.inv(np.dot(np.dot(B.T, P), B)+r), (np.dot(np.dot(B.T, P), A)))
            g = np.dot(linalg.inv(np.dot(np.dot(B.T, P), B)+r), (np.dot(np.dot(B.T, q), target)))
            #K_now = np.dot(np.dot(linalg.inv(r), B.T), P)
            #K_now = np.dot(np.linalg.pinv(r), np.dot(B.T, P))
            eigVals, eigVecs = linalg.eig(A-np.dot(B, K_now))
            #print(eigVals)
            #print(j, K_now)
            if oldKey == key:
                CK[k] = K_now
                CA[k] = A
                CB[k] = B
                print("oldKey == key in", j)
                break
            else:
                
                if j == step-1:
                    K[j] = K_now#K_dictionary[key]
                    K_mean = np.mean(K, 0)
                    u = -np.dot(K_mean, ((Input_arr[0:4]-target)))
                    #u = -np.dot(K_mean, (Input_arr[0:4]))
                    U = u*stdX[4:6]+meanX[4:6]
                    #print(j, U)
                    #return
                    u1 = U[0]
                    u2 = U[1]

                    CK[k] = 1#K_mean
                    CA[k] = A
                    CB[k] = B
                    oldKey = 0
                    print("oldKey != key")
                    break
                else:
                    oldKey = key
                    K[j] = K_now#K_dictionary[key]
                    u = -np.dot(K_now, ((Input_arr[0:4]-target)))
                    #u = -np.dot(K_now, (Input_arr[0:4]))
                    U = u*stdX[4:6]+meanX[4:6]
                    #print(j, U)
                    #return
                    u1 = U[0]
                    u2 = U[1]
        
        #CX[:, k+1] = TrainDynamic(A, B, ((CX[:, k]-meanX[0:4])*(1/stdX[0:4])), u, target)*stdX[0:4]+meanX[0:4]
        if ((CA[k] == CA[k-1]).all() or (CB[k] == CB[k-1]).all()):
            pass
        else:
            Tk_index += 1
            TK.append(k)

        CX[:, k+1] = ArmDynamic(CX[:, k], U)
        CU[0][k] = U[0]
        CU[1][k] = U[1]
    
    print("Final!")
    data = []
    for k in range(0, period):
        theta1 = CX[0][k]
        theta2 = CX[1][k]
        u1 = CU[0][k]
        u2 = CU[1][k]
        data.append([180*theta1/np.pi, 180*theta2/np.pi, u1, u2])

    plt.figure(1)
    Ax1 = plt.subplot(411)
    Ax2 = plt.subplot(412)
    Ax3 = plt.subplot(413)
    Ax4 = plt.subplot(414)
    plt.figure(2)
    x = np.linspace(0, period, period)
    data_arr = np.array(data)
    plt.sca(Ax1)
    plt.title('u1')
    plt.plot(x, data_arr[:, 2])
    plt.sca(Ax2)
    plt.title('u2')
    plt.plot(x, data_arr[:, 3])
    plt.figure(2)
    plt.title('Theta')
    plt.ylabel("Theta(rad)")
    plt.xlabel("T(ms)")
    plt.plot(x, data_arr[:, 0], label = "Theta1", color = "red")
    plt.plot(x, data_arr[:, 1], label = "Theta2", color = "blue")
    for n in range(Tk_index):
        plt.plot(TK[n], data_arr[TK[n]][0], 'rx')
        plt.plot(TK[n], data_arr[TK[n]][1], 'bo')

    plt.plot(x, x-(x-Xd1), label = "target1", color = "green")
    plt.plot(x, x-(x-Xd2), label = "target2", color = "orange")

    plt.sca(Ax3)
    plt.title('A')
    plt.plot(x, CA[:, 0, 0])
    plt.plot(x, CA[:, 0, 1])
    plt.plot(x, CA[:, 0, 2])
    plt.plot(x, CA[:, 0, 3])
    plt.plot(x, CA[:, 1, 0])
    plt.plot(x, CA[:, 1, 1])
    plt.plot(x, CA[:, 1, 2])
    plt.plot(x, CA[:, 1, 3])
    plt.plot(x, CA[:, 2, 0])
    plt.plot(x, CA[:, 2, 1])
    plt.plot(x, CA[:, 2, 2])
    plt.plot(x, CA[:, 2, 3])
    plt.plot(x, CA[:, 3, 0])
    plt.plot(x, CA[:, 3, 1])
    plt.plot(x, CA[:, 3, 2])
    plt.plot(x, CA[:, 3, 3])
    
    plt.sca(Ax4)
    plt.title('B')
    plt.plot(x, CB[:, 0, 0])
    plt.plot(x, CB[:, 0, 1])
    plt.plot(x, CB[:, 1, 0])
    plt.plot(x, CB[:, 1, 1])
    plt.plot(x, CB[:, 2, 0])
    plt.plot(x, CB[:, 2, 1])
    plt.plot(x, CB[:, 3, 0])
    plt.plot(x, CB[:, 3, 1])

    plt.legend()
    plt.show()
    print("Show Result!")

    return

def NobugPlease():
	print("                    "," _oo0oo_")
	print("                    ","o8888888o")
	print("                    ","88\" . \"88")
	print("                    ","(| -_- |)")
	print("                    ","o\\  =  /o")
	print("                ","_____/'---'\\_____")
	print("              ",".'   \\\\|     |//   '.")
	print("             ","/   \\\\|||  :  |||//   \\")
	print("            ","/   _||||| -:- |||||_   \\")
	print("            ","|    | \\\\\\  -  /// |    |")
	print("            ","|  \\_|  ''\\---/''  |_/  |")
	print("            ","\\   .-\\__  '-'  __/-.   /")
	print("         ","____'.  .'  /--.--\\  '.  .'____")
	print("      ",".\"\"  '<  '.____\\_<|>_/____.'  >'  \"\".")
	print("     ","| |  :   '- \\'.;'\\ _ /';.'/ -'   :  | |")
	print("     ","\\  \\ '-.    \\_ ___\\ /___ _/    .-' /  /")
	print("","======'-.____'-.____\\_______/____.-'____.-'======")
	print("","                    '======='")
	print("   ")
	print("   ")
	print("No bug!")
	print("Research go well!")
	print("   ")
	print("   ")

NobugPlease()
main()
