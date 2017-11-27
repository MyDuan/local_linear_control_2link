import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import linalg
import random
import types
import csv
'''
meanX = np.array([-0.012983715, 0.176911511, 7.389664058, 2.231791453, 0.089528897, -0.08092153])
meanY = np.array([-0.004594611, 0.175259144, 10.67893221, -2.192553039])
stdX = np.array([0.148645532, 0.077239627, 38.28027042, 32.22820913, 155.8045396, 155.8624478])
stdY = np.array([0.151827128, 0.08044371,  39.73674448, 34.49124595])
'''
'''
meanX = np.array([-0.012983715, 0.176911511, 3.694832029, 1.115895726, -0.121276589,  -0.362678419])
meanY = np.array([-0.009013613, 0.177060652, 4.577344593, 0.073810308])
stdX = np.array([0.148645532, 0.077239627, 19.14013521, 16.11410456, 155.8245209, 155.9038717])
stdY = np.array([0.149931339, 0.077736906, 20.17017772, 17.38957652])
'''
meanX = np.array([0.009187305,  0.181890982, 3.913660832, 0.588540817, 0.169099644, 0.117121659])
meanY = np.array([0.013237291,  0.181458486, 4.588948668, -0.628719787])
stdX = np.array([0.153151136, 0.0752278, 19.47154709, 16.36464396, 155.8866265, 155.7958725])
stdY = np.array([0.154047981, 0.076453883, 20.5097364,  17.64558234])

hidden = 36
NumState = 2
NumInput = 2
Tr = np.zeros((2*NumState, (1000)))
m1 = 0.5
m2 = 0.5
l1 = 0.15
l2 = 0.15
lg1 = (1/2)*l1 
lg2 = (1/2)*l2
g = 9.8
I1 = (1.0/3)*m1*l1*l1
I2 = (1.0/12)*m2*l2*l2
#train model data
#q = np.diag((1, 1, 1, 1))
#r = np.diag((1, 1))

#real model data no g
q = np.diag((0.01, 0.01, 0.01, 0.01))
r = np.diag((0.001, 0.001))
#q = np.diag((0.01, 0.01, 0.01, 0.01))
#r = np.diag((0.001, 0.001))


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

    nextX[0] = theta1+derutaT*omega1
    nextX[1] = theta2+derutaT*omega2
    nextX[2] = omega1+derutaT*f[0]
    nextX[3] = omega2+derutaT*f[1]
    return nextX

def Trajectory(k):

    trac = np.zeros(2)
    docTrac = np.zeros(2)
    traX = np.zeros(2)
    docTraX = np.zeros(2)
    derutaT = 0.001

    trac[0] = np.pi/6+5*np.pi/3*np.power((k*derutaT), 3)-5*np.pi/2*np.power((k*derutaT), 4)+np.pi*np.power((k*derutaT), 5)
    trac[1] = np.pi/6+5*np.pi/6*np.power((k*derutaT), 3)-5*np.pi/4*np.power((k*derutaT), 4)+np.pi/2*np.power((k*derutaT), 5)
    docTrac[0] = 5*np.pi*np.power((k*derutaT), 2)-10*np.pi*np.power((k*derutaT), 3)+5*np.pi*np.power((k*derutaT), 4)
    docTrac[1] = 5*np.pi/2*np.power((k*derutaT), 2)-5*np.pi*np.power((k*derutaT), 3)+5*np.pi/2*np.power((k*derutaT), 4)
    
    traX[0] = l1*np.cos(trac[0])+l2*np.cos(trac[0]+trac[1])
    traX[1] = l1*np.sin(trac[0])+l2*np.sin(trac[0]+trac[1])
    docTraX[0] = -l1*np.sin(trac[0])*(docTrac[0])-l2*np.sin(trac[0]+trac[1])*(docTrac[0]+docTrac[1])
    docTraX[1] = l1*np.cos(trac[0])*(docTrac[0])+l2*np.cos(trac[0]+trac[1])*(docTrac[0]+docTrac[1])
    target = (np.array([traX[0], traX[1], docTraX[0], docTraX[1]])-meanX[0:4])*(1/stdX[0:4])
    Tr[0][k] = traX[0]
    Tr[1][k] = traX[1]
    return target

def main():

    step = 10
    period = 1000

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('./modelNoBais_xy3/ANN_xy.meta', clear_devices=True)
    saver.restore(sess, './modelNoBais_xy3/ANN_xy')
    W1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W1')[0]
    W2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'W2')[0]

    W1_arr = W1.eval()
    W2_arr = W2.eval()

    K = np.zeros(((step, 2, 4)))
    
    theta1d = 30
    theta2d = 30
    omega1 = 0
    omega2 = 0
    theta1 = 1.0*theta1d*np.pi/180
    theta2 = 1.0*theta2d*np.pi/180
    u1 = 0#random.uniform(-50, 50)
    u2 = 0#random.uniform(-50, 50)
    Xd1 = 45
    Xd2 = 45
    targetTheta1 = 1.0*Xd1*np.pi/180
    targetTheta2 = 1.0*Xd2*np.pi/180
    P_xd = l1*np.cos(targetTheta1) + l2*np.cos(targetTheta1+targetTheta2)
    P_yd = l1*np.sin(targetTheta1) + l2*np.sin(targetTheta1+targetTheta2)
    #target = (np.array([P_xd, P_yd, 0, 0])-meanX[0:4])*(1/stdX[0:4])

    CX = np.zeros((2*NumState, (period+1)))
    PX = np.zeros((2*NumState, (period+1)))
    CU = np.zeros((NumInput, (period)))

    CK = np.zeros(((period, 2, 4)))
    CA = np.zeros(((period, 4, 4)))
    CB = np.zeros(((period, 4, 2)))

    TK = []
    Tk_index = 0
    
    U = np.zeros(NumInput)
    [CX[0][0], CX[1][0], CX[2][0], CX[3][0]] = [theta1, theta2, omega1, omega2]
    P_x = l1*np.cos(CX[0][0]) + l2*np.cos(CX[0][0]+CX[1][0])
    P_y = l1*np.sin(CX[0][0]) + l2*np.sin(CX[0][0]+CX[1][0])
    D_x = -l1*np.sin(CX[0][0])*CX[2][0]-l2*np.sin(CX[0][0]+CX[1][0])*(CX[2][0]+CX[3][0])
    D_y = l1*np.cos(CX[0][0])*CX[2][0]+l2*np.cos(CX[0][0]+CX[1][0])*(CX[2][0]+CX[3][0])
    PX[:, 0] = [P_x, P_y, D_x, D_y]
    Tra = False
    #oldKey = 0
    for k in range(0, period):
        if(Tra):
            target = Trajectory(k)
        else:
            target = (np.array([P_xd, P_yd, 0, 0])-meanX[0:4])*(1/stdX[0:4])

        oldKey = 0
        
        P_x = l1*np.cos(CX[0][k]) + l2*np.cos(CX[0][k]+CX[1][k])
        P_y = l1*np.sin(CX[0][k]) + l2*np.sin(CX[0][k]+CX[1][k])
        D_x = -l1*np.sin(CX[0][k])*CX[2][k]-l2*np.sin(CX[0][k]+CX[1][k])*(CX[2][k]+CX[3][k])
        D_y = l1*np.cos(CX[0][k])*CX[2][k]+l2*np.cos(CX[0][k]+CX[1][k])*(CX[2][k]+CX[3][k])
        #[P_x, P_y, D_x, D_y] = [PX[0][k], PX[1][k], PX[2][k], PX[3][k]]
        for j in range(0, step):

            Input = [[P_x], [P_y], [D_x], [D_y], [u1], [u2]]
            Input_arr = (np.array(Input).reshape(6)-meanX)*(1/stdX)
            h = np.maximum(np.dot(W1_arr.T, Input_arr), 0)
            h_index = GetIndex(h)
            W1_r, W1_size = ReduceMatrix(W1_arr, h_index, 1)
            W2_r, W2_size = ReduceMatrix(W2_arr, h_index, 0)
            search_key = encode(h_index)
            key = search_key

            W_r = np.dot(W1_r, W2_r[0:W2_size, :])# X(t+1) = AX(t) + BU(t) + C
            A = W_r[0:2*NumState, :].T

            B = W_r[2*NumState:2*NumState+NumInput, :].T
    
            P = linalg.solve_discrete_are(A, B, q, r)
            K_now = np.dot(linalg.inv(np.dot(np.dot(B.T, P), B)+r), (np.dot(np.dot(B.T, P), A)))
            #g = np.dot(linalg.inv(np.dot(np.dot(B.T, P), B)+r), (np.dot(np.dot(B.T, q), target)))

            eigVals, eigVecs = linalg.eig(A-np.dot(B, K_now))
            #print(eigVals)
            if oldKey == key:
                CK[k] = K_now
                CA[k] = A
                CB[k] = B
                print("oldKey == key", j)
                break
            else:
                
                if j == step-1:
                    K[j] = K_now#K_dictionary[key]
                    K_mean = np.mean(K, 0)
                    u = -np.dot(K_mean, (Input_arr[0:4]-target))#+g
                    U = u*stdX[4:6]+meanX[4:6]
                    u1 = U[0]
                    u2 = U[1]
                    CK[k] = K_mean
                    CA[k] = A
                    CB[k] = B
                    print("oldKey != key")
                    break
                else:
                    oldKey = key
                    K[j] = K_now
                    u = -np.dot(K_now, (Input_arr[0:4]-target))#+g
                    U = u*stdX[4:6]+meanX[4:6]
                    u1 = U[0]
                    u2 = U[1]
        
        #PX[:, k+1] = TrainDynamic(A, B, ((PX[:, k]-meanX[0:4])*(1/stdX[0:4])), u, target)*stdX[0:4]+meanX[0:4]
        #print(U)
        if ((CA[k] == CA[k-1]).all() and (CB[k] == CB[k-1]).all()):
            pass
        else:
            Tk_index += 1
            TK.append(k)
        CX[:, k+1] = ArmDynamic(CX[:, k], U)
        CU[0][k] = U[0]
        CU[1][k] = U[1]
    
    print("Final!")
    print(TK)
    data = []
    for k in range(0, period):
        #x = PX[0][k]
        #y = PX[1][k]
        x = l1*np.cos(CX[0][k]) + l2*np.cos(CX[0][k]+CX[1][k])
        y = l1*np.sin(CX[0][k]) + l2*np.sin(CX[0][k]+CX[1][k])
        u1 = CU[0][k]
        u2 = CU[1][k]
        data.append([x, y, u1, u2])

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

    plt.figure(2)
    plt.title('x')
    plt.ylabel("x(m)")
    plt.xlabel("T(ms)")
    plt.plot(x, data_arr[:, 0], label = "x", color = "red")
    plt.plot(x, data_arr[:, 1], label = "y", color = "blue")
    for n in range(Tk_index):
        plt.plot(TK[n], data_arr[TK[n]][0], 'rx')
        plt.plot(TK[n], data_arr[TK[n]][1], 'bo')
    
    if(Tra):
        plt.plot(x, Tr[0, :], label = "targetX", color = "green")
        plt.plot(x, Tr[1, :], label = "targetY", color = "orange")
    else:
        plt.plot(x, x-(x-P_xd), label = "targetX", color = "green")
        plt.plot(x, x-(x-P_yd), label = "targetY", color = "orange")
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
