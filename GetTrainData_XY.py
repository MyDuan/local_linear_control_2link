import numpy as np
import random
import csv
import h5py

m1 = 0.5
m2 = 0.5
l1 = 0.15
l2 = 0.15
lg1 = (1/2)*l1 
lg2 = (1/2)*l2
g = 9.8
I1 = (1.0/3)*m1*l1*l1
I2 = (1.0/12)*m2*l2*l2
derutaT = 0.001
X1 = np.arange(0, 135, 13.5)
X2 = np.arange(0, 135, 13.5)
X1 = 1.0*X1*np.pi/180.0
X2 = 1.0*X2*np.pi/180.0
X3 = np.arange(-150,150,30)
X4 = np.arange(-150,150,30)

#u1 = np.arange(-135,135,27)
#u2 = np.arange(-135,135,27)
print(X1)
print(X2)

time = 0

with open('AutoencodeTrainData\TrainAll_one_step_xy3.csv','a+', newline='') as f:
    writer = csv.writer(f)
    for i in range(0,10):
        for j in range(0,10):
            for k in range(0,10):
                for l in range(0,10):
                    for m in range(0,10):
                        for n in range(0,10):
                            NextStates = np.zeros((3,4))
                            p = np.zeros((3,4))
                            NextStates[0][0] = X1[m]
                            NextStates[0][1] = X2[n]
                            NextStates[0][2] = X3[k]
                            NextStates[0][3] = X4[l]
                            p[0][0] = l1*np.cos(NextStates[0][0]) + l2*np.cos(NextStates[0][0]+NextStates[0][1])
                            p[0][1] = l1*np.sin(NextStates[0][0]) + l2*np.sin(NextStates[0][0]+NextStates[0][1])
                            p[0][2] = -l1*np.sin(NextStates[0][0])*NextStates[0][2]-l2*np.sin(NextStates[0][0]+NextStates[0][1])*(NextStates[0][2]+NextStates[0][3])
                            p[0][3] = l1*np.cos(NextStates[0][0])*NextStates[0][2]+l2*np.cos(NextStates[0][0]+NextStates[0][1])*(NextStates[0][2]+NextStates[0][3])

                            line = [1]
                            for step in range(0,2):
                                u1 = random.uniform(-270, 270)
                                u2 = random.uniform(-270, 270)
                                
                                #X = np.zeros(4)
                                #X[1] = np.pi-np.arccos((np.power(l1, 2)+np.power(l2, 2)-np.power(p[step][0], 2)-np.power(p[step][1], 2))/(2*l1*l2))
                                #beta = np.arccos((np.power(p[step][0], 2)+np.power(p[step][1], 2)+np.power(l1, 2)-np.power(l2, 2))/(2*l1*np.sqrt(np.power(p[step][0], 2)+np.power(p[step][1], 2))))
                                #X[0] = np.arctan(P[1]/P[0])-beta
                                #X[0] = np.arctan2(p[step][1], p[step][0])-beta
                                #J = np.zeros((2,2))
                                #J[0][0] = -l1*np.sin(X[0])-l2*np.sin(X[0]+X[1])
                                #J[0][1] = -l2*np.sin(X[0]+X[1])
                                #J[1][0] = l1*np.cos(X[0])+l2*np.cos(X[0]+X[1])
                                #J[1][1] = l2*np.cos(X[0]+X[1])
                                
                                #_J = J.T
                                #X[2:4] = np.dot(_J, p[step, 2:4])
                                #print("X",180*X/np.pi)
                                #print("NextStates",180*NextStates[step]/np.pi)
                                M = np.zeros((2,2))
                                h = np.zeros(2)
                                G = np.zeros(2)
                                f = np.zeros(2)
                                M[0][0] = m1*np.power(lg1, 2)+m2*np.power(l1, 2)+m2*np.power(lg2, 2)+I1+I2+2*m2*l1*lg2*np.cos(NextStates[step][1])
                                M[0][1] = m2*np.power(lg2, 2)+I2+m2*l1*lg2*np.cos(NextStates[step][1])
                                M[1][0] = M[0][1]
                                M[1][1] = m2*np.power(lg2, 2)+I2
                                _M = np.linalg.inv(M)
                                h[0] = -m2*l1*lg2*(2*NextStates[step][2]+NextStates[step][3])*NextStates[step][3]*np.sin(NextStates[step][1])
                                h[1] = m2*l1*lg2*np.power(NextStates[step][2], 2)*np.sin(NextStates[step][1])
                                G[0] = (g*l1*m2+g*lg1*m1)*np.cos(NextStates[step][0])+m2*g*lg2*np.cos(NextStates[step][0]+NextStates[step][1])
                                G[1] = m2*g*lg2*np.cos(NextStates[step][0]+NextStates[step][1])
                                f[0] = _M[0][0]*(-h[0]-G[0]+u1)+_M[0][1]*(-h[1]-G[1]+u2)
                                f[1] = _M[1][0]*(-h[0]-G[0]+u1)+_M[1][1]*(-h[1]-G[1]+u2)
                                NextStates[step+1][0] = NextStates[step][0]+derutaT*NextStates[step][2]
                                NextStates[step+1][1] = NextStates[step][1]+derutaT*NextStates[step][3]
                                NextStates[step+1][2] = NextStates[step][2]+derutaT*f[0]
                                NextStates[step+1][3] = NextStates[step][3]+derutaT*f[1]
                                x = l1*np.cos(NextStates[step+1][0]) + l2*np.cos(NextStates[step+1][0]+NextStates[step+1][1])
                                y = l1*np.sin(NextStates[step+1][0]) + l2*np.sin(NextStates[step+1][0]+NextStates[step+1][1])
                                dotX = -l1*np.sin(NextStates[step+1][0])*NextStates[step+1][2] - l2*np.sin(NextStates[step+1][0]+NextStates[step+1][1])*(NextStates[step+1][2]+NextStates[step+1][3])
                                dotY = l1*np.cos(NextStates[step+1][0])*NextStates[step+1][2] + l2*np.cos(NextStates[step+1][0]+NextStates[step+1][1])*(NextStates[step+1][2]+NextStates[step+1][3])
                                p[step+1][0] = x
                                p[step+1][1] = y
                                p[step+1][2] = dotX
                                p[step+1][3] = dotY

                                line.append(p[step][0])
                                line.append(p[step][1])
                                line.append(p[step][2])
                                line.append(p[step][3])
                                line.append(u1)
                                line.append(u2)

                            '''
                            Nextu1 = random.uniform(-270, 270)
                            Nextu2 = random.uniform(-270, 270)

                            M[0][0] = m1*np.power(lg1, 2)+m2*np.power(l1, 2)+m2*np.power(lg2, 2)+I1+I2+2*m2*l1*lg2*np.cos(NextStates[1])
                            M[0][1] = m2*np.power(lg2, 2)+I2+m2*l1*lg2*np.cos(NextStates[1])
                            M[1][0] = M[0][1]
                            M[1][1] = m2*np.power(lg2, 2)+I2
                            _M = np.linalg.inv(M)
                            h[0] = -m2*l1*lg2*(2*NextStates[2]+NextStates[3])*NextStates[3]*np.sin(NextStates[1])
                            h[1] = m2*l1*lg2*np.power(NextStates[2], 2)*np.sin(NextStates[1])
                            G[0] = (g*l1*m2+g*lg1*m1)*np.cos(NextStates[0])+m2*g*lg2*np.cos(NextStates[0]+NextStates[1])
                            G[1] = m2*g*lg2*np.cos(NextStates[0]+NextStates[1])
                            f[0] = _M[0][0]*(-h[0]-G[0]+Nextu1)+_M[0][1]*(-h[1]-G[1]+Nextu2)
                            f[1] = _M[1][0]*(-h[0]-G[0]+Nextu1)+_M[1][1]*(-h[1]-G[1]+Nextu2)

                            y1 = NextStates[0]+derutaT*NextStates[2]
                            y2 = NextStates[1]+derutaT*NextStates[3]
                            '''
                            #line = [X1[i], X2[j], X3[k], X4[l], u1[m], u2[n], 1, y1, y2, y3, y4]
                            #line = [X1[m], X2[n], NextStates[0], NextStates[1], u1, u2, Nextu1, Nextu2, 1, y1, y2]
                            
                            writer.writerow(line)
'''
with h5py.File('dataset.h5', 'w') as h5:
    h5.create_dataset('theta', data = data) 
    h5.create_dataset('u', data = label)
'''    
