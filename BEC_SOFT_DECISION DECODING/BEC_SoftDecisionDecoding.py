import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

def BEC(encodedM, probability):
    noisy = np.zeros(len(encodedM),dtype=int) # noisy Message
    count = 0
    for i in encodedM:
        if random.random() <= probability: noisy[count] = -1
        else: noisy[count] = i
        count += 1
    return noisy

def decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,maxItr):
    errors_per_iter = np.zeros(maxItr, dtype=int)
    for i in range(n):
        VNs[i][0] = ReceivedMessage[i]
        if ReceivedMessage[i]==-1:errors_per_iter[0]+=1
    
    # VN to CN for 0th iteration
    for i in range(u):
        for j in range(dc, 2 * dc):
            if VNs[int(CNs[i][j - dc])][0] == 0: CNs[i][j] = 0
            elif VNs[int(CNs[i][j - dc])][0] == 1: CNs[i][j] = 1
            else: CNs[i][j] = 0.5

    # Loop Process
    for cycle in range(maxItr):
        if cycle!=0: errors_per_iter[cycle] = sum(np.transpose(VNs)[0])
        # CN to VN message passing (Sum modulo 2 process)
        for c_n in range(u):
            for idx in range(dc):
                prob = 1
                for k in range(dc,2*dc):
                    if idx+dc== k: continue
                    else: prob *= (1-2*CNs[c_n][k])
                for k in range(1,dv+1):
                    if VNs[int(CNs[c_n][idx])][k] == c_n:
                        VNs[int(CNs[c_n][idx])][k + dv] = 0.5*(1-prob)
                        break
                    
        # Doing majority voting for each VN - codeword decision
        for i in range(n):
            prob1 = 0  # just intialized
            if ReceivedMessage[i] == 0: prob1 = 0
            elif ReceivedMessage[i] == 1: prob1 = 1
            else: prob1 = 0.5
            prob0 = 1 - prob1
            for k in range(1 + dv, 2 * dv + 1):
                prob1 *= VNs[i][k]
                prob0 *= (1 - VNs[i][k])
            if prob1 >= prob0: VNs[i][0] = 1
            else: VNs[i][0] = 0

        # VN to CN message passing (Majority vote process)
        for i in range(n):
            prob1 = 0  # just intialised
            if ReceivedMessage[i] == 0: prob1 = 0
            elif ReceivedMessage[i] == 1: prob1 = 1
            else: prob1 = 0.5
            prob0 = 1 - prob1
            for idx in range(1, dv + 1):
                p_1 = p_0 = 1
                for k in range(1 + dv, 2 * dv + 1):
                    if idx+dv == k: continue
                    else:
                        p_1 *= VNs[i][k]
                        p_0 *= (1 - VNs[i][k])
                p_1 = prob1*p_1 
                p_0 = prob0*p_0 
                for v_n in range(dc):
                    if CNs[int(VNs[i][idx])][v_n] == i:
                        CNs[int(VNs[i][idx])][v_n + dc] = p_1 / (p_1 + p_0) 
                        break
        
    return np.transpose(VNs)[0],errors_per_iter

mat = sio.loadmat('D:\CT Project\BEC_SOFT_DECISION DECODING/H_matrix/Hmatrix2.mat')  # Load Parity Check Matrix From Matlab File
H = mat['H']  # Parity Check Matrix
dv = sum(np.transpose(H)[0])  # Degree of each VN
dc = sum(H[0])  # Degree of each CN
n = len(H[0])  # Length of Message
u = len(H)  # Nos of Parity bits
CNs = np.zeros((u, 2 * dc))  # for each CN - [connected VNs' index , that VNs' message]
VNs = np.zeros((n, (2 * dv) + 1))  # for each VN - [decoded Message bit , connected CNs' index , that CNs' message]
c = np.zeros(n, dtype=int)  # Transmitted Message in BSC
p = np.arange(0, 1.1, 0.1)  # BSC error Probabilities from 0 to 1 in  0.1 step increase
s = np.zeros(len(p))  # to store success probabilities
f = 0

# index storing for VNs
for i in range(n):
    index = 0
    for j in range(u):
        if H[j][i] == 1:
            VNs[i][1 + index] = j
            index += 1

# index storing for CNs
for i in range(u):
    index = 0
    for j in range(n):
        if H[i][j] == 1:
            CNs[i][index] = j
            index += 1

for pError in p:
    success = 0  # count success in Monte-Karlo simulation
    # Monte-Karlo simulation
    for Nsim in range(1000):
        ReceivedMessage = BEC(c, pError)  # Received Noisy Message from BSC
        # check for if we got real transmitted message or not
        decoded_message, errors_per_iter = decoder(ReceivedMessage, CNs, VNs, dv, dc, n, u, 25)
        if decoded_message.sum() == 0: success += 1
    s[f] = (success / 1000)
    f += 1

plt.plot(p, s)
plt.xlabel('Error Probability of BEC Channel')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(p)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Success Probability of Decoding')
plt.title('BEC Soft Decision Decoding')
plt.grid()
plt.show()

# Algorithm Convergence
# p =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
# plt.xlabel('Nos of Iterations')
# plt.ylabel('Nos of errors')
# plt.title('BEC Soft Decision Decoding - Algorithm Convergence')
# for pError in p:
#     ReceivedMessage = BEC(c, pError)  # Received Noisy Message from BSC
#     decoded_message,errors_per_iter = decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,50)
#     plt.plot(errors_per_iter,label='p=' + str(pError))
# plt.legend()
# plt.show()