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
    errors_per_iter = np.zeros(maxItr,dtype=int)
    for i in range(n): VNs[i][0] = ReceivedMessage[i]

    # Loop Process
    for cycle in range(maxItr):
        for i in range(n):
            if VNs[i][0] == -1: errors_per_iter[cycle]+=1
        # VN to CN
        for i in range(u):
            for j in range(dc, 2 * dc): CNs[i][j] = VNs[CNs[i][j - dc]][0]

        # CN to VN
        count = 0
        for c_n in range(u):
            sumOfAll_CNs = sum(CNs[c_n][dc:2 * dc])
            temp_list = CNs[c_n][dc:2 * dc]
            erasures = np.where(temp_list==-1)[0]
            if len(erasures) == 0: count += 1
            elif len(erasures)==1:
                vn_idx = CNs[c_n][erasures[0]]
                if (sumOfAll_CNs-1)%2==0: VNs[vn_idx][0]=0
                else: VNs[vn_idx][0] = -1
        if count == u: break # if yes then terminate loop process
    return np.transpose(VNs)[0],errors_per_iter

mat = sio.loadmat('D:\CT Project\BEC_HARD_DECISION DECODING/H_matrix/Hmatrix1.mat') # Load Parity Check Matrix From Matlab File
H = mat['H']  # Parity Check Matrix
dv = sum(np.transpose(H)[0]) # Degree of each VN
dc = sum(H[0])  # Degree of each CN
n = len(H[0])  # Length of Message
u = len(H)  # Nos of Parity bits
CNs = np.zeros((u, 2 * dc), dtype=int)  # for each CN - [connected VNs' index , that VNs' message]
VNs = np.zeros((n, dv + 1),dtype=int)  # for each VN - [decoded Message bit , connected CNs' index , that CNs' message]

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

c = np.zeros(n, dtype=int)  # Transmitted Message in BSC
p = np.arange(0, 1.1, 0.1)  # BSC error Probabilities from 0 to 1 in  0.1 step increase
s = np.zeros(len(p))  # to store success probabilities
f = 0

# This is for Monte-karlo simulation
# for pError in p:
#     success = 0  # count success in Monte-Karlo simulation
#     # Monte-Karlo simulation
#     for n_sim in range(1000):
#         ReceivedMessage = BEC(c, pError)  # Received Noisy Message from BSC
#         decoded_message,errors_per_iter = decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,50)
#         # check for if we got real transmitted message or not
#         if decoded_message.sum()==0: success += 1
#     s[f] = (success / 1000)
#     f += 1
#
# plt.plot(p, s)
# plt.xlabel('Error Probability of BEC Channel')
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xticks(p)
# plt.xlim([0, 1])
# plt.ylim([0, 1.1])
# plt.ylabel('Success Probability of Decoding')
# plt.title('BEC Hard Decision Decoding ')
# plt.grid()
# plt.show()

# # Algorithm Convergence
# p = [0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7]
# plt.xlabel('Nos of Iterations')
# plt.ylabel('Nos of errors')
# plt.title('BEC Hard Decision Decoding - Algorithm Convergence')
# for pError in p:
#     ReceivedMessage = BEC(c, pError)  # Received Noisy Message from BSC
#     decoded_message,errors_per_iter = decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,50)
#     plt.plot(errors_per_iter,label='p=' + str(pError))
# plt.legend()
# plt.show()