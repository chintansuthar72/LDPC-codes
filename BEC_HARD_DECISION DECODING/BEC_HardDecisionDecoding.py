import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

def BEC(encodedM, probability):
    noisy = np.zeros(len(encodedM),dtype=int) #  noisy Message
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
        codeword = np.zeros(n, dtype=int)  # For storing Previous iteration's Decoded Message
        for i in range(n): codeword[i] = VNs[i][0]

        # VN to CN
        for i in range(u):
            for j in range(dc, 2 * dc): CNs[i][j] = VNs[CNs[i][j - dc]][0]

        # Check for all CNs if they satisfy even parity or not
        # if yes then terminate loop process
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

        errors_per_iter[cycle] = sum(np.transpose(VNs)[0])
        # check if previous iteration's decoded message and present iteration's decoded message is same or not
        # if yes then terminate loop process
        check = True
        for i in range(n):
            if codeword[i] != VNs[i][0]:
                check = False
                break
        if check: break # if yes then terminate loop process
    return np.transpose(VNs)[0],errors_per_iter

mat = sio.loadmat('H_matrix/Hmatrix1.mat') # Load Parity Check Matrix From Matlab File
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

# For each probability in p
for pError in p:
    success = 0  # count success in Monte-Karlo simulation
    # Monte-Karlo simulation
    for n_sim in range(10):
        ReceivedMessage = BEC(c, pError)  # Received Noisy Message from BSC
        decoded_message,errors_per_iter = decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,5000)
        # check for if we got real transmitted message or not
        if decoded_message.sum()==0: success += 1
    s[f] = (success / 1)
    f += 1

plt.plot(p, s)
plt.xlabel('Error Probability of BEC Channel')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(p)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Success Probability of Decoding')
plt.title('BEC Hard Decision Decoding')
plt.grid()
plt.show()

# # Algorithm Convergence
# p_error = [0.1] #, 0.2, 0.3, 0.4, 0.5]
# plt.xlabel('Nos of Iterations')
# plt.ylabel('Nos of Errors')
# plt.title('BEC Hard Decision Decoding - Algorithm Convergence')
# ReceivedMessage = BEC(c, p_error[0])  # Received Noisy Message from BSC
# decoded_message,errors_per_iter = decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,5000)
# plt.plot(errors_per_iter)
# plt.show()