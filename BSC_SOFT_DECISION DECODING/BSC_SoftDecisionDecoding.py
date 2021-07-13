import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

def BSC(encodedM, probability):
    n = len(encodedM)
    noisy = np.zeros(n,dtype=int) # noisy Message
    error = np.zeros(n,dtype=int)
    for i in range(n):
        if random.random() <= probability: error[i] = 1
    for i in range(n): noisy[i] = (encodedM[i]+error[i])%2
    return noisy

def decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,maxItr,pError):
    errors_per_iter = np.zeros(maxItr, dtype=int)
    for i in range(n): VNs[i][0] = ReceivedMessage[i]

    # VN to CN for 0th iteration
    for i in range(u):
        for j in range(dc, 2 * dc): CNs[i][j] = pError if (VNs[int(CNs[i][j - dc])][0]) == 0 else (1 - pError)

    # Loop Process
    for cycle in range(maxItr):
        codeword = np.zeros(n, dtype=int)  # For storing Previous iteration's Decoded Message
        for i in range(n): codeword[i] = VNs[i][0]

        # CN to VN message passing (Sum modulo 2 process)
        for c_n in range(u):
            prob = 1
            for idx in range(dc, 2 * dc): prob *= (1 - 2 * CNs[c_n][idx])
            for idx in range(dc):
                for k in range(1, dv + 1):
                    if VNs[int(CNs[c_n][idx])][k] == c_n and CNs[c_n][idx + dc] != 0.5:
                        VNs[int(CNs[c_n][idx])][k + dv] = 0.5 * (1 - (prob / (1 - 2 * CNs[c_n][idx + dc])))
                        break

        # Doing majority voting for each VN
        for i in range(n):
            prob1 = pError if ReceivedMessage[i] == 0 else (1 - pError)
            prob0 = 1 - prob1
            for k in range(1 + dv, 2 * dv + 1):
                prob1 *= VNs[i][k]
                prob0 *= (1 - VNs[i][k])
            if prob1 >= prob0: VNs[i][0] = 1
            else: VNs[i][0] = 0

        # VN to CN message passing (Majority vote process)
        for i in range(n):
            prob1 = pError if ReceivedMessage[i] == 0 else (1 - pError)
            prob0 = 1 - prob1
            for k in range(1 + dv, 2 * dv + 1):
                prob1 *= VNs[i][k]
                prob0 *= (1 - VNs[i][k])
            for idx in range(1, dv + 1):
                p_1 = prob1 / VNs[i][idx + dv] if VNs[i][idx + dv] != 0 else 1
                p_0 = prob0 / (1 - VNs[i][idx + dv]) if (1 - VNs[i][idx + dv]) != 0 else 1
                for v_n in range(dc):
                    if CNs[int(VNs[i][idx])][v_n] == i:
                        CNs[int(VNs[i][idx])][v_n + dc] = p_1 / (p_1 + p_0) if p_1 + p_0 != 0 else 0
                        break
        errors_per_iter[cycle] = sum(np.transpose(VNs)[0])
    return np.transpose(VNs)[0],errors_per_iter

mat = sio.loadmat('H_matrix/Hmatrix1.mat')  # Load Parity Check Matrix From Matlab File
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

# # For each probability in p
# for pError in p:
#     success = 0  # count success in Monte-Karlo simulation
#
#     # Monte-Karlo simulation
#     for n_sim in range(1000):
#         ReceivedMessage = BSC(c, pError)  # Received Noisy Message from BSC
#         # check for if we got real transmitted message or not
#         decoded_message, errors_per_iter = decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,100)
#         if decoded_message.sum()==0: success += 1
#     s[f] = (success / 1000)
#     f += 1
#
# plt.plot(p, s)
# plt.xlabel('Error Probability of BSC Channel')
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.xticks(p)
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('Success Probability of Decoding')
# plt.title('BSC Soft Decision Decoding')
# plt.grid()
# plt.show()

# Algorithm Convergence
p_error = [0.2] #, 0.2, 0.3, 0.4, 0.5]
plt.xlabel('Nos of Iterations')
plt.ylabel('Nos of Errors')
plt.title('BSC Soft Decision Decoding - Algorithm Convergence')
ReceivedMessage = BSC(c, p_error[0])  # Received Noisy Message from BSC
decoded_message,errors_per_iter = decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,5000,p_error[0])
plt.plot(errors_per_iter)
plt.show()