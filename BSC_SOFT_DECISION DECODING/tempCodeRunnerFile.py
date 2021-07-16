.01,0.03,0.05,0.07,0.1,0.13,0.15,0.17,0.2] #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# plt.xlabel('Nos of Iterations')
# plt.ylabel('Nos of errors')
# plt.title('BSC Soft Decision Decoding - Algorithm Convergence')
# for pError in p:
#     ReceivedMessage = BSC(c, pError)  # Received Noisy Message from BSC
#     decoded_message,errors_per_iter = decoder(ReceivedMessage,CNs,VNs,dv,dc,n,u,100,pError)
#     plt.plot(errors_per_iter,label='p=' + str(pError))
# plt.legend()
# plt.show()