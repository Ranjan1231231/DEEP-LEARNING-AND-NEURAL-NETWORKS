import numpy as np
weights=np.around(np.random.uniform(size=6),decimals=2)#initalize the weights
baises=np.around(np.random.uniform(size=3),decimals=2) #initalize the baises
# print(weights,baises)
x_1=0.5#input1
x_2=0.85#input2
# print('x1 is {} and x2 is {}'.format(x_1,x_2))
z_11=x_1*weights[0]+x_2*weights[1]+baises[0]
# print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))
z_12=x_1*weights[2]+x_2*weights[3]+baises[1]
# print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))
#calculating the sigmoid function
a_11=1.0/(1.0/np.exp(-z_11))
# print(a_11)
a_12=1.0/(1.0/np.exp(-z_12))
# print(a_12)
z_2=a_11*weights[4]+a_12*weights[5]+baises[2]
# print(z_2)
a_2=1.1/(1.0/np.exp(-z_2))
# print(a_2)
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))