__author__ = 'Nishant'
import numpy as np
import math as math
import matplotlib.pyplot as mp
import scipy as sc
import stat as st
import sys

class docs:
    X = []


# reads the input values and computes the x array
    def read_input(self, str1):
        f1 = open(str1, "r")

        for stk in f1:
            current_array = stk.strip().split()
            desired_array = [int(numeric_string) for numeric_string in current_array]
            self.X.append(desired_array)
        self.X = np.array(self.X)
        ones = np.ones(shape=(len(self.X) ,1))
        data = np.c_[ones, self.X]
        np.random.shuffle(data)
        w=[[0,data[0][3]]]

        X1 = np.array(data[:,[0,1,2]])
        y = data[:,3]
        for i in range(1, len(data)):
            sum=0
            for j in w:
                sum+=j[1]*self.find_k_guass(X1[j[0]],X1[i])
            if( y[i]*sum <= 0):
                w.append([i,y[i]])

        print(len(w))

        x_min, x_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        y_min, y_max = data[:, 2].min() - 1, data[:, 2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max),
                             np.arange(y_min, y_max))

        xf = xx.flatten()
        yf = yy.flatten()
        Z=[]
        for i in range(len(xf)):
            sum=0
            for j in w:
                sum+=j[1]*self.find_k_guass(X1[j[0]],[1,xf[i],yf[i]])
            Z.append(math.copysign(1,sum))
        # print(Z)

        # here "model" is your model's prediction (classification) function

        Z=np.array(Z)
        print(Z)
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        mp.contourf(xx, yy, Z, cmap=mp.cm.Paired)
        mp.axis('off')

        pos_x=[]
        neg_x=[]
        for i in data:
            if(i[3]>0):
                pos_x.append(i[:3])
            else:
                neg_x.append(i[:3])

        pos_x = np.array(pos_x)
        neg_x = np.array(neg_x)

        # Plot also the training points
        mp.scatter(pos_x[:, 1], pos_x[:, 2], c="Yellow")
        mp.scatter(neg_x[:,1], neg_x[:,2], c="Blue")
        mp.show()

    def find_k_quad(self, x,z):
        return pow(1+ np.dot(x,z), 2)
    def find_k_guass(self, x,z):
        norm = -1*(pow(np.linalg.norm(x-z),2))/(2*16)
        return math.exp(norm)



L = docs()
L.read_input(r"data2.txt")
#L.naive_Bayes()


