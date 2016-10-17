
# coding: utf-8

# In[97]:

import numpy as npy
import math
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline


# In[3]:

data = [[2,1,1],[1,20,1],[1,5,1],[4,1,1],[1,40,1],[3,30,1]]
data_matrix = npy.array(data)
labels = [-1,-1,-1,1,1,1]



# In[4]:

def sigmoid(x):
    if x >= 0:
        return math.exp(-npy.logaddexp(0, -x))
    else:
        return math.exp(x - npy.logaddexp(x, 0))
    #return math.exp(-npy.logaddexp(0, -x))

def gradient(w):
    grad = npy.zeros(data_matrix.shape[1])
    for i in range(data_matrix.shape[0]):
        grad = grad + labels[i] * data_matrix[i] * sigmoid(-1 * labels[i] * npy.dot(w,data_matrix[i]))
    return -1 * grad


# In[5]:

def loss_func(w):
        sum = 0
        for i in range(data_matrix.shape[0]):
            sum += npy.logaddexp(0,-labels[i]*npy.dot(w,data_matrix[i]))
        return sum


# In[57]:

ws=[]

def gradient_des_line_search(data, l, m):
        w = npy.zeros(data_matrix.shape[1])
        t = 0
        beta = 0.5
        alpha = 0.2
        gr = gradient(w)
        while npy.linalg.norm(gr) > 0.0001:
            eta = 1
            while loss_func(w-eta*gr) > loss_func(w) - alpha*eta*math.pow(npy.linalg.norm(gr),2):
                eta = beta * eta
            w_temp = w - eta * gr
            if t == 100:
                ws.append(w)
            if t == 500:
                ws.append(w)
            if t == 1000:
                ws.append(w)
            if t == 1000:
                ws.append(w)
        
            w = w_temp
            gr = gradient(w)
            t += 1
            m -= 1
        print(w)
        print(t)
        ws.append(w)
        return ws


# In[58]:



ws = gradient_des_line_search(data_matrix,labels,500)


# In[60]:

print(ws)

for w in ws:
    plt.plot([2,1,1],[1,20,5],'ro')
    plt.plot([4,1,3],[1,40,30],'bo')
    plt.axis([-5,5,0,50])

    # line = matplotlib.lines.Line2D([-w[2]/w[1],0],[0,-w[2]/w[0]],linestyle='dashed',color='k')
    #plt.axes().add_line(line)

    xs = npy.arange(-5,5)
    ys = [(w[0]*x+w[2])/-w[1] for x in xs]

    #plt.plot([0,-w[2]/w[0]], [-w[2]/w[1],0])
    plt.plot(xs,ys)


# In[92]:

data = []
labels = []

mean1 = [10,10]
cov1 = [[600,0],[0,300]]
x1, y1 = npy.random.multivariate_normal(mean1, cov1, 50).T

for i in range(0,len(x1)):
    data.append([x1[i],y1[i],1])
    labels.append(1)

mean2 = [-30,-20]
cov2 = [[150,0],[0,400]]
x2, y2 = npy.random.multivariate_normal(mean2, cov2, 50).T


for i in range(0,len(x2)):
    data.append([x2[i],y2[i],1])
    labels.append(-1)
    
plt.plot(x1,y1,'ro')
plt.plot(x2,y2,'bo')
plt.axis('equal')



# In[93]:

data_matrix = npy.array(data)
ws = gradient_des_line_search(data_matrix,labels,500)
w = ws[0]
print(w)


# In[98]:

plt.plot(x1,y1,'ro')
plt.plot(x2,y2,'bo')
plt.axis([-60,80,-60,40])
xs = npy.arange(-25,25)
ys = [(w[0]*x+w[2])/-w[1] for x in xs]
plt.plot(xs,ys)
plt.show()


# In[ ]:



