__author__ = 'Nishant'
import numpy as np
import math as math
import matplotlib as mapLib

class docs:
    X = []
    Y=[]
    buckets=[]
    count=[]
    test_labels=[]
    doc_vectors=[]
    doc_label=[]
# reads the input values and computes the x array
    def read_input(self, str1, str2, str3, str4, str5, str6):
        k= 20
        V= 61188
        test_docs = 1000
        f1 = open(str1, "r")
        f3 = open(str3, "r")
        f4 = open(str4, "r")
        f6 = open(str6, "r")
        for stk in f1:
            current_array = stk.strip().split()
            desired_array = [int(numeric_string) for numeric_string in current_array]
            self.X.append(desired_array)

        for stk in f3:
            self.Y.append(int(stk))

        self.doc_label = np.zeros(k)

        for i in self.Y:
            self.doc_label[i-1]+=1

        total_docs = np.sum(self.doc_label)
        self.doc_label = [np.log(x/total_docs) for x in self.doc_label]
        self.buckets = np.zeros((k,V))

        for i in self.X:
            label = self.Y[i[0]-1]
            self.buckets[label-1][i[1]-1]+= i[2]

        self.count = np.sum(self.buckets,axis=1)
       #######################################################
        self.doc_vectors = np.zeros((1000,V))
        for stk in f4:

            current_array = stk.strip().split()
            test_words = [int(numeric_string) for numeric_string in current_array]
            if test_words[0] <= 1000:
                self.doc_vectors[test_words[0]-1][test_words[1]-1]+=test_words[2]

        for stk in f6:
            self.test_labels.append(int(stk))

    def naive_Bayes(self):
        V= 61188
        label_logs=np.zeros((20,V))
        no_of_test = 1000
        prediction = np.zeros(no_of_test);
        j=0
        for i in self.buckets:
            label_logs[j]= [np.log((word+1)/(self.count[j]+V)) for word in i]
            j+=1
        x=0
        for i in self.doc_vectors:
          predicted_label = np.argmax([ (np.dot(i,k)+l) for k,l in zip(label_logs, self.doc_label) ])
          prediction[x]= predicted_label+1
          x+=1

        diff = np.subtract(prediction,self.test_labels[:1000])
        print((np.count_nonzero(diff)/no_of_test)*100)

L = docs()
L.read_input(r"data\train.data",  r"data\train.map", r"data\train.label" , r"data\test.data",  r"data\test.map", r"data\test.label")
L.naive_Bayes()


