import numpy as np
import string
import matplotlib.pyplot as py

def calculate_viterbi(t, j):
    i = np.argmax(L[:,t] + np.log(A[:,j]))
    return i


def calculate_L():

    L[:,0] = np.log(PI) + np.log(B[:, O[0]])

    for t in range(1, T):
        for j in range(0,N):
            L[j,t] = np.amax(L[:,t-1] + np.log(A[:,j]))

        L[:, t] = L[:, t] + np.log(B[:, O[t]])


def calculate_S():
    S[T-1] = np.argmax(L[:, T-1])

    for t in range(T-2, -1, -1):
        S[t] = calculate_viterbi(t+1, S[t+1])


def print_message():
    message = []
    letters = list(string.ascii_uppercase)

    prev = S[0]
    message.append(letters[S[0]])
    for i in range(1, len(S)):
        if S[i] != prev:
            message.append(letters[S[i]])
        prev = S[i]

    print(''.join(str(e) for e in message))

# Open and process the PI_i values for initial state distribution
file = open("hw6_initialStateDistribution.txt")
PI = list(map(lambda s: float(s.strip('\n')), list(file.readlines())))

# Open and fill in the transition matrix A
A = []
file = open("hw6_transitionMatrix.txt")
for line in file.readlines():
    line = line.strip('\n')
    line = line.split(" ")
    line.remove('')
    line = list(map(lambda s: float(s), line))
    A.append(line)

A = np.array(A)


# Open and fill the emission matrix B
B = []
file = open("hw6_emmissionMatrix.txt")
for line in file.readlines():
    line = line.strip('\n')
    line = line.split('\t')
    line = list(map(lambda s: float(s), line))
    B.append(line)

B = np.array(B)



# Open and fill the observation matrix
O = []
file = open("hw6_observations.txt")
for line in file.readlines():
    line = line.strip()
    line = line.split()
    line = list(map(lambda s: int(s), line))
    O.extend(line)

O = np.array(O)

T = len(O)
N = 26
L = np.zeros(shape=(N,T))
calculate_L()

S = np.array(T*[0])
calculate_S()

print_message()


py.plot(S)
py.xlabel("Time(t)")
py.ylabel("State(S)")
py.show()

