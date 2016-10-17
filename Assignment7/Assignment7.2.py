# Problem 7.2
import numpy as np


def create_trans_matrix(path):
    p = np.zeros(shape=(S,S))
    file = open(path)
    for line in file.readlines():
        line = line.strip('\n')
        line = line.split('  ')
        p[int(line[0])-1, int(line[1])-1] = float(line[2])

    return p

def calculate_a(s, v):
    a1 = np.dot(Pa1[s, :], v)
    a2 = np.dot(Pa2[s, :], v)
    a3 = np.dot(Pa3[s, :], v)
    a4 = np.dot(Pa4[s, :], v)
    a = [a1[0], a2[0], a3[0], a4[0]]

    return np.array(a)

def calculate_transit(v):
    t = np.zeros(shape=(S, 1))
    for s in range(S):
        a = calculate_a(s, v)
        t[s] = np.amax(a)

    return t


def calculate_pi(v):
    p = np.array(S*[[1]])
    for s in range(S):
        a = calculate_a(s, v)
        p[s] = np.argmax(a) + 1

    return p


def update_value(v):
    val = R.transpose() + 0.99 * calculate_transit(v)
    return val


def calculate_pi_val(pi):
    p = np.zeros(shape=(S, S))
    for ds in range(S):
        for s in range(S):
            p[s, ds] = Pa[pi[s]-1][s,ds]

    inv = np.linalg.inv(I - 0.99 * p)
    v = np.dot(inv, R.transpose())

    return v


# Reward matrix
r_file = open("hw7_rewards.txt")
R = list(r_file.readlines())
R.remove('\n')
R = list(map(lambda s: int(s.strip('\n')), R))
R = np.array([R])

S = 81
A = 4

# Creating transition matrices
Pa1 = create_trans_matrix("hw7_prob_a1.txt")
Pa2 = create_trans_matrix("hw7_prob_a2.txt")
Pa3 = create_trans_matrix("hw7_prob_a3.txt")
Pa4 = create_trans_matrix("hw7_prob_a4.txt")
Pa = [Pa1, Pa2, Pa3, Pa4]

# Value iteration
V = np.zeros(shape=(S, 1))
while True:
    V1 = update_value(V)
    if (V == V1).all():
        break
    V = V1

# print non-zero v values
v_list = list(V.transpose()[0])
v_list = list(filter(lambda a: a != 0, v_list))
print(v_list)

# Part b) Calculate optimal PI from V
PI_1 = calculate_pi(V)
print(PI_1)

# Part c) Policy iteration
I = np.identity(S)
PI_2 = np.array(S*[[1]])

for i in range(10):
    val = calculate_pi_val(PI_2)
    PI_2 = calculate_pi(val)

# PI_1 and PI_2 are equal
if (PI_1 == PI_2).all():
    print("Same pi values for both methods")
