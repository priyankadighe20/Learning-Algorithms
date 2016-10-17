import string
import operator

NUM_OF_LETTERS = 5

# P(E,W)= P(E/W).P(W)
def probOfEvidenceAndWord(word, correct_guesses, incorrect_guesses):
    p_of_evidence = 1
    for letter in incorrect_guesses:
        if letter in word:
            p_of_evidence = 0
            break

    for index in range(0,NUM_OF_LETTERS):
        if correct_guesses[index] != '-' and correct_guesses[index] != word[index]:
            p_of_evidence = 0
            break

    return p_of_evidence * DICT[word]/TOTAL_COUNT

# sigma(probOfEvidenceAndWord) over all words
def probOfEvidenceAndAllWords(correct_guesses, incorrect_guesses):
    prob_sum = 0
    for word in DICT:
        prob_sum += probOfEvidenceAndWord(word, correct_guesses, incorrect_guesses)

    return prob_sum

#This is the same as P[Letter/Word] conditionally independent to state
def probOfLetterGivenWord(letter, word, correct_guesses):
    for index in range(0, NUM_OF_LETTERS):
        if correct_guesses[index] == '-' and letter == word[index]:
            return 1

    return 0


# open the file and read lines
file = open("D:\CSE 250a\hw1.txt")
lines = file.readlines()

# create a dictionary of words and their counts
DICT = {}
for line in lines:
    line = line.strip('\n')
    line = line.split(" ")
    DICT.update({line[0]: int(line[1])})


# print 8 most common words
SORTED_DICT = sorted(DICT.items(), key=operator.itemgetter(1), reverse=True)
print("\nEight most frequent  words (and their frequencies) are : ")
print(SORTED_DICT[0:8])

# print 8 least common words
SORTED_DICT = sorted(DICT.items(), key=operator.itemgetter(1))
print("\nEight least frequent words (and their frequencies) are : ")
print(SORTED_DICT[0:8])


# Get the sum of frequencies of all the words in a dictionary
TOTAL_COUNT = sum(DICT.values())

# take input from the user
CORRECT_GUESSES = list(input("\nEnter your correct guesses in the form -A-BC for a 5 letter word:"))
while len(CORRECT_GUESSES) != NUM_OF_LETTERS:
    CORRECT_GUESSES = list(input("\nEnter your correct guesses in the form -A-BC for a 5 letter word:"))

INCORRECT_GUESSES = list(input("\nEnter incorrect guesses in the form AEIOU:"))

# Calculate the denominator in the equation
PROB_EVIDENCE_ALL_WORDS = probOfEvidenceAndAllWords(CORRECT_GUESSES, INCORRECT_GUESSES)
if PROB_EVIDENCE_ALL_WORDS == 0:
    print("\n\nThe word does not exist in the dictionary")
    exit()

print(CORRECT_GUESSES, INCORRECT_GUESSES, PROB_EVIDENCE_ALL_WORDS)

# this list will have probability of all the letters from A-Z
PROBABILITY = 26*[0]
ALPHABET = list(string.ascii_uppercase)
iterator = 0

# Calculate the numerator of the equation and then calculate the equation
for letter in ALPHABET:
    if letter not in CORRECT_GUESSES and letter not in INCORRECT_GUESSES:
        for word in DICT:
            a = probOfLetterGivenWord(letter, word, CORRECT_GUESSES)
            b = probOfEvidenceAndWord(word, CORRECT_GUESSES, INCORRECT_GUESSES)
            PROBABILITY[iterator] = PROBABILITY[iterator] + a * b / PROB_EVIDENCE_ALL_WORDS
    iterator = iterator + 1

print("\n\nNext best guess is alphabet:")
print(ALPHABET[PROBABILITY.index(max(PROBABILITY))], max(PROBABILITY))

print(PROBABILITY)







