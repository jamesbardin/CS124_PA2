import sys
import random



def val_gen(dim): 
    string = ''
    for i in range(0, 2 * dim**2):
        string += str(random.randint(0,1)) + '\n'
    return string

string = val_gen(64)

file = open('ascii.txt', 'w')
file.write(string)
file.close()