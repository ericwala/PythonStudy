# print("hello world!")
# this is a note
'''
This is a note in paragraph
'''



# integer and float
money = 100
print(type(money)) # integer
money_2 = 100.0
print(type(money + money_2))
print(type(money)) # float: 浮點數 (會有誤差)
print(154.2 * 32423.232434) # package: math, numpy
debt = -100
# print(money + debt)


# basic string calculation
my_name = 'Ricardo'
ricardo_is = 'hansome'
print(my_name + ricardo_is)
print(my_name, ricardo_is)
print(my_name + str(money)) # string: word " ' is the same

# boolen
print(2 == 3) # == check if the same, note that memory address is actually checked
print(True == 1)
print(True == 0)
print(False == 0)
print(True + 1 - 3)
print(2 != 3)

# tiao tuo zi yuan
print('"')
print("hello\nricardo")
print("hello\tricardo") # tab = 4 space


# numpy exercise
import numpy as np
first_array = np.array([[1, 0],[0, 1]])
second_array = np.array([[3, 4], [5, 7]])
simple = first_array * second_array
real = first_array.dot(second_array)
print(simple)
print(real)

# about i
i = 4
i += 5
print(i)

# about input
number = input('input a number!!!!!:      ') # input is always a string
# print(int(number) + 5)
# print(number + 5)








