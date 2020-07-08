# import numpy as np
#
# def exercise(x):
#    do something here....
#    print(y)
#
# A = np.asarray([1,2])
# exercise(A)
## 1. create a 0 matrix of size 5x5
import numpy as np
x = (5,5)
print("1.",np.zeros(x))

## 2. print shape of a numpy matrix
def excercise():
    x = (5, 5)
    matrix = np.zeros(x)
    y = np.shape(matrix)

    print("2.",y)
excercise()
# ## 3. create a 1D vector with values ranging from 10 to 49 range(10,50)
a = np.random.randint(10,50,size=5)
print("3.",a)
## 4. reverse a 1D vector
print("4.",a[::-1])
## 5.create a 3x3x3 array with random values
#
def excercise(x):
    x = np.random.randn(3,3,3)
    print("5.",x)
excercise(x)
## 6.create a 2D 7x7 array with 1 on the border and 0 inside
x = np.ones((7,7))
x[1:6,1:6]=0
print("6.", x)
## 7. create a 5x5 random matrix then normalize it (x-mean(x)/std(x))
x = np.random.randn(5,5)
print(x)
xmean = x.mean()
xstd = x.std()
normalize = (x - xmean)/xstd
print("7.x's normalize")
print(normalize)

## 8. consider two random array A and B, check if they are equal
x = np.random.randint(0, 2, 4)
y = np.random.randint(0, 2, 4)

while np.allclose(x,y)==False:
    print("x=",x)
    print("y=",y)
    x = np.random.randint(0, 2, 4)
    y = np.random.randint(0, 2, 4)
    if np.allclose(x,y)==True:
        print("x=",x)
        print("y=",y)
        print("match")
## 9. subtract the mean of each row of a matrix
x = np.asarray([[1,2,3],[4,5,6],[7,8,9]])
mean = np.mean(x, axis= 1)
y = mean.reshape(3,1)
m = x - y
print(x)
print(mean)
print(m)
##10. given three random 2D matrix with same shapes, combine them to a 3D matrix
a = np.random.random((3,3))
b = np.random.random((3,3))
c = np.random.random((3,3))
d = np.dstack([a,b,c])
print(d.shape)

##11.given a interger matrix , find all unique elements
x = np.random.seed(133)
x = np.random.randint(10,20,25).reshape(5,5)
y = np.unique(x)
print(x)
print(y)
##12.consider a 4D matrix with shape 4x5x5x3, slice it to 4 3D matrixs
x= np.random.seed(133)
x = np.random.randint(0,10,300).reshape(4,5,5,3)
d1 = x[0,:,:,:]
d2 = x[1,:,:,:]
d3 = x[2,:,:,:]
d4 = x[3,:,:,:]
d1.reshape(5,5,3)

# print("x:",x)
print("d1:",d1)
print(d1.shape)
##13.convert a 1D Numpy array (class: ndarray) to a regular python list(class: list)
x = np.arange(5)
print(type(x))
print(x)
x1 = x.tolist()
print(x1)
print(type(x1))
