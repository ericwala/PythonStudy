import numpy as np
##1
# arr = np.array([1,1,5,8,2,2,3,3,8])
# uni = np.unique(arr)
#
# def findSingle(arr):
#     s = 2*sum(uni) - sum(arr)
#
#     return s
# print(arr)
# a = findSingle(arr)
# if a==0:
#     print("none")
# else:
#     print(findSingle(arr))
##2

# arr = np.asarray([[0,1,1],
#                   [1,1,1],
#                   [1,1,0]])
# # def rever(arr):
# #     d1 = arr[0, :]
# #     d1r = d1[::-1]
# #     d2 = arr[1, :]
# #     d2r = d2[::-1]
# #     d3 = arr[2, :]
# #     d3r = d3[::-1]
# #     arr_reverse = np.dstack([d1r, d2r, d3r])
# #     arr_reverse.reshape(3,3,1)
# #     return arr_reverse
#
# rematrix = np.fliplr(arr)
# inmatrix = np.where((rematrix==0)|(rematrix==1),rematrix^1,rematrix)
# print(rematrix)
# print(inmatrix)
# print(inmatrix.shape)
##3.
def reverse(x:int) :
    sign = [1,-1][x<0]
    rst =  sign * int(str(abs(x))[::-1])
    return rst if -(2**31) < rst < 2**31-1 else 0
print(reverse(120))

