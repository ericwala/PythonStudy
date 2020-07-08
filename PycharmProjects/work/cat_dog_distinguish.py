import os, sys
import PIL
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import numpy as np
import random
# a = r"cat_dog/dataset/training_set/cats"
b = r"cat_dog/dataset/training_set/dogs"
#
# ## make image size into 300X300
new_cat_path = r"cat_dog/dataset/training_set/new_cats/"
new_dog_path = r"cat_dog/dataset/training_set/new_dogs/"
os.makedirs(new_dog_path)
# n = 1
# for file in os.listdir(a):
#
#     f_img = a +"/"+file
#     img = Image.open(f_img)
#     img = img.resize((300,300))
#     img.save(r"cat_dog/dataset/training_set/new_cats/" + file)
#     old_name = r"cat_dog/dataset/training_set/new_cats/" + file
#     new_name = r"cat_dog/dataset/training_set/new_cats/" +"resize_cat_" +str(n)+".jpg"
#     os.rename(old_name, new_name)
#
#     n += 1
# print(n)
n = 1
for file in os.listdir(b):
    f_img = b + "/" + file
    img = Image.open(f_img).convert('L')
    img.save(new_dog_path + file )
    old_name = r"cat_dog/dataset/training_set/new_dogs/" + file
    new_name = r"cat_dog/dataset/training_set/new_dogs/" + "grayscale_dog_" + str(n) + ".png"
    os.rename(old_name, new_name)
    n += 1
print("down")

## 3. merge four resize cat picture into one.
os.makedirs('cat_dog/dataset/training_set/merge_cat')
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
n = 0
a = 0
for file in os.listdir(new_cat_path):
   im1 = cv2.imread("cat_dog/dataset/training_set/new_cats/resize_cat_"+str(n+1)+".jpg")
   im2 = cv2.imread("cat_dog/dataset/training_set/new_cats/resize_cat_"+str(n+2)+".jpg")
   im3 = cv2.imread("cat_dog/dataset/training_set/new_cats/resize_cat_" + str(n+3) + ".jpg")
   im4 = cv2.imread("cat_dog/dataset/training_set/new_cats/resize_cat_" + str(n+4) + ".jpg")
   im_v = concat_tile([[im1, im2],
                      [im3, im4]])
   if im_v is not None:
       cv2.imwrite('cat_dog/dataset/training_set/merge_cat/new_merge'+str(a+1)+'.jpg', im_v)
   n += 4
   a += 1
print("down")

##5.draw circle on dog pictures
##os.makedirs("cat_dog/dataset/training_set/draw_on_dog")

def circle_the_dog(limit):
    im = cv2.imread("cat_dog/dataset/training_set/new_dogs/grayscale_dog_1.png")

    list_xl = []
    list_xr = []
    list_yu = []
    list_yd = []

    # Start Drawing
    while True:

        compare_list = []

        # randomize a new center point
        center_x = random.randint(1, 300)
        center_y = random.randint(1, 400)

        # randomize a radius number
        radius = random.randint(10, 70)

        # calculate borders
        x1 = center_x - radius
        x2 = center_x + radius
        y1 = center_y - radius
        y2 = center_y + radius

        # check if out of border
        if x1 >= 0 and x2 <= 300 and y1 >= 0 and y2 <= 400:

            # checking process starts
            # first iteration
            if len(list_xl) == 0:
                print("First Interation")
                list_xl.append(x1)
                list_xr.append(x2)
                list_yd.append(y2)
                list_yu.append(y1)
                print(x1, x2, y1, y2)
                cv2.circle(im, (center_x, center_y), radius, (255, 255, 255), 2)

            elif len(list_xl) >= 1:
                overlap = False
                for i, j in enumerate(list_xl):
                    # print(list_xl[i], x1, center_x, x2, list_xr[i])
                    # print(list_yu[i], y1, center_y, y2, list_yd[i])
                    # print("===")
                    if list_xl[i] <= x1 <= list_xr[i] and list_yu[i] <= y1 <= list_yd[i] or \
                            list_xl[i] <= x1 <= list_xr[i] and list_yu[i] <= y2 <= list_yd[i] or \
                            list_xl[i] <= x2 <= list_xr[i] and list_yu[i] <= y1 <= list_yd[i] or \
                            list_xl[i] <= x2 <= list_xr[i] and list_yu[i] <= y2 <= list_yd[i] or \
                            list_xl[i] <= center_x <= list_xr[i] and list_yu[i] <= y2 <= list_yd[i] or \
                            list_xl[i] <= center_x <= list_xr[i] and list_yu[i] <= y1 <= list_yd[i] or \
                            list_xl[i] <= x1 <= list_xr[i] and list_yu[i] <= center_y <= list_yd[i] or \
                            list_xl[i] <= center_x <= list_xr[i] and list_yu[i] <= center_y <= list_yd[i] or \
                            list_xl[i] <= x2 <= list_xr[i] and list_yu[i] <= center_y <= list_yd[i]:
                        print("overlap!")
                        overlap = True

                if not overlap:
                    list_xl.append(x1)
                    list_xr.append(x2)
                    list_yd.append(y2)
                    list_yu.append(y1)
                    print(x1, x2, y1, y2)
                    cv2.circle(im, (center_x, center_y), radius, (255, 255, 255), 2)

        # Break when list length reach given limit
        if len(list_xl) == int(limit):
            break

    cv2.imshow("dog_img", im)
    cv2.imwrite("cat_dog/dataset/training_set/draw_on_dog/grayscale_dog_1.png",im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    circle_the_dog(limit=random.randint(2,5))

# # hough circle
# img = cv2.imread("cat_dog/dataset/training_set/draw_on_dog/grayscale_dog_1.png")
# coin = cv2.imread("coin.jpg")
# def hough_circle_demo(image):
#     # 霍夫圆检测对噪声敏感，边缘检测消噪
#     # dst = cv2.pyrMeanShiftFiltering(image, 100, 100)  # 边缘保留滤波EPF
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     img = cv2.medianBlur(gray, 3)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1 ,30, param1=10, param2=75, minRadius=10, maxRadius=80)
#     circles = np.uint16(np.around(circles))  #把circles包含的圆心和半径的值变成整数
#     for i in circles[0,:]:
#         cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 3)
#     cv2.imshow("circle image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# hough_circle_demo(img)
# hough_circle_demo(coin)
