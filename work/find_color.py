import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import csv


def read_images():
    for root, dirs, files in os.walk('../data/sea-of-plants/images'):
        for filename in files:
            filepath = root + os.sep + filename
            img = cv2.imread(filepath)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)         
            # cv2.destroyAllWindows()
            ## ^^ <UNCOMMENT TO VIEW IMG>
            # color_space(img,'HSV', filename)
            ## ^^ <UNCOMMENT TO DO COLOR SPACE PLOT>
            a, b, c = segment(img)
            with open(r'document.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([filename,a,b,c])
            # break

# def color_space(img, color, name): # THE PLOT IS INSANELY SLOW
#     name.replace('.jpg','.png') # we want to save plots as png

#     if color == 'RGB': # check how to conver color space
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     else:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     r, g, b = cv2.split(img)
#     fig = plt.figure()
#     axis = fig.add_subplot(1, 1, 1, projection="3d")
#     pixelcolors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
#     norm = colors.Normalize(vmin=-1.,vmax=1.)
#     norm.autoscale(pixelcolors)
#     pixelcolors = norm(pixelcolors).tolist()
#     axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixelcolors, marker=".")

#     if color == 'RGB': # for visuals
#         axis.set_xlabel("Red")
#         axis.set_ylabel("Green")
#         axis.set_zlabel("Blue")
#     else:
#         axis.set_xlabel("Hue")
#         axis.set_ylabel("Saturation")
#         axis.set_zlabel("Value")

#     plt.savefig('./plots/color_space/{}{}'.format(color,name)) # save because show just crashes PC 


def segment(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    Lred = (123, 66, 0)
    Hred = (180, 255, 255)

    Lgreen = (10, 8, 0)
    Hgreen = (156,255,111)
    
    maskR = cv2.inRange(imgHSV, Lred, Hred)
    resultR = cv2.bitwise_and(imgRGB, imgRGB, mask=maskR)

    maskG = cv2.inRange(imgHSV, Lgreen, Hgreen)
    resultG = cv2.bitwise_and(imgRGB, imgRGB, mask=maskG)



    plt.subplot(2, 2, 1)
    plt.imshow(maskR, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(resultR)
    plt.subplot(2 ,2, 3)
    plt.imshow(maskG, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(resultG)
    a,b,c = calc_px(img, maskR, maskG)
    # print("-----------------------")
    # print(a)
    # print(b)
    # print(c)
    # plt.show()
    # print(img.shape)
    # print(img.size)
    # print(np.prod(img[:,:,1].shape))
    # print(np.count_nonzero(maskR))
    return a, b, c
    

def calc_px(img, maskR, maskG):
    imgPX = np.prod(img[:,:,1].shape)
    maskRPX = np.count_nonzero(maskR)
    maskGPX = np.count_nonzero(maskG)
    # print(imgPX)
    # print(maskRPX)
    # print(maskGPX)

    return imgPX, maskRPX, maskGPX



read_images()
