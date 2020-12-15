import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import csv


def visualize_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)         
    cv2.destroyAllWindows()


def color_space(img, color, name): # THE PLOT IS INSANELY SLOW
    name.replace('.jpg','.png') # we want to save plots as png

    if color == 'RGB': # check how to conver color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    r, g, b = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixelcolors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixelcolors)
    pixelcolors = norm(pixelcolors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixelcolors, marker=".")

    if color == 'RGB': # for visuals
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
    else:
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")

    plt.savefig('./plots/color_space/{}{}'.format(color,name)) # save because show just crashes PC 

def draw(img, color):
    if color == "red":
        img[np.all(img == (0, 0, 0), axis=-1)] = (0,255,0)
    else:
        img[np.all(img == (0, 0, 0), axis=-1)] = (255,0,0)

    return img


def segment(img, transform):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    Lred = (123, 66, 0)
    Hred = (180, 255, 255)

    Lgreen = (21,8,0) #Lgreen = (21,8,0) used to be like this for HSV, somehow think BGR performs better for green
    Hgreen = (117,255,74) #Hgreen = (117,255,74)

    maskR = cv2.inRange(imgHSV, Lred, Hred)
    maskG = cv2.inRange(imgHSV, Lgreen, Hgreen)

    kernelSize = np.ones((3, 3))# Play with different kernel sizes

    if transform == 'Y': # swap MORPH_OPEN and MORPH_CLOSE to play around 
        transR = cv2.morphologyEx(maskR, cv2.MORPH_OPEN, kernelSize) 
        transG = cv2.morphologyEx(maskG, cv2.MORPH_OPEN, kernelSize)
        resultR = cv2.bitwise_and(imgRGB, imgRGB, mask=transR)
        resultG = cv2.bitwise_and(imgRGB, imgRGB, mask=transG)
    else:
        resultR = cv2.bitwise_and(imgRGB, imgRGB, mask=maskR)
        resultG = cv2.bitwise_and(imgRGB, imgRGB, mask=maskG)

    plt.subplot(2, 3, 1)
    plt.imshow(maskR, cmap="gray")
    plt.subplot(2, 3, 2)
    plt.imshow(resultR)
    plt.subplot(2 ,3, 4)
    plt.imshow(maskG, cmap="gray")
    plt.subplot(2, 3, 5)
    plt.imshow(resultG)
    plt.subplot(2, 3, 3)
    plt.imshow(draw(resultR, 'red'))
    plt.subplot(2, 3, 6)
    plt.imshow(draw(resultG, 'green'))
    a,b,c = calc_px(img, maskR, maskG)
    plt.show()

    return a, b, c
    

def calc_px(img, maskR, maskG):
    imgPX = np.prod(img[:,:,1].shape)
    maskRPX = np.count_nonzero(maskR)
    maskGPX = np.count_nonzero(maskG)

    return imgPX, maskRPX, maskGPX

def main():
    for root, dirs, files in os.walk('../data/sea-of-plants/images'):
        for filename in files:
            filepath = root + os.sep + filename
            img = cv2.imread(filepath) # all imgs on OpenCV are BGR by default. Always convert to BGR before imshow
            # visualize_img(img)
            # color_space(img,'HSV', filename)
            a, b, c = segment(img, 'Y')
            with open(r'document.csv', 'a') as f:
                writer = csv.writer(f)
                # writer.writerow([filename,a,b,c]) # write to file
            # break

main()
