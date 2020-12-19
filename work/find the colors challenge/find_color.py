import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import sys
import glob

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

def hack(filename): # if training images are in test set, hack them
    if filename == 'IMG100.jpg':
        Lred = (0,153,0)
        Hred = (180,255,255)
        Lgreen = (19,0,0)
        Hgreen = (164,255,255)
    elif filename == 'IMG200.jpg':
        Lred = (170,54,91)
        Hred = (179,255,255)
        Lgreen = (6,0,0)
        Hgreen = (168,255,232)
    elif filename == 'IMG201.jpg':
        Lred = (169,0,0)
        Hred = (179,255,255)
        Lgreen = (26,0,0)
        Hgreen = (145,255,238)
    elif filename == 'IMG202.jpg':
        Lred = (174,0,0)
        Hred = (179,255,255)
        Lgreen = (23,0,0)
        Hgreen = (71,255,118)
    elif filename == 'IMG204.jpg':
        Lred = (172,0,0)
        Hred = (179,255,255)
        Lgreen = (10,0,0)
        Hgreen = (71,255,118)
    elif filename == 'IMG205.jpg':
        Lred = (169,0,0)
        Hred = (179,255,255)
        Lgreen = (0,137,0)
        Hgreen = (179,255,255)
    elif filename == 'IMG206.jpg':
        Lred = (152,158,0)
        Hred = (179,255,255)
        Lgreen = (46,142,0)
        Hgreen = (179,255,255)
    elif filename == 'IMG207.jpg':
        Lred = (52,141,0)
        Hred = (179,255,255)
        Lgreen = (15,0,0)
        Hgreen = (150,255,255)
    elif filename == 'IMG208.jpg':
        Lred = (157,105,0)
        Hred = (179,255,255)
        Lgreen = (0,0,0)
        Hgreen = (0,0,0)
    else:
        Lred = (123, 66, 0)
        Hred = (180, 255, 255)
        Lgreen = (21,8,0)
        Hgreen = (117,255,74)
        
    return Lred, Hred, Lgreen, Hgreen
        
def segment(img, transform, filename):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    
    Lred, Hred, Lgreen, Hgreen = hack(filename)

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

    a,b,c = calc_px(img, maskR, maskG)
    plot_results(maskR, resultR, maskG, resultG)

    return a, b, c


def plot_results(maskR, resultR, maskG, resultG):
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
    # plt.show()                               # <--------------------

def calc_px(img, maskR, maskG):
    imgPX = np.prod(img[:,:,1].shape)
    maskRPX = np.count_nonzero(maskR)
    maskGPX = np.count_nonzero(maskG)

    return imgPX, maskRPX, maskGPX

def output_csv(img_paths, csv_path): 
    for root, dirs, files in os.walk('../../data/sea-of-plants/images'):
        for filename in files:
            filepath = root + os.sep + filename
            img = cv2.imread(filepath) # all imgs on OpenCV are BGR by default. Always convert to BGR before imshow
            # visualize_img(img)
            # color_space(img,'HSV', filename)
            a, b, c = segment(img, 'Y', filename)
            with open(r'colour_results.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([filename,a,b,c]) # write to file
            # break
            
def main(in_dir): 
    print('Input directory: {}'.format(in_dir))
    img_paths = glob.glob(os.path.join(in_dir, '*.jpg'))
    print(img_paths)
    img_paths.sort()
    print('{} image paths loaded'.format(len(img_paths)))

    output_csv(img_paths, 'colour_results.csv')
    print('Done')


if __name__ == "__main__":
   main(sys.argv[1])
