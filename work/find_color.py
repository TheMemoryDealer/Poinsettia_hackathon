import cv2
import os


def read_images():
    for root, dirs, files in os.walk('../data/sea-of-plants/images'):
        for filename in files:
            # print(filename)
            filepath = root + os.sep + filename
            img = cv2.imread(filepath)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)         
            # cv2.destroyAllWindows()

read_images()
