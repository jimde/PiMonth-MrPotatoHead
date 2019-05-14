#!/usr/bin/python3

import cv2
import numpy as np
import sys

def resize_image(img, height):
    ratio = float(height) / img.shape[0]
    dim = (int(img.shape[1]*ratio), height)

    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return image

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    potato = cv2.imread('images/potato-square-wide.png', cv2.IMREAD_UNCHANGED)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    print("{} faces found!".format(len(faces)))

    for (x,y,w,h) in faces:
        smallpotato = resize_image(potato, h)

        alpha_potato = smallpotato[:, :, 3] / 255.0
        alpha_img = 1.0 - alpha_potato

        y1, y2 = y, y + smallpotato.shape[0]
        x1, x2 = x, x + smallpotato.shape[1]

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_potato * smallpotato[:, :, c] + alpha_img * img[y1:y2, x1:x2, c])


    cv2.imshow('potato', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()