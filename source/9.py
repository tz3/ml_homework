# import cv2
from cv2 import *

from cv2.cv import fromarray

image = imread('2.jpg')
img_rsz = resize(image, (0, 0), fx=2, fy=2)
hog = HOGDescriptor()
hog.setSVMDetector(HOGDescriptor_getDefaultPeopleDetector())
hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
result = list(hog.detectMultiScale(img_rsz, **hogParams))


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


found_filtered = []

for r in result[0]:
    insidef = False
    for q in result[0]:
        if inside(r, q):
            insidef = True
            break
    if not insidef:
        found_filtered.append(r)
for r in found_filtered:
    rx, ry, rw, rh = r
    tl = (rx + int(rw * 0.1), ry + int(rh * 0.07))
    br = (rx + int(rw * 0.9), ry + int(rh * 0.87))
    cv.Rectangle(fromarray(img_rsz), tl, br, (0, 255, 0), 3)

imshow('something', img_rsz)
waitKey(0)
