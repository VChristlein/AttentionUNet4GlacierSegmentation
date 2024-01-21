import imutils
import cv2
# input: image to be data_augmented
# K-> k/4 >= 1: FLIP HORIZONTAL, k%4==0: NO_ROT, k%4==1: 90, k%4==2: 180, k%4==3: 270
# return: augmented image
def data_augmentation(image,k):
    out = image.copy()
    if k/4 >= 1:
        out=cv2.flip(out,0)
    if k % 4 == 1:
        out = imutils.rotate_bound(out,90)
    elif k % 4 == 2:
        out = imutils.rotate_bound(out,180)
    elif k % 4 == 3:
        out = imutils.rotate_bound(out,270)
    return out
