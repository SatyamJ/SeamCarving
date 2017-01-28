__author__ = 'Satyam'
from pylab import *
from skimage import img_as_float

img = imread("HJoceanSmall.png")
img = img_as_float(img)
w, h = img.shape[:2]
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]
figure()
gray()

subplot(2, 2, 1); imshow(img); title('RGB')
subplot(2, 2, 2); imshow(R); title('Red')
subplot(2, 2, 3); imshow(G); title('Green')
subplot(2, 2, 4); imshow(B); title('Blue')

show()


# print B[25][250]

