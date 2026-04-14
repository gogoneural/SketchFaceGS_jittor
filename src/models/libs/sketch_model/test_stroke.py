#-------------------------Test stroke dilate, 2022_2_15--------------------------
from PIL import Image
import cv2
import numpy as np

stroke_path = './test_sketch/2022-08-14-13-41-47_sket.png'
stroke_img = Image.open(stroke_path)
stroke_img = np.asarray(stroke_img)
gray = cv2.cvtColor(stroke_img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#cv.imshow("binary", binary)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dst = cv2.dilate(binary, kernel)
print(dst.shape)
cv2.imwrite('./test_sketch/2022-08-14-13-41-47_sket_refine.png', dst)


