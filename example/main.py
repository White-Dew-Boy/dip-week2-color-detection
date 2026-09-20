import os

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# 图片路径从环境变量读取（.env 里的 IMG_PATH，相对项目根目录）
img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.getenv("IMG_PATH"))

# 用 imdecode 读取：cv2.imread 在本项目的中文路径下会返回 None
frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)

cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
k = cv2.waitKey(0) & 0xFF

cv2.destroyAllWindows()
