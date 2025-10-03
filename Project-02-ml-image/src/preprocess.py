import os
import cv2
import numpy as np
from PIL import Image

def resize_image(image, size=(1024, 1024)):
img_pil = Image.fromarray(image)
img_resized = img_pil.resize(size, Image.LANCZOS)
return np.array(img_resized)

def apply_clahe(image):
if len(image.shape) == 3:
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
img_gray = image
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
return clahe.apply(img_gray)

def normalize_image(image):
return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def sharpen_image(image):
kernel = np.array([[0, -1, 0],
[-1, 5, -1],
[0, -1, 0]])
return cv2.filter2D(image, -1, kernel)

def preprocess_image(image):
resized = resize_image(image)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
clahe = apply_clahe(gray)
normalized = normalize_image(clahe)
sharpened = sharpen_image(normalized)
return sharpened
