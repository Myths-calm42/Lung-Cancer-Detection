import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

def extract_glcm_features(image):
features = {}
glcm = graycomatrix(image, distances=[1],
angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
levels=256, symmetric=True, normed=True)
for prop in GLCM_PROPERTIES:
features[prop] = graycoprops(glcm, prop).mean()
return features

def extract_features_from_dataset(input_folder, output_csv):
all_features = []
for image_name in os.listdir(input_folder):
if image_name.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff','.pgm')):
path = os.path.join(input_folder, image_name)
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
gray = img_as_ubyte(image)
feats = extract_glcm_features(gray)
if "malignant" in image_name.lower():
label = "malignant"
elif "bengin" in image_name.lower():
label = "bengin"
elif "normal" in image_name.lower():
label = "normal"
else:
label = "unknown"
feats["Label"] = label
all_features.append(feats)
df = pd.DataFrame(all_features)
df.to_csv(output_csv, index=False)
return df
