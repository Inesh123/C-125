import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
X,Y = fetch_openml('mnist_784',version = 1,return_X_Y = True)
X_train,X_test = Y_train,Y_test = train_test_split(X_Y,random_state = 9,train_size = 7500,test_size = 2500)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(X_train_scaled,Y_train)
def get_prediction(image):
    im_PIL = Image.open(image)
    Image_bw = im_PIL.convert('L')
    Image_bw_resized = Image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(Image_bw_resized,pixel_filter)
    Image_bw_resized_inverted_scaled = np.clip(Image_bw_resized - min_pixel,0,255)
    max_pixel = np.max(Image_bw_resized)
    Image_bw_resized_inverted_scaled = np.asarray(Image_bw_resized_inverted_scaled) / max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]