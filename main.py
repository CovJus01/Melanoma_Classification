import numpy as np
import tensorflow as tf
import matplotlib.pyplot
from PIL import Image
import os

image_dimension = (1022,767)
features = image_dimension[0]*image_dimension[1]
X_train = []
Y_train = []

class_indicator = 0
for path in os.listdir("./skin-lesions/train"):
    print(f"||||||||||||||||||||||||||||Processing: {path}|||||||||||||||||||||||||||||||||||||")
    for filename in os.listdir(f"./skin-lesions/train/{path}"):
        print(f"Processing {filename}")
        im =  Image.open(f"./skin-lesions/train/{path}/{filename}").convert("L")
        resized_im = im.resize(image_dimension)
        X_train.append(list(resized_im.getdata()))
        Y_train.append(class_indicator)
        im.close()
    class_indicator+=1


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = np.reshape(X_train, (-1, features))
Y_train = np.reshape(Y_train, (-1, 1))

print(f"X shape: {X_train.shape}")
print(f"Y shape: {Y_train.shape}")