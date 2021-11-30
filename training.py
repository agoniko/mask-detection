import pandas as pd
from numpy import asarray
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import autokeras as ak
from tqdm import tqdm
from PIL import Image
import glob
from sklearn.utils import shuffle


def create_df():
    """
    Creates a df (with shuffled entries)
    with two columns: file name and label (0: normal or 1: pneumonia)
    """
    path = 'dataset'
    files1 = glob.glob(path+"/with_mask/*.jpeg")
    files1.extend(glob.glob(path+"/with_mask/*.jpg"))
    files2 = glob.glob(path+"/without_mask/*.jpeg")
    files2.extend(glob.glob(path + "/without_mask/*.jpg"))
    # creating the train set
    df_n = pd.DataFrame()
    df_p = pd.DataFrame()
    df_n['name'] = [x for x in files1]
    df_n['outcome'] = 0.
    df_p['name'] = [x for x in files2]
    df_p['outcome'] = 1.
    df = pd.concat([df_n, df_p], axis=0, ignore_index=True)
    df = shuffle(df)  # shuffle the dataset

    return df

def create_x_and_y(train_df):
    X = np.array([img_preprocess(p) for p in train_df.name.values])
    y = train_df.outcome.values

    return X, y


def img_preprocess(img):
    """
    Opens the image and does some preprocessing
    such as converting to RGB, resize and converting to array
    """
    img = Image.open(img)
    img = img.convert('RGB')
    img = img.resize((256, 256))
    img = asarray(img) / 255
    return img

def get_model():
   X,y = create_x_and_y(create_df())
   print(len(X))
   print(len(y))
   model = ak.ImageClassifier(max_trials = 50, metrics='accuracy')
   model.fit(X, y, epochs=3, validation_split=0.2)
   model = model.export_model()
   model.save('model.h5')

