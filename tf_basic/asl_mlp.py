import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt

from mnist import dir_name, y_train, x_train, x_valid
from tb import make_TensorBoard

dir_name = 'log_asl'
TensorB = make_TensorBoard(dir_name)

train_df = pd.read_csv('data/asl_data/sign_mnist_train.csv')
valid_df = pd.read_csv('data/asl_data/sign_mnist_valid.csv')


print(train_df.head())

y_train = train_df['label']
y_valid = valid_df['label']

del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values

# 정규화 x_train, x_valid : 0에서 1사이로

# y에 대한 one hot vector 만들기
num_classes = 24

# 모델링 (mnist 필기 숫자 인식과 동일
model = Sequential()

# Hidden 1 : 512,input 784
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))

# Hidden layer 1
model.add(Dense(512, activation='relu'))

# Hidden 2 : 512
model.add(Dense(num_classes, activation='softmax'))

# Output
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])