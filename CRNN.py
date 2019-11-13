import os
import cv2
import numpy as np
import string
import time
from SamplePreprocessor import preprocessor
from Model_CRNN import model_crnn
import warnings

with warnings.catch_warnings():
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint

    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.activations import relu, sigmoid, softmax
    from keras.utils import to_categorical
    import tensorflow as tf
    from keras import backend as K
    from keras.utils.data_utils import get_file
    from keras.preprocessing import image
    import keras.callbacks
    from tensorflow.python.client import device_lib

# Check all available devices if GPU is available
print(device_lib.list_local_devices())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

char_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt '
fnTrain = 'train.txt'
fnTest = 'valid.txt'


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except ValueError:
            print(char)

    return dig_lst


# lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []

# lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []
max_label_len = 56

i = 1
flag = 0

train_paths = open(fnTrain, 'r').read().split('\n')[:-1]
test_paths = open(fnTest, 'r').read().split('\n')[:-1]

for i1 in train_paths:
    gtText1 = i1.split('_')[-1].split('.')[0]
    training_txt.append(encode_to_labels(gtText1))
    train_label_length.append(len(gtText1))
    img = preprocessor(i1, (256, 64), rotation=True)
    training_img.append(img)
    train_input_length.append(max_label_len)

for i2 in test_paths:
    gtText2 = i2.split('_')[-1].split('.')[0]
    valid_orig_txt.append(gtText2)
    valid_txt.append(encode_to_labels(gtText2))
    valid_label_length.append(len(gtText2))
    img = preprocessor(i2, (256, 64), rotation=True)
    valid_img.append(img)
    valid_input_length.append(max_label_len)

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value=len(char_list))

inputs, model_infer, outputs = model_crnn()

labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels_, input_length_, label_length_ = args
    return K.ctc_batch_cost(labels_, y_pred, input_length_, label_length_)


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

# model to be used at training time
model_train = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

model_train.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

filepath = "best_model.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

batch_size = 16
epochs = 10
model_train.load_weights('best_model.hdf5')
model_train.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length],
                y=np.zeros(len(training_img)),
                batch_size=batch_size, epochs=epochs, validation_data=(
        [valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose=1,
                callbacks=callbacks_list)

# load the saved best model weights
model_infer.load_weights('best_model.hdf5')

# predict outputs on validation images
prediction = model_infer.predict(valid_img[:])

# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                               greedy=True)[0][0])

# see the results
i = 0
for x in out:
    print("original_text =  ", valid_orig_txt[i])
    print("predicted text = ", end='')
    for p in x:
        if int(p) != -1:
            print(char_list[int(p)], end='')
    print('\n')
    i += 1
