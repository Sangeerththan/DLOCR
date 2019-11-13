from Model_CRNN import model_crnn
import warnings
from keras_image_generater_training import image_generator, image_generator_test
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.layers import Input, Lambda
    from keras.models import Model
    from keras.optimizers import Adam
    from keras import backend as K
    from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard


def ctc_lambda_func(args):
    y_pred, labels_, input_length_, label_length_ = args
    return K.ctc_batch_cost(labels_, y_pred, input_length_, label_length_)


char_list = 'අආඇඈඉඊඋඌඍඎඑඒඓඔඕඖකකාකැකෑකිකීකුකූක්‍ර්‍ර්‍ර්‍ර්කෙකේකොකෝඛඛාඛැඛෑඛිඛීඛුඛූඛෙඛේඛොඛෝගගාගැගෑගිගීගුගූගෙගේගොගෝඝඝාඝැඝෑඝිඝීඝුඝූඝෙඝේඝොඝෝචචාචැචෑචිචීචුචූචෙචේචොචෝඡඡාඡැඡෑඡිඡීඡුඡූඡෙඡේඡොඡෝජජාජැජෑජිජීජුජූජෙජේජොජෝඣඣාඣැඣෑඣිඣීඣුඣූඣෙඣේඣොඣෝටටාටැටෑටිටීටුටූටෙටේටොටෝඨඨාඨැඨෑඨිඨීඨුඨූඨෙඨේඨොඨෝඩඩාඩැඩෑඩිඩීඩුඩූඩෙඩේඩොඩෝඪඪාඪැඪෑඪිඪීඪුඪූඪෙඪේඪොඪෝතතාතැතෑතිතීතුතූතෙතේතොතෝථථාථැථෑථිථීථුථූථෙථේථොථෝදදාදැදෑදිදීදුදූදෙදේදොදෝධධාධැධෑධිධීධුධූධෙධේධොධෝපපාපැපෑපිපීපුපූපෙපේපොපෝඵඵාඵැඵෑඵිඵීඵුඵූඵෙඵේඵොඵෝබබාබැබෑබිබීබුබූබෙබේබොබෝභභාභැභෑභිභීභුභූභෙභේභොභෝයයායැයෑයියීයුයූයෙයේයොයෝරරාරැරෑරිරීරුරූරෙරේරොරෝලලාලැලෑලිලීලුලූලෙලේලොලෝවවාවැවෑවිවීවුවූවෙවේවොවෝශශාශැශෑශිශීශුශූශෙශේශොශෝෂෂාෂැෂෑෂිෂීෂුෂූෂෙෂේෂොෂෝසසාසැසෑසිසීසුසූසෙසේසොසෝහහාහැහෑහිහීහුහූහෙහේහොහෝෆෆාෆැෆෑෆිෆීෆුෆූෆෙෆේෆොංනමණළ'
fnTrain = 'train.txt'
fnTest = 'valid.txt'

max_label_len = 30
img_size = (256, 64)
epochs = 300
batch_size = 16

train_paths = open(fnTrain, 'r').read().split('\n')[:1000]
test_paths = open(fnTest, 'r').read().split('\n')[:1000]

inputs, model_infer, outputs = model_crnn()

labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

# model to be used at training time
model_train = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

adam = Adam(lr=0.001)
model_train.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam,metrics=[])

# reduces learning rate if no improvement are seen
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=5,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0000001)
# stop training if no improvements are seen
early_stop = EarlyStopping(monitor="val_loss",
                           mode="min",
                           patience=16,
                           restore_best_weights=True)

# saves model weights to file
filepath = "model_best.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_weights_only=True)

tensorboard = TensorBoard(log_dir='logs',
                          histogram_freq=0,
                          batch_size=batch_size,
                          write_graph=True,
                          write_grads=True,
                          write_images=False, update_freq='batch')

callbacks_list = [learning_rate_reduction, early_stop, tensorboard, checkpoint]

# model_train.load_weights('best_model.hdf5')

train_generator = image_generator(files=train_paths, batch_size=batch_size, max_label_length=max_label_len,
                                  imgSize=img_size)
validation_generator = image_generator(files=test_paths, batch_size=batch_size, max_label_length=max_label_len,
                                       imgSize=img_size)
test_generator = image_generator_test(files=train_paths, batch_size=batch_size, imgSize=img_size)

steps_in_epoch_train = len(train_paths) // batch_size
steps_in_epoch_valid = len(test_paths) // batch_size

model_train.fit_generator(train_generator, steps_per_epoch=steps_in_epoch_train, epochs=epochs, verbose=1,
                          validation_data=validation_generator, callbacks=callbacks_list,
                          validation_steps=steps_in_epoch_valid, workers=-1, use_multiprocessing=True)

