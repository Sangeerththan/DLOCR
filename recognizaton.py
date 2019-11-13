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
fnTest = 'valid.txt'

test_paths = open(fnTest, 'r').read().split('\n')[:-1]

inputs, model_infer, outputs = model_crnn()

batch_size = 16
img_size = (256, 64)
test_generator = image_generator_test(files=test_paths, batch_size=batch_size, imgSize=img_size)

steps_in_epoch_valid = len(test_paths) // batch_size

# load the saved best model weights
model_infer.load_weights('saved-model-0005-27.01.hdf5')

# predict outputs on validation images
prediction = model_infer.predict_generator(test_generator, steps=100)

# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                               greedy=True)[0][0])

# see the results
i = 0
for imdx, x in enumerate(out):
    actual = test_paths[imdx].split('/')[-1].split('.')[0]
    recog = ''
    for p in x:
        if int(p) != -1:
            recog = recog + char_list[int(p)]
    prediction = recog

    print(actual + ' > ' + prediction)
