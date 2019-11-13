import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

    from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPooling2D, add, Dropout, \
        concatenate
    from keras.layers import ReLU
    from keras.models import Model


def model_crnn():
    conv_chanel = [64, 128, 256, 256, 512, 512, 512]
    lstm_units = 256
    img_width = 256
    img_height = 64
    dropout_rate = 0.25
    char_list = 'අආඇඈඉඊඋඌඍඎඑඒඓඔඕඖකකාකැකෑකිකීකුකූක්‍ර්‍ර්‍ර්‍ර්කෙකේකොකෝඛඛාඛැඛෑඛිඛීඛුඛූඛෙඛේඛොඛෝගගාගැගෑගිගීගුගූගෙගේගොගෝඝඝාඝැඝෑඝිඝීඝුඝූඝෙඝේඝොඝෝචචාචැචෑචිචීචුචූචෙචේචොචෝඡඡාඡැඡෑඡිඡීඡුඡූඡෙඡේඡොඡෝජජාජැජෑජිජීජුජූජෙජේජොජෝඣඣාඣැඣෑඣිඣීඣුඣූඣෙඣේඣොඣෝටටාටැටෑටිටීටුටූටෙටේටොටෝඨඨාඨැඨෑඨිඨීඨුඨූඨෙඨේඨොඨෝඩඩාඩැඩෑඩිඩීඩුඩූඩෙඩේඩොඩෝඪඪාඪැඪෑඪිඪීඪුඪූඪෙඪේඪොඪෝතතාතැතෑතිතීතුතූතෙතේතොතෝථථාථැථෑථිථීථුථූථෙථේථොථෝදදාදැදෑදිදීදුදූදෙදේදොදෝධධාධැධෑධිධීධුධූධෙධේධොධෝපපාපැපෑපිපීපුපූපෙපේපොපෝඵඵාඵැඵෑඵිඵීඵුඵූඵෙඵේඵොඵෝබබාබැබෑබිබීබුබූබෙබේබොබෝභභාභැභෑභිභීභුභූභෙභේභොභෝයයායැයෑයියීයුයූයෙයේයොයෝරරාරැරෑරිරීරුරූරෙරේරොරෝලලාලැලෑලිලීලුලූලෙලේලොලෝවවාවැවෑවිවීවුවූවෙවේවොවෝශශාශැශෑශිශීශුශූශෙශේශොශෝෂෂාෂැෂෑෂිෂීෂුෂූෂෙෂේෂොෂෝසසාසැසෑසිසීසුසූසෙසේසොසෝහහාහැහෑහිහීහුහූහෙහේහොහෝෆෆාෆැෆෑෆිෆීෆුෆූෆෙෆේෆොංනමණළ'

    inputs = Input((img_width, img_height, 1))
    conv_1 = Conv2D(conv_chanel[0], (3, 3), activation='relu', padding='same', name='conv_1')(inputs)
    conv_2 = Conv2D(conv_chanel[1], (3, 3), activation='relu', padding='same', name='conv_2')(conv_1)
    conv_3 = Conv2D(conv_chanel[2], (3, 3), use_bias=False, padding='same', name='conv_3')(conv_2)
    batch_normaization_3 = BatchNormalization(name='batch_normaization_3')(conv_3)
    relu_3 = ReLU(name='Relu_3')(batch_normaization_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(relu_3)

    conv_4 = Conv2D(conv_chanel[3], (3, 3), activation='relu', padding='same', name='conv_4')(pool_3)
    conv_5 = Conv2D(conv_chanel[4], (3, 5), use_bias=False, padding='same', name='conv_5')(conv_4)
    batch_normaization_5 = BatchNormalization(name='batch_normaization_5')(conv_5)
    relu_5 = ReLU(name='Relu_5')(batch_normaization_5)
    pool_5 = MaxPooling2D(pool_size=(2, 2), name='maxpool_5')(relu_5)

    conv_6 = Conv2D(conv_chanel[5], (3, 3), activation='relu', padding='same', name='conv_6')(pool_5)
    conv_7 = Conv2D(conv_chanel[6], (3, 3), padding='same', name='conv_7', use_bias=False)(conv_6)
    batch_normaization_7 = BatchNormalization(name='batch_normaization_7')(conv_7)
    relu_7 = ReLU(name='Relu_7')(batch_normaization_7)
    relu_7_shape = relu_7.get_shape()

    reshape = Reshape(target_shape=(
        int(relu_7_shape[1]), int(relu_7_shape[2] * relu_7_shape[3])),
        name='reshape')(relu_7)

    fc_9 = Dense(lstm_units, activation='relu', name='fc_9')(reshape)

    lstm_10 = LSTM(lstm_units, kernel_initializer="he_normal", return_sequences=True, name='lstm_10')(fc_9)
    lstm_10_back = LSTM(lstm_units, kernel_initializer="he_normal", go_backwards=True, return_sequences=True,
                        name='lstm_10_back')(fc_9)
    lstm_10_add = add([lstm_10, lstm_10_back])

    lstm_11 = LSTM(lstm_units, kernel_initializer="he_normal", return_sequences=True, name='lstm_11')(lstm_10_add)
    lstm_11_back = LSTM(lstm_units, kernel_initializer="he_normal", go_backwards=True, return_sequences=True,
                        name='lstm_11_back')(lstm_10_add)
    lstm_11_concat = concatenate([lstm_11, lstm_11_back])
    do_11 = Dropout(dropout_rate, name='dropout')(lstm_11_concat)

    outputs = Dense(len(char_list) + 1, kernel_initializer='he_normal', activation='softmax', name='fc_12')(do_11)

    model = Model(inputs=inputs, outputs=outputs)
    return inputs, model, outputs

#
# def _conv_block(inp, convs, skip=True):
#     x = inp
#     for conv in convs:
#         x = Conv2D(conv['filter'],
#                    conv['kernel'],
#                    strides=conv['stride'],
#                    padding='same',
#                    name='conv_' + str(conv['layer_idx']),
#                    use_bias=True if conv['bnorm'] else True)(x)
#         if conv['bnorm']:
#             x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
#         if conv['relu']:
#             x = ReLU(name='relu_' + str(conv['layer_idx']))(x)
#     return x

#
# def model_crnn():
#     lstm_units = 128
#     img_width = 256
#     img_height = 32
#     dropout_rate = 0
#     char_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt '
#
#     inputs = Input((img_width, img_height, 1))
#
#     x = _conv_block(inputs,
#                     [{'filter': 16, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 0},
#                      {'filter': 16, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 1},
#                      {'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True,
#                       'layer_idx': 2}])
#
#     x = MaxPooling2D(pool_size=(2, 1), name='maxpool_1')(x)
#
#     x = _conv_block(x,
#                     [{'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 3},
#                      {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 4},
#                      {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True,
#                       'layer_idx': 5}])
#
#     x = MaxPooling2D(pool_size=(2, 1), name='maxpool_2')(x)
#
#     x = _conv_block(x,
#                     [{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 6},
#                      {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 7},
#                      {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True,
#                       'layer_idx': 8}])
#
#     x = MaxPooling2D(pool_size=(1, 2), name='maxpool_3')(x)
#     x_shape = x.get_shape()
#     x = Reshape(target_shape=(
#         int(x_shape[1]), int(x_shape[2] * x_shape[3])),
#         name='reshape')(x)
#
#     x = Dense(lstm_units, activation='relu', name='fc_1')(x)
#     x = Dropout(dropout_rate, name='dropout_1')(x)
#
#     lstm_1 = LSTM(lstm_units, kernel_initializer="he_normal", return_sequences=True, name='lstm_1')(x)
#     lstm_1_back = LSTM(lstm_units, kernel_initializer="he_normal", go_backwards=True, return_sequences=True,
#                        name='lstm_1_back')(x)
#     x = add([lstm_1, lstm_1_back])
#     x = Dropout(dropout_rate, name='dropout_2')(x)
#
#     lstm_2 = LSTM(lstm_units, kernel_initializer="he_normal", return_sequences=True, name='lstm_2')(x)
#     lstm_2_back = LSTM(lstm_units, kernel_initializer="he_normal", go_backwards=True, return_sequences=True,
#                        name='lstm_2_back')(x)
#     x = concatenate([lstm_2, lstm_2_back])
#     x = Dropout(dropout_rate, name='dropout_3')(x)
#
#     outputs = Dense(len(char_list) + 1, kernel_initializer='he_normal', activation='softmax', name='fc_2')(x)
#
#     model = Model(inputs=inputs, outputs=outputs)
#     return inputs, model, outputs

# _, m, _ = model_crnn()
# print(m.summary())
