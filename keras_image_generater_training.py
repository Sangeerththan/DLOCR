import numpy as np
from SamplePreprocessor import preprocessor
import warnings

with warnings.catch_warnings():
    from keras.preprocessing.sequence import pad_sequences

char_list = 'අආඇඈඉඊඋඌඍඎඑඒඓඔඕඖකකාකැකෑකිකීකුකූක්‍ර්‍ර්‍ර්‍ර්කෙකේකොකෝඛඛාඛැඛෑඛිඛීඛුඛූඛෙඛේඛොඛෝගගාගැගෑගිගීගුගූගෙගේගොගෝඝඝාඝැඝෑඝිඝීඝුඝූඝෙඝේඝොඝෝචචාචැචෑචිචීචුචූචෙචේචොචෝඡඡාඡැඡෑඡිඡීඡුඡූඡෙඡේඡොඡෝජජාජැජෑජිජීජුජූජෙජේජොජෝඣඣාඣැඣෑඣිඣීඣුඣූඣෙඣේඣොඣෝටටාටැටෑටිටීටුටූටෙටේටොටෝඨඨාඨැඨෑඨිඨීඨුඨූඨෙඨේඨොඨෝඩඩාඩැඩෑඩිඩීඩුඩූඩෙඩේඩොඩෝඪඪාඪැඪෑඪිඪීඪුඪූඪෙඪේඪොඪෝතතාතැතෑතිතීතුතූතෙතේතොතෝථථාථැථෑථිථීථුථූථෙථේථොථෝදදාදැදෑදිදීදුදූදෙදේදොදෝධධාධැධෑධිධීධුධූධෙධේධොධෝපපාපැපෑපිපීපුපූපෙපේපොපෝඵඵාඵැඵෑඵිඵීඵුඵූඵෙඵේඵොඵෝබබාබැබෑබිබීබුබූබෙබේබොබෝභභාභැභෑභිභීභුභූභෙභේභොභෝයයායැයෑයියීයුයූයෙයේයොයෝරරාරැරෑරිරීරුරූරෙරේරොරෝලලාලැලෑලිලීලුලූලෙලේලොලෝවවාවැවෑවිවීවුවූවෙවේවොවෝශශාශැශෑශිශීශුශූශෙශේශොශෝෂෂාෂැෂෑෂිෂීෂුෂූෂෙෂේෂොෂෝසසාසැසෑසිසීසුසූසෙසේසොසෝහහාහැහෑහිහීහුහූහෙහේහොහෝෆෆාෆැෆෑෆිෆීෆුෆූෆෙෆේෆො'


def encode_to_labels(txt):
    global char_list
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except ValueError:
            char_list = char_list + char
            with open('charList.txt','w') as f:
                f.write(char_list)
    return dig_lst


def image_generator(files, max_label_length, batch_size=16, imgSize=(256, 32)):
    index_ = 0
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = files[index_ * batch_size:(index_ + 1) * batch_size]

        training_img = []
        training_txt = []
        train_label_length = []
        train_input_length = []
        max_label_len = max_label_length
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            gtText1 = input_path.split('/')[-1].split('.')[0]
            # print(gtText1)
            encode_to_labels(gtText1)
            training_txt.append(encode_to_labels(gtText1))
            train_label_length.append(len(gtText1))

            img = preprocessor(filePath=input_path, imgSize=imgSize, rotation=True)
            training_img.append(img)
            train_input_length.append(max_label_len)

        train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
        train_input_length = np.array(train_input_length)
        train_label_length = np.array(train_label_length)
        training_img = np.array(training_img)
        output = np.zeros(len(training_img))

        input_ = [training_img, train_padded_txt, train_input_length, train_label_length]
        index_ = index_ + 1
        if index_ >= len(files) // batch_size:
            index_ = 0
        yield (input_, output)


def image_generator_test(files, imgSize=(256, 32), batch_size=16):
    index_ = 0
    while True:
        # Select files (paths/indices) for the batch
        batch_paths = files[index_ * batch_size:(index_ + 1) * batch_size]

        training_img = []

        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            img = preprocessor(filePath=input_path, imgSize=imgSize, rotation=True)
            training_img.append(img)

        training_img = np.array(training_img)

        index_ = index_ + 1
        if index_ >= len(files) // batch_size:
            index_ = 0

        yield training_img
