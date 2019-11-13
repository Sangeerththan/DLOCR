import os
from sklearn.model_selection import train_test_split
import numpy as np

own_crop_name_folder = '../OutputTest'


own_img_list = np.array([os.path.join(own_crop_name_folder, file) for file in sorted(os.listdir(own_crop_name_folder))])
own_img_train, own_img_test = train_test_split(own_img_list, test_size=0.5, shuffle=False)

np.random.shuffle(own_img_train)

with open('train.txt', 'w') as f:
    for i1 in own_img_train:
        f.write('%s\n' % i1)

with open('valid.txt', 'w') as f:
    for i2 in own_img_test :
        f.write('%s\n' % i2)
