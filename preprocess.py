import cv2
import numpy as np

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder

from bounding_box_utils.bounding_box_utils import *
from data_aug.data_aug import *

def customDatagen(csv_path_array, csv_neg_path_array, images_path, negatives_path, batch_size,ssd_input_encoder, neg_pos_ratio=3):
    randomHflip = RandomHorizontalFlip(p=.5)
    randomRot = RandomRotate(angle=90)
    randomScale = RandomScale(scale=.15)
    randomTranslate = RandomTranslate(translate=.2)
    randomShear = RandomShear(shear_factor=.2)
    randomHSV = RandomHSV((1, 2),(1, 2),(-48, 48))
    resizer = Resize(224)

    transformation_dict = {
        "hflip": randomHflip,
        "rotation":randomRot,
        "scale":randomScale,
        # "translate": randomTranslate,
        "shear":randomShear,
        "hsv":randomHSV,
        "resizer":resizer
    }

    keys = list(transformation_dict.keys())
    keys.pop(-1)
    keys.pop(-1)

    idx = np.arange(0, len(keys), 1)
    imgs_dict = {}
    for row in csv_path_array:
        img_path = row.split(',')[0]
        if img_path not in imgs_dict:
            imgs_dict[img_path] = {'classes':[],'coords':[]}
        imgs_dict[img_path]['classes'].append(row.split(',')[1])
        imgs_dict[img_path]['coords'].append(np.asarray(row.split(',')[2:],dtype='float32'))
        
    negs_dict = {}
    for row in csv_neg_path_array:
        img_path = row.split(',')[0]
        if img_path not in negs_dict:
            negs_dict[img_path] = {'classes':[],'coords':[]}
        negs_dict[img_path]['classes'].append(row.split(',')[1])
        negs_dict[img_path]['coords'].append(np.asarray(row.split(',')[2:],dtype='float32'))
    nb_negs = len(list(negs_dict.keys()))
    negs_index = 0

    while 1:
        batch_index = 0 
        for image_name in list(imgs_dict.keys()):
            if batch_index == 0:
                X_train = []
                bboxes = []
                classes = []
                encoder_output_data = []

            img = cv2.imread(os.path.join(images_path,image_name))
            bboxes_ = np.asarray(imgs_dict[image_name]['coords']).astype('Float32')
            classes_ = np.asarray(imgs_dict[image_name]['classes']).astype('Float32')

            # apply 3 random transformations
            permut = np.random.permutation(idx)
            for i in permut[:3]:
                try:
                    img, bboxes_ = transformation_dict[keys[i]](img,bboxes_)   
                except ValueError:
                    pass

            img, bboxes_ = resizer(img, bboxes_)
            img, bboxes_ = randomHSV(img, bboxes_)
            X_train.append(img)
            tmp = []
            for c, b in list(zip(classes_, bboxes_)):
                tmp_ = np.concatenate([[c], b]).tolist()
                tmp.append(tmp_)
            
            encoded_data = ssd_input_encoder(np.asarray([tmp]))
            encoder_output_data.append(encoded_data)
            batch_index += 1

            for i in range(neg_pos_ratio):
                image_name = list(negs_dict.keys())[negs_index]
                bboxes_ = np.asarray(negs_dict[image_name]['coords']).astype('Float32')
                classes_ = np.asarray(negs_dict[image_name]['classes']).astype('Float32')

                img = cv2.imread(os.path.join(negatives_path,image_name))
                img, _ = resizer(img, bboxes_)
                X_train.append(img)
                encoded_data = ssd_input_encoder(background=True)
                encoder_output_data.append(encoded_data)
                batch_index += 1
                negs_index += 1

                if negs_index == nb_negs:
                    negs_index = 0

            if batch_index % batch_size == 0:
                # for img in X_train:
                #     cv2.imshow('img',img)
                #     cv2.waitKey() 
                X_train = np.array(X_train, dtype=np.float32)/255.
                
                yield (X_train, encoder_output_data)

                X_train = []
                bboxes = []
                classes = []
                encoder_output_data = []

