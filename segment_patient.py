
import os
import numpy as np
import SimpleITK as sitk
from hybridnet import dense_rnn_net
from scipy import ndimage
from keras import backend as K
from skimage import measure


def preprocessing(image_array):
    '''
    Apply preprocessing to an image before segmentation
    '''
    image_array[image_array < -200] = -200
    image_array[image_arrayage > 250] = 250
    image_array = np.array(image_array, dtype='float32')
    return image_array


def predict_liver_and_tumor(model, image, batch=1, input_size=512, img_cols=8,
                            thresh_liver=0.5, thresh_tumor=0.9):
    '''
    Prediction of segmentation of both liver and tumor
    '''

    window_cols = (img_cols/4)
    count = 0
    box_test = np.zeros((batch, input_size, input_size, img_cols, 1),
                        dtype="float32")

    x = imgs_test.shape[0]
    y = imgs_test.shape[1]
    z = imgs_test.shape[2]
    right_cols = int(min(z, maxi[2]+10)-img_cols)
    left_cols = max(0, min(mini[2]-5, right_cols))
    score = np.zeros((x, y, z, num), dtype='float32')
    score_num = np.zeros((x, y, z, num), dtype='int16')

    for cols in xrange(left_cols, right_cols+window_cols, window_cols):
        # print ('and', z-img_cols,z)
        if cols > z - img_cols:
            patch_test = imgs_test[0:input_size, 0:input_size, z-img_cols:z]
            box_test[count, :, :, :, 0] = patch_test
            # print ('final', img_cols-window_cols, img_cols)
            patch_test_mask = model.predict(box_test, batch_size=batch,
                                            verbose=0)
            patch_test_mask = K.softmax(patch_test_mask)
            patch_test_mask = K.eval(patch_test_mask)
            patch_test_mask = patch_test_mask[:, :, :, 1:-1, :]

            for i in xrange(batch):
                score[0:input_size, 0:input_size, z-img_cols+1:z-1, :] += patch_test_mask[i]
                score_num[0:input_size, 0:input_size,  z-img_cols+1:z-1, :] += 1
        else:
            patch_test = imgs_test[0:input_size, 0:input_size, cols:cols + img_cols]
            box_test[count, :, :, :, 0] = patch_test
            patch_test_mask = model.predict(box_test, batch_size=batch,
                                            verbose=0)
            patch_test_mask = K.softmax(patch_test_mask)
            patch_test_mask = K.eval(patch_test_mask)
            patch_test_mask = patch_test_mask[:, :, :, 1:-1, :]
            for i in xrange(batch):
                score[0:input_size, 0:input_size, cols+1:cols+img_cols-1, :] += patch_test_mask[i]
                score_num[0:input_size, 0:input_size, cols+1:cols+img_cols-1, :] += 1

    score = score/(score_num+1e-4)
    result1 = score[:, :, :, num-2]
    result2 = score[:, :, :, num-1]

    K.clear_session()

    result1[result1 >= thresh_liver] = 1
    result1[result1 < thresh_liver] = 0
    result2[result2 >= thresh_tumor] = 1
    result2[result2 < thresh_tumor] = 0
    result1[result2 == 1] = 1

    print('-' * 30)
    print('Postprocessing on mask ...' + str(id))
    print('-' * 30)

    #  preserve the largest liver
    Segmask = result2
    box = list()
    [liver_res, num] = measure.label(result1, return_num=True)
    region = measure.regionprops(liver_res)
    for i in xrange(num):
        box.append(region[i].area)
    label_num = box.index(max(box)) + 1
    liver_res[liver_res != label_num] = 0
    liver_res[liver_res == label_num] = 1

    #  preserve the largest liver
    mask = ndimage.binary_dilation(mask, iterations=1).astype(mask.dtype)
    box = []
    [liver_labels, num] = measure.label(mask, return_num=True)
    region = measure.regionprops(liver_labels)
    for i in xrange(num):
        box.append(region[i].area)
    label_num = box.index(max(box)) + 1
    liver_labels[liver_labels != label_num] = 0
    liver_labels[liver_labels == label_num] = 1
    liver_labels = ndimage.binary_fill_holes(liver_labels).astype(int)

    #  preserve tumor within ' largest liver' only
    Segmask = Segmask * liver_labels
    Segmask = ndimage.binary_fill_holes(Segmask).astype(int)
    Segmask = np.array(Segmask, dtype='uint8')
    liver_res = np.array(liver_res, dtype='uint8')
    liver_res = ndimage.binary_fill_holes(liver_res).astype(int)
    liver_res[Segmask == 1] = 2
    liver_res = np.array(liver_res, dtype='uint8')

    return liver, tumor


def segment_patient(image_file_in, model, liver_file_out, lesion_file_out,
                    mean=48):
    '''
    From an input image:
    1. Apply preprocessing
    2. Segment the liver
    3. Segment the lesion
    '''
    if not os.path.exists(image_file_in):
        raise KeyError(('Image {} does not exist!').format(image_file_in))

    if not os.path.exists(liver_file_in):
        raise KeyError(('Liver mask {} does not exist!').format(liver_file_in))

    # Read mage and convert to array
    image = sitk.ReadImage(image_file_in)
    image = sitk.GetArrayFromImage(image)

    # Apply preprocessing
    image = preprocessing(image)
    img_test -= mean

    # Load the H-DenseUnet model
    model = dense_rnn_net(args)
    model.load_weights(args.model_weight)
    sgd = SGD(lr=1e-2, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[weighted_crossentropy])

    # Predict the tumor segmentation
    print('Predicting masks on test data...' + str(id))
    liver, lesions = predict_liver_and_tumor(model, image)

    # Save the output
    sitk.WriteImage(liver, liver_file_out)
    sitk.WriteImage(lesions, lesion_file_out)


def main():
    image =
