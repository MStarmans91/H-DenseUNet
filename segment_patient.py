#import sys
#sys.path.insert(0,'Keras-2.0.8')
from keras import backend as K
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure
from keras.optimizers import SGD
from hybridnet import dense_rnn_net
from loss import weighted_crossentropy
import glob


def padandcrop(image, sz):
    szi = image.GetSize()
    use3d = len(szi) == 3
    if szi[0] < sz[0]:
        # Pad
        image = sitk.ConstantPad(image, [sz[0] - szi[0], 0, 0])
    elif szi[0] > sz[0]:
        # Crop
        image = sitk.Crop(image, [szi[0] - sz[0], 0, 0])

    if szi[1] < sz[1]:
        # Pad
        image = sitk.ConstantPad(image, [0, sz[1] - szi[1], 0])
    elif szi[1] > sz[1]:
        # Crop
        image = sitk.Crop(image, [0, szi[1] - sz[1], 0])

    if use3d:
        if szi[2] < sz[2]:
            # Pad
            image = sitk.ConstantPad(image, [0, 0, sz[2] - szi[2]])
        elif szi[1] > sz[1]:
            # Crop
            image = sitk.Crop(image, [0, 0, szi[2] - sz[2]])

    return image


def preprocessing(image_array, input_size=512):
    '''
    Apply preprocessing to an image before segmentation
    '''
    image_array[image_array < -200] = -200
    image_array[image_array > 250] = 250
    image_array = np.array(image_array, dtype='float32')
    return image_array


def predict_liver_and_tumor(model, image, batch, input_size, input_cols,
                            thresh_liver=0.5, thresh_tumor=0.9, num=3):
    '''
    Prediction of segmentation of both liver and tumor
    '''
    print('Prediction segmentations...')
    window_cols = (input_cols/4)
    count = 0
    box_test = np.zeros((batch, input_size, input_size, input_cols, 1),
                        dtype="float32")

    x = image.shape[0]
    y = image.shape[1]
    z = image.shape[2]
    right_cols = int(min(z, y+10)-input_cols)
    left_cols = max(0, min(0-5, right_cols))
    score = np.zeros((x, y, z, num), dtype='float32')
    score_num = np.zeros((x, y, z, num), dtype='int16')

    iterator = xrange(left_cols, right_cols+window_cols, window_cols)
    cnum = 0
    for cols in iterator:
        cnum += 1
        # print ('and', z-input_cols,z)
        print(('\t Running column {} / {}.').format(cnum, len(iterator)))
        if cols > z - input_cols:
            patch_test = image[0:input_size, 0:input_size, z-input_cols:z]
            box_test[count, :, :, :, 0] = patch_test
            # print ('final', input_cols-window_cols, input_cols)
            patch_test_mask = model.predict(box_test, batch_size=batch,
                                            verbose=0)
            patch_test_mask = K.softmax(patch_test_mask)
            patch_test_mask = K.eval(patch_test_mask)
            patch_test_mask = patch_test_mask[:, :, :, 1:-1, :]

            for i in xrange(batch):
                score[0:input_size, 0:input_size, z-input_cols+1:z-1, :] += patch_test_mask[i]
                score_num[0:input_size, 0:input_size,  z-input_cols+1:z-1, :] += 1
        else:
            patch_test = image[0:input_size, 0:input_size, cols:cols + input_cols]
            box_test[count, :, :, :, 0] = patch_test
            patch_test_mask = model.predict(box_test, batch_size=batch,
                                            verbose=0)
            patch_test_mask = K.softmax(patch_test_mask)
            patch_test_mask = K.eval(patch_test_mask)
            patch_test_mask = patch_test_mask[:, :, :, 1:-1, :]
            for i in xrange(batch):
                score[0:input_size, 0:input_size, cols+1:cols+input_cols-1, :] += patch_test_mask[i]
                score_num[0:input_size, 0:input_size, cols+1:cols+input_cols-1, :] += 1

    score = score/(score_num + 1e-4)
    result1 = score[:, :, :, num-2]
    result2 = score[:, :, :, num-1]

    K.clear_session()

    print('\t Applying postprocessing.')
    result1[result1 >= thresh_liver] = 1
    result1[result1 < thresh_liver] = 0
    result2[result2 >= thresh_tumor] = 1
    result2[result2 < thresh_tumor] = 0
    result1[result2 == 1] = 1

    print('-' * 30)
    print('Postprocessing on mask ...' + str(id))
    print('-' * 30)

    #  preserve the largest liver
    lesions = result2
    box = list()
    [liver_res, num] = measure.label(result1, return_num=True)
    region = measure.regionprops(liver_res)
    for i in xrange(num):
        box.append(region[i].area)

    label_num = box.index(max(box)) + 1
    liver_res[liver_res != label_num] = 0
    liver_res[liver_res == label_num] = 1

    #  preserve the largest liver
    liver_res = ndimage.binary_dilation(liver_res, iterations=1).astype(liver_res.dtype)
    box = []
    [liver_labels, num] = measure.label(liver_res, return_num=True)
    region = measure.regionprops(liver_labels)
    for i in xrange(num):
        box.append(region[i].area)
    label_num = box.index(max(box)) + 1
    liver_labels[liver_labels != label_num] = 0
    liver_labels[liver_labels == label_num] = 1
    liver_labels = ndimage.binary_fill_holes(liver_labels).astype(int)

    #  preserve tumor within ' largest liver' only
    lesions = lesions * liver_labels
    lesions = ndimage.binary_fill_holes(lesions).astype(int)
    liver_res = np.array(liver_res, dtype='uint8')
    liver_res = ndimage.binary_fill_holes(liver_res).astype(int)
    # liver_res[lesions == 1] = 2

    # transpose back and convert to float 64 for uint8 saving
    lesions = np.transpose(lesions, [2, 1, 0])
    liver_res = np.transpose(liver_res, [2, 1, 0])
    lesions = np.array(lesions, dtype='uint8')
    liver_res = np.array(liver_res, dtype='uint8')

    return liver_res, lesions


def segment_patient(image_file_in, model_weights, liver_file_out, lesion_file_out,
                    mean=48, batch=1, input_cols=8):
    '''
    From an input image:
    1. Apply preprocessing
    2. Segment the liver
    3. Segment the lesion
    '''
    # Input size as defined by the pretrained network from the paper
    input_size = 512

    if not os.path.exists(image_file_in):
        raise KeyError(('Image {} does not exist!').format(image_file_in))

    # Read mage and convert to array
    image = sitk.ReadImage(image_file_in)
    image = sitk.GetArrayFromImage(image)
    image = np.transpose(image, [2, 1, 0])

    # Apply preprocessing
    image = preprocessing(image, input_size)
    image -= mean

    # Load the H-DenseUnet model
    model = dense_rnn_net(batch, input_size, input_cols)
    model.load_weights(model_weights)
    sgd = SGD(lr=1e-2, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[weighted_crossentropy])

    # Predict the tumor segmentation
    print('Predicting masks on test data...' + str(id))
    liver, lesions = predict_liver_and_tumor(model, image, batch=batch,
                                             input_size=input_size,
                                             input_cols=input_cols)

    # Save the output
    sitk.WriteImage(sitk.GetImageFromArray(liver), liver_file_out)
    sitk.WriteImage(sitk.GetImageFromArray(lesions), lesion_file_out)


def main():
    images = glob.glob('/scratch-shared/mstar/Data/CLM/*/*/*/image.nii.gz')
    images.sort()
    model_weights = '/scratch-shared/mstar/Data/CLM/model_best.hdf5'
    for imnum, image in enumerate(images):
        print(('Proccessing image: {} ( {} / {}).').format(image, imnum, len(images)))
        liver_file_out = os.path.join(os.path.dirname(image), 'liver.nii.gz')
        lesion_file_out = os.path.join(os.path.dirname(image), 'lesions.nii.gz')
        segment_patient(image_file_in=image,
                        model_weights=model_weights,
                        liver_file_out=liver_file_out,
                        lesion_file_out=lesion_file_out)


if __name__ == "__main__":
    main()
