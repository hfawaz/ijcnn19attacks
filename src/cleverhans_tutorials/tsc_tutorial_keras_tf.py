from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sklearn
import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
import keras
from keras import backend
import pandas as pd

from cleverhans_copy.attacks import FastGradientMethod
from cleverhans_copy.attacks import BasicIterativeMethod
from cleverhans_copy.utils import AccuracyReport
from cleverhans_copy.utils_keras import KerasModelWrapper
from cleverhans_copy.utils_tf import model_eval

from sklearn.preprocessing import LabelEncoder

FLAGS = flags.FLAGS

BATCH_SIZE = 1024

def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}

    file_name = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
    x_train, y_train = readucr(file_name + '_TRAIN')
    x_test, y_test = readucr(file_name + '_TEST')
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())

    return datasets_dict

def transform_labels(y_train,y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train,y_test),axis =0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test

def prepare_data(datasets_dict,dataset_name):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test,y_true, nb_classes

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        os.makedirs(directory_path)
    return directory_path

def add_labels_to_adv_test_set(dataset_dict,dataset_name, adv_data_dir,original_y):
    x_test_perturbed = np.loadtxt(adv_data_dir+dataset_name+'-adv', delimiter=',')
    test_set = np.zeros((original_y.shape[0],x_test_perturbed.shape[1]+1),dtype=np.float64)
    test_set[:,0] = original_y
    test_set[:,1:] = x_test_perturbed
    np.savetxt(adv_data_dir+dataset_name+'-adv',test_set,delimiter=',')

def tsc_tutorial(attack_method='fgsm',batch_size=BATCH_SIZE,
                 dataset_name='Adiac',eps=0.1,attack_on='train'):

    keras.layers.core.K.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    root_dir = '/b/home/uha/hfawaz-datas/dl-tsc/'

    # dataset_name = 'Adiac'
    archive_name = 'TSC'
    classifier_name = 'resnet'
    out_dir = 'ucr-attack/'
    file_path = root_dir + 'results/' + classifier_name + '/' + archive_name +\
                '/' + dataset_name + '/best_model.hdf5'

    adv_data_dir = out_dir+attack_method+'/'+archive_name+'/'+attack_on+\
                   '/eps-'+str(eps)+'/'

    if os.path.exists(adv_data_dir+dataset_name+'-adv'):
        print('Already_done:',dataset_name)
        return
    else:
        print('Doing:',dataset_name)

    dataset_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train, y_train, x_test, y_test, _, nb_classes = prepare_data(dataset_dict,dataset_name)

    if attack_on == 'train':
        X = x_train
        Y = y_train
        original_y = dataset_dict[dataset_name][1]
    elif attack_on =='test':
        X = x_test
        Y = y_test
        original_y = dataset_dict[dataset_name][3]
    else:
        print('Error either train or test options for attack_on param')
        exit()

    # for big datasets we should decompose in batches the evaluation of the attack
    # loop through the batches
    ori_acc = 0
    adv_acc = 0

    res_dir = out_dir + 'results'+attack_method+'.csv'
    if os.path.exists(res_dir):
        res_ori = pd.read_csv(res_dir, index_col=False)
    else:
        res_ori = pd.DataFrame(data=np.zeros((0, 3), dtype=np.float), index=[],
                               columns=['dataset_name', 'ori_acc', 'adv_acc'])

    test_set = np.zeros((Y.shape[0], x_train.shape[1] + 1), dtype=np.float64)

    for i in range(0,len(X),batch_size):
        curr_X = X[i:i+batch_size]
        curr_Y = Y[i:i+batch_size]

        # Obtain series Parameters
        img_rows, nchannels = x_train.shape[1:3]

        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=(None, img_rows, nchannels))
        y = tf.placeholder(tf.float32, shape=(None, nb_classes))

        # Define TF model graph
        model = keras.models.load_model(file_path)
        preds = model(x)
        print("Defined TensorFlow model graph.")

        def evaluate():
            # Evaluate the accuracy of the model on legitimate test examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds, curr_X, curr_Y, args=eval_params)
            report.clean_train_clean_eval = acc
            print('Test accuracy on legitimate examples: %0.4f' % acc)
            return acc

        wrap = KerasModelWrapper(model)

        ori_acc += evaluate() * len(curr_X)/len(X)

        if attack_method == 'fgsm':
            # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {'eps': eps }
            adv_x = fgsm.generate(x, **fgsm_params)
        elif attack_method == 'bim':
            # BasicIterativeMethod
            bim = BasicIterativeMethod(wrap,sess=sess)
            bim_params = {'eps':eps, 'eps_iter':0.05, 'nb_iter':10}
            adv_x = bim.generate(x,**bim_params)
        else:
            print('Either bim or fgsm are acceptable as attack methods')
            return

        # Consider the attack to be constant
        adv_x = tf.stop_gradient(adv_x)

        adv = adv_x.eval({x: curr_X}, session=sess)
        adv = adv.reshape(adv.shape[0],adv.shape[1])

        preds_adv = model(adv_x)

        # Evaluate the accuracy of the model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, curr_X, curr_Y, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc
        adv_acc += acc * len(curr_X)/len(X)

        test_set[i:i+batch_size,0] = original_y[i:i+batch_size]
        test_set[i:i+batch_size,1:] = adv


    create_directory(adv_data_dir)

    np.savetxt(adv_data_dir+dataset_name+'-adv',test_set, delimiter=',')

    add_labels_to_adv_test_set(dataset_dict, dataset_name, adv_data_dir,original_y)

    res = pd.DataFrame(data = np.zeros((1,3),dtype=np.float), index=[0],
            columns=['dataset_name','ori_acc','adv_acc'])
    res['dataset_name'] = dataset_name+str(eps)
    res['ori_acc'] = ori_acc
    res['adv_acc'] = adv_acc
    res_ori = pd.concat((res_ori,res),sort=False)
    res_ori.to_csv(res_dir,index=False)

    return report

def main(argv=None,attack_method='fgsm'):
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')

    # full datasets
    dataset_names = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
                     'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
                     'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
                     'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                     'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
                     'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
                     'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
                     'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
                     'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain',
                     'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
                     'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                     'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
                     'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                     'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Two_Patterns',
                     'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                     'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga']

    # dataset_names = ['Coffee']

    # epsilons = np.arange(start=0.0,stop=2.0,step=0.025,dtype=np.float32)
    epsilons = [0.1]

    for dataset_name in dataset_names:
        for ep in epsilons:

            tsc_tutorial(attack_method=attack_method,
                batch_size=FLAGS.batch_size,
                           dataset_name=dataset_name,
                           eps = ep)

# if __name__ == '__main__':
#     flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
#                          'Number of epochs to train model')
#     flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
#     flags.DEFINE_float('learning_rate', LEARNING_RATE,
#                        'Learning rate for training')
#     flags.DEFINE_string('train_dir', TRAIN_DIR,
#                         'Directory where to save model.')
#     flags.DEFINE_string('filename', FILENAME, 'Checkpoint filename.')
#     flags.DEFINE_boolean('load_model', LOAD_MODEL,
#                          'Load saved model or train.')
#     tf.app.run()
