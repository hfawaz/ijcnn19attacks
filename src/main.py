import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sys
import cleverhans_tutorials.tsc_tutorial_keras_tf as attack

root_dir = '/b/home/uha/hfawaz-datas/dl-tsc/'
archive_name = 'UCR_TS_Archive_2015'
root_dir_archive = root_dir+'archives/'+archive_name+'/'
root_dir_attack = 'ucr-attack/'

def zNormalize(x):
    x_mean = x.mean(axis=0) # mean for each dimension of time series x
    x_std = x.std(axis = 0) # std for each dimension of time series x
    x = (x - x_mean)/(x_std)
    return x

def calculate_metrics(y_true, y_pred,duration,clustering=False):
    """
    Return a data frame that contains the precision, accuracy, recall and the duration
    """
    res = pd.DataFrame(data = np.zeros((1,5),dtype=np.float), index=[0],
        columns=['precision','accuracy','error','recall','duration'])
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['accuracy'] = accuracy_score(y_true,y_pred)
    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['duration'] = duration
    res['error'] = 1-res['accuracy']
    return res

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

def draw(method):
    dataset_name = 'Coffee'
    classifier_name = 'resnet'
    dir_model = root_dir+'results/' + classifier_name + '/'+archive_name+'/'\
                + dataset_name + '/best_model.hdf5'
    eps = 0.1

    # load attack
    data_adv_dir = root_dir_attack+method+'/eps-'+str(eps)+'/'
    file_name_attack = data_adv_dir+dataset_name+'-adv'
    data = np.loadtxt(file_name_attack,delimiter=',')
    x_test_attack = data[:,1:]

    # load original
    file_name = root_dir_archive + dataset_name +'/'+dataset_name+'_TEST'
    data = np.loadtxt(file_name,delimiter=',')
    x_test = data[:,1:]
    y_test = data[:,0]
    # load train
    file_name = root_dir_archive + dataset_name + '/' + dataset_name + '_TRAIN'
    data = np.loadtxt(file_name, delimiter=',')
    y_train = data[:, 0]

    # number of examples
    n = len(x_test)
    # n_train = len(x_train)

    y_train, y_test = transform_labels(y_train, y_test)
    # load model
    model = keras.models.load_model(dir_model)
    # predictions for perturbed instances
    y_pred_attack_proba = model.predict(x_test_attack.reshape(n, -1, 1))
    # get labels for perturbed instances
    y_pred_attack = np.argmax(y_pred_attack_proba, axis=1)


    index = 0

    print(y_pred_attack[index],y_test[index])

    plt.figure()
    plt.plot(x_test[index], color='blue',label='original')
    plt.plot(x_test_attack[index], color='red',label='fake')
    plt.legend(loc='best')
    plt.savefig(data_adv_dir+dataset_name+'.pdf')
    plt.show()

def mds(method):
    dataset_name = 'Coffee'
    classifier_name = 'resnet'
    dir_model = root_dir+'results/'+classifier_name+'/'+archive_name+'/'+\
                dataset_name+'/best_model.hdf5'
    eps = 0.1

    # load attack
    data_adv_dir = root_dir_attack + method + '/eps-' + str(eps) + '/'
    file_name_attack = data_adv_dir+ dataset_name + '-adv'
    data = np.loadtxt(file_name_attack, delimiter=',')
    x_test_attack = data[:, 1:]

    # load original
    file_name = root_dir_archive + dataset_name + '/' + dataset_name + '_TEST'
    data = np.loadtxt(file_name, delimiter=',')
    x_test = data[:, 1:]
    y_test = data[:, 0]
    # load train
    file_name = root_dir_archive + dataset_name + '/' + dataset_name + '_TRAIN'
    data = np.loadtxt(file_name, delimiter=',')
    # x_train = data[:,1:]
    y_train = data[:,0]

    # number of examples
    n = len(x_test)
    # n_train = len(x_train)

    y_train, y_test = transform_labels(y_train, y_test)
    # load model
    model = keras.models.load_model(dir_model)
    # predictions for perturbed instances
    y_pred_attack_proba = model.predict(x_test_attack.reshape(n,-1,1))
    # get labels for perturbed instances
    y_pred_attack = np.argmax(y_pred_attack_proba,axis=1)

    ##### mds sur resnet
    new_input_layer = model.inputs  # same input of the original model
    new_output_layer = [model.layers[-2].output]
    new_feed_forward = keras.backend.function(new_input_layer,new_output_layer)

    x_test = new_feed_forward([x_test.reshape(n,-1,1)])
    x_test_attack = new_feed_forward([x_test_attack.reshape(n,-1,1)])

    x_test = np.array(x_test, dtype=np.float16)[0]
    x_test_attack = np.array(x_test_attack, dtype=np.float16)[0]
    # #####

    markers = ['o','x','^']

    # get confiedence for each class
    classes,counts = np.unique(y_test,return_counts=True)
    for i in range(len(classes)):
        clas = classes[i]
        count = counts[i]
        conf = y_pred_attack_proba[y_pred_attack!=y_test]
        conf = conf[np.argmax(conf,axis=1)!=clas]
        num = len(conf)
        conf = np.max(conf,axis=1)
        print('Class '+str(clas))
        print('Number of misclassified instances: '+str(num))
        print('Average confidence is: '+str(conf.mean()))
        print('Number of class instances: '+str(count))
        print('#####################')

    # concat
    new_x_test = np.concatenate((x_test_attack,x_test))

    # color with attacked or not
    colors_attack = ['red' for i in range(n)]
    colors_original = ['blue' for i in range(n)]

    colors = np.concatenate((colors_attack,colors_original))

    # apply mds
    embedding = MDS(n_components=2,random_state=12)
    x_test_transformed = embedding.fit_transform(new_x_test)
    plt.figure()

    # plot perturbed
    for c in classes:
        x = x_test_transformed[:n][y_test == c][:,0]
        y = x_test_transformed[:n][y_test == c][:,1]
        plt.scatter(x, y, color='red', marker=markers[c])
    # plot originale
    for c in classes:
        x = x_test_transformed[n:][y_test==c][:,0]
        y = x_test_transformed[n:][y_test==c][:,1]
        plt.scatter(x,y,color='blue',marker=markers[c])

    plt.savefig(data_adv_dir+dataset_name+'.pdf')
    plt.show()

def noise(method):
    dataset_name = 'Coffee'
    classifier_name = 'resnet'
    dir_model = root_dir+'results/' + classifier_name + \
                '/'+archive_name+'/' + dataset_name + '/best_model.hdf5'
    eps = 0.1

    # load attack
    data_adv_dir = root_dir_attack + method + '/eps-' + str(eps) + '/'
    file_name_attack = data_adv_dir + dataset_name + '-adv'
    data = np.loadtxt(file_name_attack, delimiter=',')
    x_test_attack = data[:, 1:]

    # load original
    file_name = root_dir_archive + dataset_name + '/' + dataset_name + '_TEST'
    data = np.loadtxt(file_name, delimiter=',')
    x_test = data[:, 1:]
    y_test = data[:,0]
    # load train
    file_name = root_dir_archive + dataset_name + '/' + dataset_name + '_TRAIN'
    data = np.loadtxt(file_name, delimiter=',')
    x_train = data[:,1:]
    y_train = data[:,0]

    # add a dimension to make it multivariate with one dimension
    # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test_attack = x_test_attack.reshape((x_test.shape[0], x_test.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    y_train, y_test = transform_labels(y_train, y_test)

    # load model
    model = keras.models.load_model(dir_model)

    index = 1

    label_ori = y_test[index]
    print('original label',label_ori)
    ori_instance = x_test[index]
    pert_instance = x_test_attack[index]
    noise_ = ori_instance - pert_instance

    pred_ori = model.predict(ori_instance.reshape(1,-1,1))[0]
    label_pred_ori = np.argmax(pred_ori,axis=0)

    pred_pert = model.predict(pert_instance.reshape(1,-1,1))[0]
    label_pred_pert = np.argmax(pred_pert,axis=0)

    normalized_noise = zNormalize(noise_)
    pred_noise = model.predict(normalized_noise.reshape(1,-1,1))[0]
    label_pred_noise = np.argmax(pred_noise,axis=0)
    print('noise label',label_pred_noise)

    assert label_ori == label_pred_ori
    assert label_pred_ori!=label_pred_pert

    conf_ori = pred_ori[label_pred_ori]
    conf_pert = pred_pert[label_pred_pert]
    conf_noise = pred_noise[label_pred_noise]

    print('conf_ori',conf_ori)
    print('conf_pert',conf_pert)
    print('conf_noise',conf_noise)

    plt.figure()
    plt.plot(ori_instance, color='blue', label='original')
    plt.plot(pert_instance, color='red', label='fake')
    plt.plot(noise_-1.5, color='gray', label='noise')
    plt.legend(loc='best')
    plt.savefig(data_adv_dir+dataset_name+'.pdf')
    plt.show()

if sys.argv[1]=='attack':
    attack_method = sys.argv[2]
    attack.main(attack_method=attack_method)

elif sys.argv[1] == 'draw':
    attack_method = sys.argv[2]
    draw(attack_method)

elif sys.argv[1]== 'mds':
    attack_method = sys.argv[2]
    mds(attack_method)

elif sys.argv[1]=='noise':
    attack_method = sys.argv[2]
    noise(attack_method)