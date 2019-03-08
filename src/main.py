import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import keras
from sklearn.metrics.pairwise import paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import knn
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from scipy.interpolate import spline
import os
import imageio

# DATASETS = ['50words','Adiac','ArrowHead','Beef','BeetleFly',
#                             'BirdChicken','Car','CBF','ChlorineConcentration',
#                             'CinC_ECG_torso','Coffee','Computers','Cricket_X',
#                             'Cricket_Y','Cricket_Z','DiatomSizeReduction',
#                             'DistalPhalanxOutlineAgeGroup',
#                             'DistalPhalanxOutlineCorrect','DistalPhalanxTW',
#                             'Earthquakes','ECG200','ECG5000','ECGFiveDays',
#                             'ElectricDevices','FaceAll','FaceFour','FacesUCR',
#                             'FISH','FordA','FordB','Gun_Point','Ham',
#                             'HandOutlines','Haptics','Herring','InlineSkate',
#                             'InsectWingbeatSound','ItalyPowerDemand',
#                             'LargeKitchenAppliances','Lighting2','Lighting7',
#                             'MALLAT','Meat','MedicalImages',
#                             'MiddlePhalanxOutlineAgeGroup',
#                             'MiddlePhalanxOutlineCorrect','MiddlePhalanxTW',
#                             'MoteStrain','NonInvasiveFatalECG_Thorax1',
#                             'NonInvasiveFatalECG_Thorax2','OliveOil','OSULeaf',
#                             'PhalangesOutlinesCorrect','Phoneme','Plane',
#                             'ProximalPhalanxOutlineAgeGroup',
#                             'ProximalPhalanxOutlineCorrect',
#                             'ProximalPhalanxTW','RefrigerationDevices',
#                             'ScreenType','ShapeletSim','ShapesAll',
#                             'SmallKitchenAppliances','SonyAIBORobotSurface',
#                             'SonyAIBORobotSurfaceII','StarLightCurves',
#                             'Strawberry','SwedishLeaf','Symbols',
#                             'synthetic_control','ToeSegmentation1',
#                             'ToeSegmentation2','Trace','TwoLeadECG',
#                             'Two_Patterns','UWaveGestureLibraryAll',
#                             'uWaveGestureLibrary_X','uWaveGestureLibrary_Y',
#                             'uWaveGestureLibrary_Z','wafer','Wine',
#                             'WordsSynonyms','Worms','WormsTwoClass','yoga']

DATASETS = ['FordA','FordB','Gun_Point','Ham',
                            'HandOutlines','Haptics','Herring','InlineSkate',
                            'InsectWingbeatSound','ItalyPowerDemand',
                            'LargeKitchenAppliances','Lighting2','Lighting7',
                            'MALLAT','Meat','MedicalImages',
                            'MiddlePhalanxOutlineAgeGroup',
                            'MiddlePhalanxOutlineCorrect','MiddlePhalanxTW',
                            'MoteStrain','NonInvasiveFatalECG_Thorax1',
                            'NonInvasiveFatalECG_Thorax2','OliveOil','OSULeaf',
                            'PhalangesOutlinesCorrect','Phoneme','Plane',
                            'ProximalPhalanxOutlineAgeGroup',
                            'ProximalPhalanxOutlineCorrect',
                            'ProximalPhalanxTW','RefrigerationDevices',
                            'ScreenType','ShapeletSim','ShapesAll',
                            'SmallKitchenAppliances','SonyAIBORobotSurface',
                            'SonyAIBORobotSurfaceII','StarLightCurves',
                            'Strawberry','SwedishLeaf','Symbols',
                            'synthetic_control','ToeSegmentation1',
                            'ToeSegmentation2','Trace','TwoLeadECG',
                            'Two_Patterns','UWaveGestureLibraryAll',
                            'uWaveGestureLibrary_X','uWaveGestureLibrary_Y',
                            'uWaveGestureLibrary_Z','wafer','Wine',
                            'WordsSynonyms','Worms','WormsTwoClass','yoga']


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

def draw():
    root_dir = '/mnt/nfs/casimir/archives/UCR_TS_Archive_2015/'
    root_dir_attack = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/'
    method = 'fgsm'
    dataset_name = 'FordA'
    classifier_name = 'resnet'
    dir_model = '/mnt/nfs/casimir/results/' + classifier_name + '/UCR_TS_Archive_2015/' + dataset_name + '/best_model.hdf5'

    # load attack
    eps = ''
    file_name_attack = root_dir_attack+method+'/'+dataset_name+str(eps)+'-adv'
    data = np.loadtxt(file_name_attack,delimiter=',')
    x_test_attack = data[:,1:]

    # load original
    file_name = root_dir + dataset_name +'/'+dataset_name+'_TEST'
    data = np.loadtxt(file_name,delimiter=',')
    x_test = data[:,1:]
    y_test = data[:,0]
    # load train
    file_name = root_dir + dataset_name + '/' + dataset_name + '_TRAIN'
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
    plt.savefig('test'+str(eps)+'.pdf')
    plt.show()

def mds():
    root_dir = '/mnt/nfs/casimir/archives/UCR_TS_Archive_2015/'
    root_dir_attack = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/'
    method = 'bim'
    dataset_name = 'Ham'
    classifier_name = 'resnet'
    dir_model = '/mnt/nfs/casimir/results/'+classifier_name+'/UCR_TS_Archive_2015/'+dataset_name+'/best_model.hdf5'

    # load attack
    file_name_attack = root_dir_attack + method + '/' + dataset_name + '-adv'
    data = np.loadtxt(file_name_attack, delimiter=',')
    x_test_attack = data[:, 1:]

    # load original
    file_name = root_dir + dataset_name + '/' + dataset_name + '_TEST'
    data = np.loadtxt(file_name, delimiter=',')
    x_test = data[:, 1:]
    y_test = data[:, 0]
    # load train
    file_name = root_dir + dataset_name + '/' + dataset_name + '_TRAIN'
    data = np.loadtxt(file_name, delimiter=',')
    # x_train = data[:,1:]
    y_train = data[:,0]

    # reduce the test set
    # max_instances = 50
    # indices = np.random.permutation(max_instances)
    # x_test = x_test[indices,:]
    # y_test = y_test[indices]
    # x_test_attack = x_test_attack[indices,:]

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
    # colors_train = ['cyan' for i in range(n_train)]

    # # color with classes
    # colors_attack = y_test
    # colors_original = y_test

    # color with fooled or not
    # colors_original = np.zeros((n,),dtype=np.float32) # blue
    # colors_attack = y_test == y_pred_attack
    # colors_attack = colors_attack  +1

    colors = np.concatenate((colors_attack,colors_original))

    # apply mds
    embedding = MDS(n_components=2,random_state=12)
    x_test_transformed = embedding.fit_transform(new_x_test)
    plt.figure()
    # x = x_test_transformed[:,0]
    # y = x_test_transformed[:,1]
    # plt.scatter(x,y,c=colors)
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

    plt.savefig('test.pdf')
    plt.show()

def distortion():
    root_dir = '/mnt/nfs/casimir/archives/UCR_TS_Archive_2015/'
    root_dir_attack = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/'
    method = 'bim'
    file_df = root_dir_attack+method+'/results.csv'

    df = pd.read_csv(file_df, index_col=False)
    df['distortion_avg'] = 0
    df['distortion_std'] = 0
    df['dist_avg'] = 0
    df['dist_std'] = 0

    for dataset_name in DATASETS:

        # load attack
        file_name_attack = root_dir_attack + method + '/' + dataset_name + '-adv'
        data = np.loadtxt(file_name_attack, delimiter=',')
        x_test_attack = data[:, 1:]

        # load original
        file_name = root_dir + dataset_name + '/' + dataset_name + '_TEST'
        data = np.loadtxt(file_name, delimiter=',')
        x_test = data[:, 1:]
        y_test = data[:, 0]

        eds = paired_euclidean_distances(x_test,x_test_attack)

        distortion_avg = eds.mean()
        distortion_std = eds.std()

        df.loc[df.dataset_name==dataset_name,'distortion_avg'] = distortion_avg
        df.loc[df.dataset_name==dataset_name,'distortion_std'] = distortion_std

        # average distances between original samples
        eds_ori = euclidean_distances(x_test,x_test)
        dist_avg = eds_ori.mean()
        dist_std = eds_ori.std()

        df.loc[df.dataset_name==dataset_name,'dist_avg'] = dist_avg
        df.loc[df.dataset_name==dataset_name,'dist_std'] = dist_std

    df.to_csv(file_df,index=False)

def NN():

    root_dir = '/mnt/nfs/casimir/archives/UCR_TS_Archive_2015/'
    root_dir_attack = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/results-sdm-2019/'
    method = 'fgsm'
    file_df = root_dir_attack + method + '/results.csv'

    # df = pd.DataFrame(data=np.zeros(shape=(len(DATASETS),2),dtype=np.float32),
    #                   index=DATASETS,columns=['ori_acc','adv_acc'])

    out_file_dir = 'results-nn-dtw.csv'

    df = pd.read_csv(out_file_dir, index_col=0)

    for dataset_name in DATASETS:

        # load attack
        file_name_attack = root_dir_attack + method + '/' + dataset_name + '-adv'
        data = np.loadtxt(file_name_attack, delimiter=',')
        x_test_attack = data[:, 1:]

        # load original
        file_name = root_dir + dataset_name + '/' + dataset_name + '_TEST'
        data = np.loadtxt(file_name, delimiter=',')
        x_test = data[:, 1:]
        y_test = data[:, 0]
        # load train
        file_name = root_dir + dataset_name + '/' + dataset_name + '_TRAIN'
        data = np.loadtxt(file_name, delimiter=',')
        x_train = data[:,1:]
        y_train = data[:,0]
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test_attack = x_test_attack.reshape((x_test.shape[0], x_test.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        y_train, y_test = transform_labels(y_train, y_test)

        # print('Acc on attack:')
        y_pred = knn.knn(x_train, y_train, x_test_attack, 1, distance_algorithm='dtw')
        df_metrics = calculate_metrics(y_test, y_pred, 0.0)
        adv_acc = df_metrics['accuracy'][0]

        # print(df_metrics)
        # print('################################')
        # print('Ori acc:')
        y_pred = knn.knn(x_train, y_train, x_test, 1, distance_algorithm='dtw')
        df_metrics = calculate_metrics(y_test, y_pred, 0.0)
        ori_acc = df_metrics['accuracy'][0]

        df.loc[dataset_name,'ori_acc'] = ori_acc
        df.loc[dataset_name,'adv_acc'] = adv_acc

        df.to_csv(out_file_dir)

    print('DONE')

def noise():
    root_dir = '/mnt/nfs/casimir/archives/UCR_TS_Archive_2015/'
    root_dir_attack = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/results-sdm-2019/'
    method = 'fgsm'
    dataset_name = 'TwoLeadECG'
    classifier_name = 'resnet'
    dir_model = '/mnt/nfs/casimir/results/' + classifier_name + '/UCR_TS_Archive_2015/' + dataset_name + '/best_model.hdf5'

    # load attack
    file_name_attack = root_dir_attack + method + '/' + dataset_name + '-adv'
    data = np.loadtxt(file_name_attack, delimiter=',')
    x_test_attack = data[:, 1:]

    # load original
    file_name = root_dir + dataset_name + '/' + dataset_name + '_TEST'
    data = np.loadtxt(file_name, delimiter=',')
    x_test = data[:, 1:]
    y_test = data[:,0]
    # load train
    file_name = root_dir + dataset_name + '/' + dataset_name + '_TRAIN'
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
    # plt.plot(normalized_noise-1.5, color='gray', label='normalized_noise')
    plt.savefig('test.pdf')
    plt.show()

def results():
    root_dir_attack = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/results-sdm-2019/'

    df_res = pd.DataFrame(data=np.zeros((len(DATASETS),4),dtype=np.float32),
                          columns=['fgsm_acc','bim_acc','fgsm_dist','bim_dist'],
                          index=DATASETS)

    df_fgsm = pd.read_csv(root_dir_attack + 'fgsm/results.csv',
                          index_col=False)
    df_bim = pd.read_csv(root_dir_attack + 'bim/results.csv',
                         index_col=False)

    df_res['fgsm_acc'] = (df_fgsm['ori_acc'] - df_fgsm['adv_acc']).values * 100
    df_res['bim_acc'] = (df_bim['ori_acc'] - df_bim['adv_acc']).values * 100

    df_res['fgsm_dist'] = ((df_fgsm['dist_avg'] - df_fgsm['distortion_avg'])
                           / df_fgsm['dist_avg']).values *100
    df_res['bim_dist'] = ((df_bim['dist_avg'] - df_bim['distortion_avg'])
                           / df_bim['dist_avg']).values * 100

    df_res.to_csv(root_dir_attack+'res.csv',index=True,header=True,
                  sep=',',float_format='%.1f')

    print('avg acc decrease for fgsm',df_res['fgsm_acc'].mean())
    print('avg acc decrease for bim',df_res['bim_acc'].mean())

    # plot acc
    ser_fgsm = df_res['fgsm_acc']/100
    ser_bim = df_res['bim_acc']/100
    x=np.arange(start=0,stop=1,step=0.01)
    plt.figure()
    plt.xlim(xmax=1.02,xmin=0.0)
    plt.ylim(ymax=1.02,ymin=0.0)
    plt.scatter(x=ser_fgsm,y=ser_bim,color='blue')
    plt.xlabel('fgsm')
    plt.ylabel('bim')
    plt.plot(x,x,color='black')
    plt.savefig('acc-plot.pdf')
    plt.show()

    uniq, counts = np.unique(ser_fgsm < ser_bim, return_counts=True)
    print('Wins', counts[-1])
    uniq, counts = np.unique(ser_fgsm == ser_bim, return_counts=True)
    print('Draws', counts[-1])
    uniq, counts = np.unique(ser_fgsm > ser_bim, return_counts=True)
    print('Losses', counts[-1])
    p_value = wilcoxon(ser_fgsm, ser_bim, zero_method='pratt')[1]
    print(p_value)

    # plot dist
    # x = np.arange(start=0, stop=1, step=0.01)
    # plt.figure(1)
    # plt.xlim(xmax=1.02, xmin=0.0)
    # plt.ylim(ymax=1.02, ymin=0.0)
    # plt.scatter(x=df_res['fgsm_dist'] / 100, y=df_res['bim_dist'] / 100, color='blue')
    # plt.xlabel('fgsm')
    # plt.ylabel('bim')
    # plt.plot(x, x, color='black')
    # plt.show()
    # plt.savefig(root_dir_attack + 'dist-plot.pdf')

def test_fcn():
    root_dir = '/mnt/nfs/casimir/archives/UCR_TS_Archive_2015/'
    root_dir_attack = '../cleverhans/ucr-attack/results-sdm-2019/'
    method = 'bim'
    classifier_name = 'fcn'
    batch_size = 1024
    res_df = pd.DataFrame(data=np.zeros(shape=(len(DATASETS),2),dtype=np.float32),
                          index=DATASETS,columns=['acc_adv','acc_ori'])

    for dataset_name in ['Adiac','Coffee']:
        dir_model = '/mnt/nfs/casimir/results/' + classifier_name + \
                    '/UCR_TS_Archive_2015/' + dataset_name + '/best_model.hdf5'
        # load model
        model = keras.models.load_model(dir_model)
        # load attack
        eps = ''
        file_name_attack = root_dir_attack+method+'/'+dataset_name+str(eps)+'-adv'
        data = np.loadtxt(file_name_attack,delimiter=',')
        x_test_attack = data[:,1:]

        # load original
        file_name = root_dir + dataset_name + '/' + dataset_name + '_TEST'
        data = np.loadtxt(file_name, delimiter=',')
        x_test = data[:, 1:]
        y_test = data[:, 0]
        # load train
        file_name = root_dir + dataset_name + '/' + dataset_name + '_TRAIN'
        data = np.loadtxt(file_name, delimiter=',')
        x_train = data[:, 1:]
        y_train = data[:, 0]

        n = len(x_test)
        # n_train = len(x_train)

        y_train, y_test = transform_labels(y_train, y_test)

        # test on adv

        # predictions for perturbed instances
        y_pred_attack_proba = model.predict(x_test_attack.reshape(n, -1, 1),
                                            batch_size=batch_size)
        # get labels for perturbed instances
        y_pred_attack = np.argmax(y_pred_attack_proba, axis=1)

        df_metrics = calculate_metrics(y_test,y_pred_attack,0.0)

        res_df.loc[dataset_name,'acc_adv'] = df_metrics['accuracy'][0]

        # test on original
        y_pred_proba = model.predict(x_test.reshape(n, -1, 1),
                                            batch_size=batch_size)
        y_pred = np.argmax(y_pred_proba, axis=1)
        df_metrics = calculate_metrics(y_test, y_pred, 0.0)
        res_df.loc[dataset_name,'acc_ori'] = df_metrics['accuracy'][0]

    res_df.to_csv('res-fcn-'+method+'.csv')

def plteps():
    root_dir_attack = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/'
    dataset_name = 'FordA'
    df_fgsm = pd.read_csv(root_dir_attack+dataset_name+'/fgsm/results.csv')
    df_bim = pd.read_csv(root_dir_attack+dataset_name+'/bim/results.csv')

    plt.figure()
    plt.ylim(ymin=0.0,ymax=100)
    plt.xlim(xmin=-0.05,xmax=1.6)
    plt.xticks([0.0,0.5,1.0,1.5])
    plt.plot(df_fgsm['eps'],df_fgsm['ori_acc']*100,color='green')
    plt.plot(df_fgsm['eps'],df_fgsm['adv_acc']*100,color='blue')
    plt.plot(df_fgsm['eps'],df_bim['adv_acc']*100,color='red')
    plt.savefig('plot-eps.pdf')
    plt.show()

def printfcn():
    df_fgsm = pd.read_csv('res-fcn-fgsm.csv')
    df_bim = pd.read_csv('res-fcn-bim.csv')


    print('Avg fgsm on fcn ',(df_fgsm['acc_ori']-df_fgsm['acc_adv']).mean())
    print('std fgsm on fcn ',(df_fgsm['acc_ori']-df_fgsm['acc_adv']).std())
    print('Avg bim on fcn ',(df_bim['acc_ori']-df_bim['acc_adv']).mean())
    print('std bim on fcn ',(df_bim['acc_ori']-df_bim['acc_adv']).std())

    ser_fgsm = df_fgsm['acc_adv']
    ser_bim = df_bim['acc_adv']

    uniq, counts = np.unique(ser_fgsm < ser_bim, return_counts=True)
    print('Wins', counts[-1])
    uniq, counts = np.unique(ser_fgsm == ser_bim, return_counts=True)
    print('Draws', counts[-1])
    uniq, counts = np.unique(ser_fgsm > ser_bim, return_counts=True)
    print('Losses', counts[-1])
    p_value = wilcoxon(ser_fgsm, ser_bim, zero_method='pratt')[1]
    print(p_value)

def allresults():
    df_fcn_fgsm = pd.read_csv('res-fcn-fgsm.csv')
    df_fcn_bim = pd.read_csv('res-fcn-bim.csv')

    root_dir_attack = '../cleverhans/ucr-attack/'
    file_path = root_dir_attack + 'fgsm/results.csv'
    df_resnet_fgsm = pd.read_csv(file_path)
    file_path = root_dir_attack + 'bim/results.csv'
    df_resnet_bim = pd.read_csv(file_path)

    columns = ['resnet_ori','resnet_fgsm_adv','resnet_bim_adv' ,
               'fcn_ori', 'fcn_fgsm_adv', 'fcn_bim_adv']

    df_res = pd.DataFrame(data=np.zeros(shape=(len(DATASETS),6),dtype=np.float32),
                          index=DATASETS,columns=columns)

    df_res['resnet_ori'] = df_resnet_fgsm['ori_acc'].values*100
    df_res['resnet_fgsm_adv'] = df_resnet_fgsm['adv_acc'].values*100

    # df_res['resnet_bim_ori'] = df_resnet_bim['ori_acc'].values*100
    df_res['resnet_bim_adv'] = df_resnet_bim['adv_acc'].values*100

    df_res['fcn_ori'] = df_fcn_fgsm['acc_ori'].values*100
    df_res['fcn_fgsm_adv'] = df_fcn_fgsm['acc_adv'].values*100

    # df_res['fcn_bim_ori'] = df_fcn_bim['acc_ori'].values*100
    df_res['fcn_bim_adv'] = df_fcn_bim['acc_adv'].values*100

    df_res.to_csv('all-results.csv',float_format="%.1f")

def plot_length():
    df_res = pd.read_csv('all-results.csv',index_col=0)
    root_dir = '/mnt/nfs/casimir/archives/UCR_TS_Archive_2015/'
    method = 'fgsm'
    classifier = 'resnet'

    # get the length for each dataset
    for dataset_name in DATASETS:
        file_name = root_dir + dataset_name + '/' + dataset_name + '_TRAIN'
        data = np.loadtxt(file_name, delimiter=',')
        x_train = data[:, 1:]

        df_res.loc[dataset_name,'length'] = x_train.shape[1]

    x = df_res['length'].values
    y = (df_res[classifier+'_ori'] - df_res[classifier+'_'+method+'_adv']).values
    y = y/df_res[classifier+'_ori'].values

    reg = LinearRegression().fit(x.reshape(-1,1),y)
    xs = np.array(range(int(x.max())))
    ys = reg.predict(xs.reshape(-1,1))

    plt.figure()
    plt.xlabel('length')
    plt.ylabel('accuracy')
    plt.ylim(ymax = 1.0,ymin=0)
    plt.scatter(x=x , y=y)
    plt.plot(ys,color='red')
    plt.savefig('plot-length.pdf')
    plt.show()

def gif():
    root_dir = '/mnt/nfs/casimir/archives/UCR_TS_Archive_2015/'
    root_dir_attack = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/'
    dataset_name = 'ItalyPowerDemand'
    out_dir = 'gif/'

    df_res = pd.read_csv(root_dir_attack+dataset_name+'/results.csv',index_col=False)

    df_res = df_res.sort_values(by=['eps'])

    ser_eps = df_res['eps'].values
    ser_adv_acc = df_res['adv_acc'].values*100
    ser_ori_acc = df_res['ori_acc'].values*100

    # ser_eps, ser_adv_acc = smooth(ser_eps,ser_adv_acc)

    # ser_eps = ser_eps.values
    # ser_adv_acc = ser_adv_acc.values
    # ser_ori_acc = ser_ori_acc.values

    # plot acc
    n = len(ser_eps)+1
    for i in range(1,n):
        plt.figure()
        plt.ylim(ymin=0.0, ymax=100)
        plt.xlim(xmin=-0.05, xmax=2.05)
        plt.xticks([0.0, 0.5, 1.0, 1.5,2.0])

        plt.plot(ser_eps[:i], ser_adv_acc[:i],color='red',label='with attack')
        plt.plot(ser_eps, ser_ori_acc,color='blue',label='without attack')
        plt.xlabel('amount of perturbation')
        plt.ylabel('accuracy')
        plt.title('Accuracy ItalyPowerDemand')
        plt.legend(loc='lower left')

        plt.savefig(out_dir+'acc/plot-'+str(f'{i:03}')+'.png')
        plt.close()


    # plot the series
    index = 3
    for i in range(len(ser_eps)):
        eps = ser_eps[i]

        # load attack
        file_name_attack = root_dir_attack + dataset_name + '/' + dataset_name + str(eps) + '-adv'
        data = np.loadtxt(file_name_attack, delimiter=',')
        x_test_attack = data[:, 1:]
        x_adv = x_test_attack[index]

        # load original
        file_name = root_dir + dataset_name + '/' + dataset_name + '_TEST'
        data = np.loadtxt(file_name, delimiter=',')
        x_test = data[:, 1:]
        x_ori = x_test[index]

        plt.figure()
        plt.ylim(ymax=5.0,ymin=-4.0)

        plt.xlabel('time')
        plt.ylabel('value')

        plt.plot(x_ori,color='blue',label = 'original time series')
        plt.plot(x_adv,color='red' , label= 'perturbed time series')

        plt.title('Example ItalyPowerDemand')
        plt.legend(loc='upper left')


        plt.savefig(out_dir+'ts/plot-'+str(f'{i:03}')+'.png')
        plt.close()

    # gif of time series
    images = []
    out_ts_dir  = out_dir+'ts/'
    for subdir,dirs,files in os.walk(out_ts_dir):
        files.sort()
        for file_name in files:
            images.append(imageio.imread(out_ts_dir + file_name))
        output_path = out_dir + 'ts.gif'
        kargs = {'duration': 0.25}
        imageio.mimsave(output_path, images, 'GIF', **kargs)

    # gif of plots acc
    images = []
    out_ts_dir = out_dir + 'acc/'
    for subdir, dirs, files in os.walk(out_ts_dir):
        files.sort()
        for file_name in files:
            images.append(imageio.imread(out_ts_dir + file_name))
        output_path = out_dir + 'acc.gif'
        kargs = {'duration': 0.25}
        imageio.mimsave(output_path, images, 'GIF', **kargs)

def smooth(x,y):
    # similar to sp2m to get the window length for the moving average
    tenth = int(len(x) / 1)
    if tenth % 2 == 1:
        tenth = tenth + 1
    w = tenth + 1
    # moving average to eliminate spikes in curve
    y = y.rolling(window=w, center=False, min_periods=1).mean()
    # smoothness
    smoothness = 300
    # linear interpolate to smooth
    x_new = np.linspace(x.min(), x.max(), smoothness)
    # print(dataset_name,algorithm_name)

    y_new = spline(x, y, x_new)

    return x_new,y_new

def test_models():
    root_dir = '/mnt/todel/pre-trained-resnet/'
    root_dir_ori = '/mnt/nfs/casimir-home/gits/cleverhans/ucr-attack/results-sdm-2019/'
    root_dir_new = '/mnt/todel/results-sdm-2019/'
    method = 'fgsm'
    for d in DATASETS:
        print('dataset:',d)
        # m = keras.models.load_model(root_dir+d+'/best_model.hdf5')
        # print('success')

        # load
        file_name_attack = root_dir_ori + method + '/' + d + '-adv'
        data_ori = np.loadtxt(file_name_attack, delimiter=',')


        file_name_attack = root_dir_new + method + '/' + d + '-adv'
        data_new = np.loadtxt(file_name_attack, delimiter=',')

        if data_new.all() == data_ori.all() :
            print('equal')
        else:
            print('falseeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            exit()




# main
mds()