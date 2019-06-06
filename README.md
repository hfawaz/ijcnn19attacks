# Adversarial Attacks on Deep Neural Networks for Time Series Classification
This is the companion repository for our paper also available on [ArXiv](https://arxiv.org/abs/1903.07054) titled "Adversarial Attacks on Deep Neural Networks for Time Series Classification". 
This paper has been accepted at the [IEEE International Joint Conference on Neural Networks (IJCNN) 2019](https://www.ijcnn.org/). 

## Approach 
![fgsm](https://github.com/hfawaz/ijcnn19attacks/blob/master/img/pert-example.png)

## Data 
The data used in this project comes from the [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data/UCR_TS_Archive_2015.zip), which contains the 85 univariate time series datasets. 

## Pre-trained models
You can download the pre-trained ResNet models for each dataset in the archive [here](https://germain-forestier.info/src/ijcnn2019/pre-trained-resnet.zip). 
These are the models used to generate the adversarial time series examples. 
They are published for reproducibility, nevertheless the code can be applied to any model in the [h5py](http://docs.h5py.org/en/latest/build.html) format. 

## Prerequisites
All python packages needed are listed in [pip-requirements.txt](https://github.com/hfawaz/ijcnn19ensemble/blob/master/src/utils/pip-requirements.txt) file and can be installed simply using the pip command.

* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  
* [keras](https://keras.io/)  
* [h5py](http://docs.h5py.org/en/latest/build.html)

Note that for convience we copied some of the code from the [Cleverhans API](https://github.com/tensorflow/cleverhans) and modified it to adapt it for time series data instead of images. 

## Code
To perform the ```fgsm``` (or ```bim```) attack on the datasets in the UCR archive you can run the following command: 
```
python3 main.py attack fgsm 
```

Once the perturbed time series are generated, you can launch this command to plot and visualize the difference: 
```
python3 main.py draw fgsm 
```

If you want to visualize the noise as well you can run this command: 
```
python3 main.py noise fgsm 
```

Finally to plot the Multi-Dimensional Scaling (MDS) and visualize the difference between an original and perturbed dataset, you can issue this command: 
```
python3 main.py mds fgsm 
```

## Results 

The following animation shows how the accuracy and the time series shape variation with respect to the amount of perturbation allowed. 
![fgsm](https://github.com/hfawaz/ijcnn19attacks/blob/master/img/acc-ts.gif)

The folllowing table shows the accuracy over the 85 datasets with and without adversarial perturbation, using both attacks FGSM and BIM for two models ResNet (white-box mode) and FCN (black-box mode).
The raw csv results can be found [here](https://github.com/hfawaz/ijcnn19attacks/blob/master/results/all-results.csv).
For example column 'resnet_ori' shows the original accuracy of ResNet over the 85 datasets, while column 'resnet_fgsm_adv' shows the accuracy after performing the FGSM attack.

|                                | resnet_ori | resnet_fgsm_adv | resnet_bim_adv | fcn_ori | fcn_fgsm_adv | fcn_bim_adv | 
|--------------------------------|------------|-----------------|----------------|---------|--------------|-------------| 
| 50words                        | 73.2       | 17.1            | **8.8**            | 45.5    | **7.7**          | 8.8         | 
| Adiac                          | 83.1       | 3.1             | **1.5**            | 84.7    | 2.8          | **2.0**         | 
| ArrowHead                      | 85.1       | 33.1            | **14.3**           | 82.3    | 41.7         | **29.1**        | 
| Beef                           | 76.7       | 20.0            | **10.0**           | 70.0    | **26.7**         | 36.7        | 
| BeetleFly                      | 85.0       | **15.0**            | **15.0**           | 90.0    | **25.0**         | **25.0**        | 
| BirdChicken                    | 95.0       | 55.0            | **15.0**           | 100.0   | 60.0         | **45.0**        | 
| Car                            | 93.3       | 21.7            | **6.7**            | 90.0    | 21.7         | **11.7**        | 
| CBF                            | 98.9       | 86.1            | **84.8**           | 99.4    | 95.3         | **94.7**        | 
| ChlorineConcentration          | 83.5       | 12.3            | **11.8**           | 82.4    | **12.3**         | 12.5        | 
| CinC_ECG_torso                 | 83.8       | 25.4            | **23.3**           | 83.8    | 25.7         | **23.6**        | 
| Coffee                         | 100.0      | 50.0            | **35.7**           | 100.0   | 75.0         | **64.3**        | 
| Computers                      | 81.2       | 40.8            | **24.0**           | 81.6    | 58.4         | **30.8**        | 
| Cricket_X                      | 79.0       | 35.4            | **20.8**           | 79.5    | 43.8         | **34.1**        | 
| Cricket_Y                      | 80.5       | 24.9            | **13.8**           | 76.7    | 28.5         | **20.8**        | 
| Cricket_Z                      | 81.5       | 27.7            | **16.2**           | 80.3    | 35.4         | **26.2**        | 
| DiatomSizeReduction            | 30.1       | 46.7            | **34.6**           | 30.4    | **43.1**         | 57.8        | 
| DistalPhalanxOutlineAgeGroup   | 79.8       | **16.0**            | 17.0           | 82.8    | **16.8**         | 17.5        | 
| DistalPhalanxOutlineCorrect    | 82.0       | 35.2            | **20.7**           | 79.8    | 35.8         | **25.3**        | 
| DistalPhalanxTW                | 74.8       | **9.8**             | 12.5           | 75.8    | **11.2**         | 12.2        | 
| Earthquakes                    | 78.6       | 51.2            | **48.8**           | 78.3    | 68.9         | **69.6**        | 
| ECG200                         | 89.0       | 61.0            | **46.0**           | 89.0    | 74.0         | **66.0**        | 
| ECG5000                        | 93.5       | 76.1            | **36.4**           | 93.9    | 90.0         | **88.0**        | 
| ECGFiveDays                    | 96.2       | 30.2            | **3.9**            | 99.0    | 51.2         | **31.4**        | 
| ElectricDevices                | 73.5       | 48.6            | **31.2**           | 70.9    | 50.3         | **48.9**        | 
| FaceAll                        | 85.5       | 76.7            | **72.5**           | 95.7    | 90.2         | **89.6**        | 
| FaceFour                       | 95.5       | 71.6            | **43.2**           | 92.0    | 71.6         | **70.5**        | 
| FacesUCR                       | 95.3       | 79.4            | **76.1**           | 94.7    | **86.4**         | 85.9        | 
| FISH                           | 97.7       | 12.6            | **4.0**            | 96.0    | 12.6         | **9.7**         | 
| FordA                          | 91.8       | 33.9            | **21.6**           | 90.1    | 59.6         | **57.3**        | 
| FordB                          | 91.1       | 27.8            | **14.3**           | 88.2    | 70.0         | **67.7**        | 
| Gun_Point                      | 99.3       | 31.3            | **6.7**            | 100.0   | 62.0         | **16.0**        | 
| Ham                            | 80.0       | 21.0            | **20.0**           | 71.4    | **27.6**         | **27.6**        | 
| HandOutlines                   | 86.0       | **36.2**            | **36.2**           | 74.6    | **36.2**         | **36.2**        | 
| Haptics                        | 51.6       | 19.2            | **14.6**           | 48.7    | 18.8         | **17.9**        | 
| Herring                        | 64.1       | 43.8            | **35.9**           | 65.6    | 59.4         | **57.8**        | 
| InlineSkate                    | 37.8       | 14.9            | **12.5**           | 32.4    | **9.6**          | 11.1        | 
| InsectWingbeatSound            | 50.6       | 17.7            | **15.7**           | 39.3    | **11.5**         | 12.1        | 
| ItalyPowerDemand               | 95.9       | 92.5            | **91.6**           | 96.1    | 89.8         | **89.6**        | 
| LargeKitchenAppliances         | 90.4       | 74.7            | **65.3**           | 89.6    | 66.4         | **63.5**        | 
| Lighting2                      | 77.0       | **42.6**            | **42.6**           | 73.8    | 41.0         | **39.3**        | 
| Lighting7                      | 78.1       | 50.7            | **35.6**           | 80.8    | 57.5         | **54.8**        | 
| MALLAT                         | 96.6       | 33.0            | **4.6**            | 97.0    | 32.6         | **24.2**        | 
| Meat                           | 98.3       | 35.0            | **1.7**            | 81.7    | **1.7**          | 31.7        | 
| MedicalImages                  | 76.2       | 52.1            | **28.7**           | 77.9    | 60.9         | **57.6**        | 
| MiddlePhalanxOutlineAgeGroup   | 74.2       | 58.0            | **12.8**           | 72.8    | 62.0         | **54.0**        | 
| MiddlePhalanxOutlineCorrect    | 80.5       | 29.8            | **19.5**           | 80.7    | 25.8         | **20.2**        | 
| MiddlePhalanxTW                | 60.9       | **13.3**            | 14.5           | 58.4    | **21.1**         | 24.3        | 
| MoteStrain                     | 92.4       | 74.3            | **68.8**           | 93.4    | 80.5         | **77.4**        | 
| NonInvasiveFatalECG_Thorax1    | 94.6       | 5.5             | **2.4**            | 95.6    | 7.4          | **5.1**         | 
| NonInvasiveFatalECG_Thorax2    | 94.4       | 5.2             | **1.2**            | 95.6    | 4.4          | **1.6**         | 
| OliveOil                       | 86.7       | 20.0            | **3.3**            | 86.7    | **13.3**         | **13.3**        | 
| OSULeaf                        | 97.9       | 15.7            | **0.0**            | 98.3    | 17.4         | **6.6**         | 
| PhalangesOutlinesCorrect       | 85.7       | 36.8            | **16.2**           | 81.5    | 35.9         | **24.9**        | 
| Phoneme                        | 33.3       | 15.0            | **10.3**           | 32.1    | 21.0         | **15.5**        | 
| Plane                          | 100.0      | 81.0            | **56.2**           | 100.0   | 58.1         | **56.2**        | 
| ProximalPhalanxOutlineAgeGroup | 83.9       | 46.3            | **8.3**            | 81.5    | 46.8         | **9.8**         | 
| ProximalPhalanxOutlineCorrect  | 91.4       | 32.0            | **10.7**           | 91.1    | 35.7         | **20.6**        | 
| ProximalPhalanxTW              | 77.8       | **10.2**            | 11.8           | 81.0    | 15.0         | **13.0**        | 
| RefrigerationDevices           | 51.7       | 32.0            | **30.1**           | 50.7    | **38.4**         | 40.0        | 
| ScreenType                     | 60.8       | 31.7            | **25.9**           | 60.8    | 36.5         | **28.0**        | 
| ShapeletSim                    | 100.0      | 53.9            | **36.1**           | 75.6    | 60.0         | **58.3**        | 
| ShapesAll                      | 91.7       | 5.2             | **1.0**            | 89.5    | 6.7          | **6.3**         | 
| SmallKitchenAppliances         | 78.9       | 40.5            | **21.9**           | 78.7    | 47.5         | **28.8**        | 
| SonyAIBORobotSurface           | 96.8       | 83.9            | **82.2**           | 96.0    | 85.0         | **84.2**        | 
| SonyAIBORobotSurfaceII         | 98.6       | 89.2            | **88.7**           | 98.1    | **91.5**         | 91.6        | 
| StarLightCurves                | 97.2       | 58.8            | **57.7**           | 96.6    | 73.0         | **60.1**        | 
| Strawberry                     | 96.2       | 21.9            | **3.8**            | 95.8    | 14.4         | **13.7**        | 
| SwedishLeaf                    | 95.4       | 31.2            | **16.0**           | 97.3    | 34.6         | **30.4**        | 
| Symbols                        | 92.7       | 36.6            | **12.9**           | 94.3    | 58.4         | **28.3**        | 
| synthetic_control              | 100.0      | 94.3            | **94.0**           | 98.3    | **94.7**         | 95.3        | 
| ToeSegmentation1               | 96.9       | 62.3            | **46.9**           | 96.1    | 63.2         | **57.5**        | 
| ToeSegmentation2               | 91.5       | 63.8            | **53.8**           | 90.8    | 54.6         | **52.3**        | 
| Trace                          | 100.0      | 58.0            | **52.0**           | 100.0   | **47.0**         | 52.0        | 
| TwoLeadECG                     | 100.0      | 5.3             | **0.4**            | 100.0   | 13.0         | **5.2**         | 
| Two_Patterns                   | 100.0      | 98.2            | **96.7**           | 86.8    | 82.9         | **82.6**        | 
| UWaveGestureLibraryAll         | 86.2       | 21.8            | **7.1**            | 81.7    | 25.3         | **22.3**        | 
| uWaveGestureLibrary_X          | 78.0       | 32.1            | **11.1**           | 75.7    | 32.7         | **27.2**        | 
| uWaveGestureLibrary_Y          | 66.7       | 27.7            | **14.9**           | 63.9    | 29.6         | **22.4**        | 
| uWaveGestureLibrary_Z          | 75.0       | 37.0            | **14.0**           | 72.0    | 27.1         | **21.0**        | 
| wafer                          | 99.8       | 86.6            | **7.3**            | 99.7    | **64.3**         | 81.2        | 
| Wine                           | 61.1       | **38.9**            | **38.9**           | 55.6    | **38.9**         | **38.9**        | 
| WordsSynonyms                  | 62.5       | 15.7            | **13.5**           | 55.0    | **9.7**          | 12.7        | 
| Worms                          | 64.6       | 27.6            | **19.9**           | 66.9    | 27.1         | **23.8**        | 
| WormsTwoClass                  | 74.6       | 45.3            | **31.5**           | 74.6    | 55.8         | **44.8**        | 
| yoga                           | 87.2       | 45.4            | **12.8**           | 84.1    | 44.9         | **19.2**        | 


## Reference

If you re-use this work, please cite:

```
@InProceedings{IsmailFawaz2019adversarial,
  Title                    = {Adversarial Attacks on Deep Neural Networks for Time Series Classification},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  booktitle                = {IEEE International Joint Conference on Neural Networks},
  Year                     = {2019}
}
```

## Acknowledgement

We would like to thank NVIDIA Corporation for the Quadro P6000 grant and the Mésocentre of Strasbourg for providing access to the cluster.
