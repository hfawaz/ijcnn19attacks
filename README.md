# Adversarial Attacks on Deep Neural Networks for Time Series Classification
This is the companion repository for our paper also available on ArXiv titled "Adversarial Attacks on Deep Neural Networks for Time Series Classification". 
This paper has been accepted at the [IEEE International Joint Conference on Neural Networks (IJCNN) 2019](https://www.ijcnn.org/). 

## Approach 
![fgsm](https://github.com/hfawaz/ijcnn19attacks/blob/master/img/pert-example.png)

## Data 
The data used in this project comes from the [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data/UCR_TS_Archive_2015.zip), which contains the 85 univariate time series datasets. 

## Code

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
* [keras_contrib](https://www.github.com/keras-team/keras-contrib.git)

## Results 

The following animation shows how the accuracy and the time series shape variation with respect to the amount of perturbation allowed. 
![fgsm](https://github.com/hfawaz/ijcnn19attacks/blob/master/img/acc-ts.gif)

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
