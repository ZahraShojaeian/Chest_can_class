# Lung Cancer Classification using VGG19 and Spatial Attention

This project utilizes a deep learning approach to classify lung cancer using chest CT scan images from the Kaggle dataset. The model is built upon the VGG19 architecture, enhanced with a custom Convolutional Neural Network (CNN) layer and a spatial attention mechanism to improve classification accuracy.

## Model Architecture

1. **Base Model**: VGG19 pre-trained on ImageNet.
2. **Spatial Attention Mechanism**: Applied to the output of VGG19 to focus on relevant features in the input images.
3. **Custom CNN Layers**: Additional convolutional, batch normalization, and pooling layers to further process the features.
4. **Fully Connected Layers**: Dense layers with dropout to prevent overfitting.

## Results

- **Test Loss**: 0.1959
- **Test Accuracy**: 92.86%




### MLflow and dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI= your URI

MLFLOW_TRACKING_USERNAME=your USERNAME \

MLFLOW_TRACKING_PASSWORD=your PASSWORD \

python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=  your URI

export MLFLOW_TRACKING_USERNAME= your USERNAME

export MLFLOW_TRACKING_PASSWORD= your PASSWORD 

```



### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag
## References

1. Kaggle Chest CT Scans Dataset: [Link to the dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/data)
2. Dina M. Ibrahim, Nada M. Elshennawy, Amany M. Sarhan, "Deep-chest: Multi-classification deep learning model for diagnosing COVID-19, pneumonia, and lung cancer chest diseases",
Computers in Biology and Medicine, Volume 132,2021, doi.org/10.1016/j.compbiomed.2021.104348.



