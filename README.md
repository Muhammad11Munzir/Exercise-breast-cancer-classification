# Exercise 1-breast-cancer-classification
## Methodology
In this project, the neural network model is built with TensorFlow Keras functional API framework. Modules used in this project include:
* numpy
* pandas
* tensorflow
* matplotlib.pyplot
* sklearn.model_selection.train_test_split
* sklearn.preprocessing

## STEP 1: Prepare that data
Numpy and pandas imported in order to extract data from the csv file. After import the data, the data was inspected with pandas.DataFrame.info method in order to find out is there any missing value, also to confirm the data types. After inspection, there is a column name "Unnamed: 32" with no row value. This column was dropped using pandas.DataFrame.drop method. The data is complete without missing value and the features are in float64 data type.

## STEP 2: Split the data into features and label
After inpection of the data. The label for this model is column "diagnosis". The "diagnosis" column contain data with data type of string. There are only 2 unique value where "M" represent malignant and 'B' represent benign. The label was encoded with one-hot encoding using pandas.get_dummies method. 

After one-hot encoding, "diagnosis" column transformed into 2 column with the name "diagnosis_D" and "diagnosis_M". Since a value of 0 in "diagnosis_M" can represent value 1 in "diagnosis_D", I added drop_first=True when I use pandas.get_dummies method as it helps in reducing the extra column created during dummy variable creation.

## STEP 3: Train Test Split and Standardization
In this step, the train and test data prepared using train_test_split method from sklearn.model_selection module. THe standardization of data is done by using StandardScaler method from sklearn.preprocessing module.

## STEP 4: Design of neural network
 
#### The Neural Network model as following table
| Layer (type)          | Output Shape | Activation Function | Param # |
|-----------------------|--------------|---------------------|---------|
| input_11 (InputLayer) | [(None, 30)] |                     | 0       |
| dense_31 (Dense)      | (None, 64)   | relu                | 1984    |
| dense_32 (Dense)      | (None, 32)   | relu                | 2080    |
| dense_33 (Dense)      | (None, 16)   | relu                | 528     |
| dense_34 (Dense)      | (None, 2)    | softmax             | 34      |

## STEP 5: Training of model
The model trained by using batch size of 32 with the epoch of 128. EarlyStopping with a patience of 5 added to the training, which cause the training stopped at epoch number 14/128.

## Result
Model accuracy shown in the following graph:

![Model Accuracy!](/model_accurancy.png "Model Accuracy")
