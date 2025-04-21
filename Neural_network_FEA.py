#Thesis code Neural Network- Finite Element Dataset
#Author- Pushkar Wadagbalkar
#************************************************************************************

#importing all the required python libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#importing training dataset and target files
dataset = pd.read_csv('Finite_element_entire_data_set.csv')
X = dataset.iloc[:, 0:5]
Y = dataset.iloc[:, 5].values


#Column Transfer and one hot encoding for categorical features
ct=ColumnTransformer(transformers=[("oh",OneHotEncoder(),[2,3,4])],remainder="passthrough")
X= ct.fit_transform(X)

fig, (ob1,ob2)= plt.subplots(ncols=2 , figsize= (10,10))
ob1.set_title('Before Scaling')
sns.kdeplot(dataset['Time'],ax=ob1)
sns.kdeplot(dataset['Velocity'],ax=ob1)

#Feature scaling using MinMaxScaler
sc = preprocessing.MinMaxScaler(feature_range=(0,1))
X[:,[10,11]] = sc.fit_transform((X[:, [10,11]]))
Z= pd.DataFrame(X)

#Plotting the KDE plot for data after feature scaling
scaled = Z.iloc[:,:]
ob2.set_title('After Scaling')
sns.kdeplot(scaled[10],ax=ob2)
sns.kdeplot(scaled[11],ax=ob2)

#Splitting the training data from the unseen inputs for which predictions are to be made
print(Z.shape)
Train = Z.iloc[:3041]      # 0 to 3039 → total 3040 rows
Y_train = Y[:3041]         # match the same rows in target
Input = Z.tail(1)          # safe way to grab the last row (3040)



#Splitting the data into training, validation and testing datasets
train_x, test_x, train_y, test_y = train_test_split(Train, Y, test_size=0.20, random_state=415)
validation_x, testing_x, validation_y, testing_y = train_test_split(test_x,test_y,test_size=0.50, random_state=415)

#importing and implementing Neural network regression model
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(12,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))


model.add(Dense(1,))
model.compile( loss='mse', optimizer='adam')

#Compiling the model with early stopping algorithm
#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
#Computing training and validation loss for the network
#history= model.fit(train_x,train_y,validation_data=(validation_x,validation_y),callbacks=[monitor],verbose=2,epochs=1000)
history= model.fit(train_x,train_y,validation_data=(validation_x,validation_y),verbose=2,epochs=500)

history_dict=history.history
loss_values= history_dict['loss']
val_loss_values=history_dict['val_loss']

#Getting predictions for training dataset
y_train_pred =model.predict(train_x)
#Getting predictions for validation dataset
y_validation_pred= model.predict(validation_x)
#Getting predictions for testing dataset
final_result= model.predict(testing_x)
#Getting predictions for unseen inputs
Unseen_result = model.predict(Input)

#Calculating relative error for validation dataset
Error= ((validation_y-y_validation_pred)/validation_y)

def average(Error):
    return sum(abs(Error))/len(Error)

Relative_Error= average(Error)

#Calculating relative error for testing dataset
Error_testing= ((testing_y-final_result)/testing_y)

def average(Error_testing):
    return sum(abs(Error_testing))/len(Error_testing)

Testing_Relative_Error= average(Error_testing)



mae_val = mean_squared_error(validation_y, y_validation_pred)
r2_val = r2_score(validation_y, y_validation_pred)

print("Validation MSE:", mae_val)
print("Validation R²:", r2_val)















