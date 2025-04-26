# BASED on Thesis code Neural Network- Finite Element Dataset Author- Pushkar Wadagbalkar
# Change Blame Kevin Schriml
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV


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



param_grid = {
    'min_samples_split': list(range(2, 15, 1)),

    'min_samples_leaf': list(range(1, 15, 1)),

    'max_depth': [None] + list(range(1, 10, 1)),
}

base_model = DecisionTreeRegressor(random_state=415)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(train_x, train_y)
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

model = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='best',
    max_depth=None,
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_weight_fraction_leaf = 0.0,
    random_state=415
)

model.fit(train_x, train_y)

#Predict and evaluate
preds = model.predict(validation_x)
mae = mean_absolute_error(validation_y, preds)
mse = mean_squared_error(validation_y, preds)
r2 = r2_score(validation_y, preds)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
#print(validation_y)
#print(preds)
print(f'r_squared: {r2:.3f}')


slope, intercept = np.polyfit(preds, validation_y, 1)
regression_x = np.array([min(preds), max(preds)])
regression_y = slope * regression_x + intercept
print(f"Regression equation: y = {slope:.2f}x + {intercept:.2f}")


x = np.asarray(preds)
y = np.asarray(validation_y)

fig2, ax = plt.subplots()

ax.plot(regression_x, regression_y, color='red', label=f'y = {slope:.2f}x + {intercept:.2f}')
ax.scatter(x, y, label='Data points')

ax.set_xlabel("Predicted Values m/s")
ax.set_ylabel("Actual Vales m/s")
ax.set_title("Actual vs Predicted Residual Velocity Decision Tree")

ax.grid(True, alpha=0.7)
plt.tight_layout()
plt.show()








