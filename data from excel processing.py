"""import pandas as pd
import numpy as np
dataset = pd.read_csv("C:\python codes\machinelearningdata.xlsx");
print (dataset)

x=data.Ploc[: , : -1]
print("###")
print(x)
y=data.iloc[: , 4:5]
print(y)

from sklearn.preprocessing import labelencoder

x=x.apply(x.labelencoder().fit_transform)
print(x)

from sklearn.tree import DecisionTreeClassifier

Dec_tree = DecisionTreeClassifier()
Dec_tree = fit(x.iloc[: , 0:4], 4)

x_in = np.array([1,1,0,0])

y_pred = Dec_tree.predict([x_in])
print(y_pred)

#splitting tree dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x)

#splitting training and test set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, testsize=1/4)

from sklearn.tree import DecisionTreeClassifier
Dec_tree = DecisionTreeClassifier()
Dec_tree.fit(x_train, y_train)
y_pred = Dec_tree.predict(x_test)
print(y_pred)"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Read the CSV file
dataset = pd.read_csv("C:\python codes\machinelearningdata.xlsx")

# Select features (excluding the last column)
x = dataset.iloc[:, :-1]  # Use dataset.iloc for consistency

# Option 1: Label Encoding for categorical features (if applicable)
le = LabelEncoder()  # Create LabelEncoder instance
for col in x.select_dtypes(include=['object']):
    x[col] = le.fit_transform(x[col])

# Option 2: One-Hot Encoding for categorical features with many unique values
# onehot_encoder = OneHotEncoder(sparse=False)
# x = onehot_encoder.fit_transform(x)

# Split data into training and testing sets (test_size=0.25 for 25% split)
x_train, x_test, y_train, y_test = train_test_split(x, dataset.iloc[:, -1], test_size=0.25)

# Create and train the decision tree classifier
Dec_tree = DecisionTreeClassifier()
Dec_tree.fit(x_train, y_train)

# Make prediction on a new data point
x_in = np.array([1, 1, 0, 0])
y_pred = Dec_tree.predict([x_in])
print(y_pred)

