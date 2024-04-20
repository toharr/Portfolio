import pickle
from madlan_data_prep import prepare_data
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from IPython.display import display
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# Load the training data

# data =pd.read_excel('output_all_students_v9.xlsx')
# train_data = pd.DataFrame(data)
url = 'C:\\Users\\User\\Downloads\\output_all_students_Train_v10.csv'
train_data = pd.read_csv(url)
df = prepare_data()

# פיצול הנתונים למשתנים תלותיים ולמשתנה תוצאה
X = df[["floor","old","save","Not_mentioned","renovation","furniture_encoded",'room_number', 'Area_val','city_avg',"type_avg","street_avg","num_of_images","hasParking ","hasBars ","hasAirCondition ","hasBalcony ","hasMamad "]]
y = df['price']

# טיפוח המשתנים הקטגוריים

# # חיפוש חוגר (Grid Search)
param_grid = {'alpha': [0.1, 0.5, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}
grid_search = GridSearchCV(ElasticNet(), param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X, y)
best_params_grid = grid_search.best_params_
#print("Best params (Grid Search):", best_params_grid)

model = ElasticNet(alpha=best_params_grid['alpha'], l1_ratio=best_params_grid['l1_ratio'])
model.fit(X, y)
y_pred = model.predict(X)
pickle.dump(model, open("trained_model.pkl","wb"))
