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

url = 'C:\\Users\\User\\Downloads\\output_all_students_Train_v10.csv'
df = pd.read_csv(url)
df=df[["City","Street","type","price","Area"]]

def for_street1(for_street):
    try:
        str(for_street)
    except:
        for_street=np.nan
    if type(for_street)is float:
        return np.nan
    else:
        if re.findall(r'^(.*?)(?:\d+|,)',for_street)==[]:
            return (for_street)
        else:
            return (re.findall(r'^(.*?)(?:\d+|,)',for_street)[0])


def for_price(for_price):
    if type(for_price) is float or for_price=="בנה ביתך":
        return(np.nan)
    else:
        if re.findall("([0-9]+.[0-9]*).*",for_price)==[]:
            return for_price
        else:
            return(re.sub(r'[^\d]', '', for_price))

def for_area(for_area):
    if type(for_area) is float:
        return(np.nan)
    else:
        try:
            return float(re.findall("([0-9]+.[0-9]).*",for_area)[0])
        except:
            try:
                return float(re.sub(r'[^\d]','',for_area))
            except:
                return (np.nan)

df=df.dropna()
df["Street"]=df["Street"].apply(lambda x:for_street1(x))
df["price"]=df["price"].apply(lambda x:for_price(x))
df=df.dropna(subset=["price"])
df["price"]=df["price"].astype(int)
df["Area"]=df["Area"].apply(lambda x:for_area(x))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming you have a DataFrame named df with a "price" column

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Reshape the price column to have the shape (n_samples, 1)
reshaped_prices = df["Area"].values.reshape(-1, 1)

# Fit the MinMaxScaler to the reshaped price data
scaler.fit(reshaped_prices)

# Transform the price column using the fitted scaler
scaled_prices = scaler.transform(reshaped_prices)

# Update the "price" column in the DataFrame with the scaled prices
df["Area_val"] = scaled_prices

grouped_data =df.groupby(["City"])[["price"]].mean()
average_of_averages = grouped_data.mean()
df_merged = pd.merge(df, grouped_data, on='City', suffixes=('', '_avg')).reset_index()
df_merged=df_merged.drop(["index"],axis=1)
df=df_merged.copy()
df["city_avg"]=df["price_avg"].apply(lambda x: x/average_of_averages)

grouped_data2 =df.groupby(["type"])[["price"]].mean()
average_of_averages2 = grouped_data2.mean()
df_merged2 = pd.merge(df, grouped_data2, on='type', suffixes=('', '_avg2')).reset_index()
df_merged2=df_merged2.drop(["index"],axis=1)
df=df_merged2.copy()
df["type_avg"]=df["price_avg2"].apply(lambda x: x/average_of_averages2)

grouped_data3 =df.groupby(["Street","City"])[["price"]].mean()
average_of_averages3 = grouped_data3.mean()
df_merged3 = pd.merge(df, grouped_data3, on='Street', suffixes=('', '_avg3')).reset_index()
df_merged3=df_merged3.drop(["index"],axis=1)
df=df_merged3.copy()
df["street_avg"]=df["price_avg3"].apply(lambda x: x/average_of_averages3)
df=df.drop_duplicates()

def the_city(city):
    return df[df.loc[:,"City"]==city]["city_avg"].mean()

def the_street(street,city):
    condit=df.loc[(df['Street'] == street) & (df['City'] ==city), ["street_avg"]].mean()
    if condit is np.nan:
        return df.loc[(df['Street'] == street) & (df['City'] ==city), ["street_avg"]].mean()
    else:
        return df[df.loc[:,"City"]==city]["street_avg"].mean()

def the_type(type1,city,street):
    condit=df.loc[(df['type'] == type1) & (df['City'] ==city)& (df['Street'] ==street),["type_avg"]].mean()
    if condit is np.nan:
        return condit
    else:
        return df[df.loc[:,"City"]==city]["type_avg"].mean()

def the_area(area,city,street):
    condit=df.loc[(df['Area'] == area) & (df['City'] ==city)& (df['Street'] ==street),["Area_val"]].mean()
    if condit is np.nan:
        return condit
    else:
        return df[df.loc[:,"City"]==city]["Area_val"].mean()

def the_condition(condition):
    save=0
    old=0
    Not_mentioned=0
    renovation=0
    if condition=="save":
        save=1
    elif condition=="old":
        old=1
    elif condition=="renovation":
        renovation=1
    else:
        Not_mentioned=1
    return [old,save,Not_mentioned,renovation]


def finall(floor,condition,type1,city,street,room_num,area,fruniture,num_of_images,hasParking,hasBars,hasAirCondition,hasBalcony,hasMamad):
    conf_re=the_condition(condition)
    to_type=the_type(type1,city,street)
    to_city=the_city(city)
    to_street=the_street(street,city)
    to_area=the_area(area,city,street)
    new_df={"floor":floor,"old":conf_re[0],"save":conf_re[1],
            "Not_mentioned":conf_re[2],"renovation":conf_re[3],"furniture_encoded":fruniture,
           "room_number":room_num,"Area_val":to_area,"city_avg":to_city,"type_avg":to_type,
           "street_avg":to_street,"num_of_images":num_of_images,"hasParking":hasParking,
           "hasBars":hasBars,"hasAirCondition":hasAirCondition,"hasBalcony":hasBalcony,"hasMamad":hasMamad}
    return new_df

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os


app = Flask(__name__)
rf_model = pickle.load(open('trained_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    city = request.form.get('city')
    street = request.form.get('street')
    area = float(request.form.get('Area'))
    type1 = request.form.get('type1')
    num_of_images = int(request.form.get('num_of_images'))
    room_number = int(request.form.get('room_number'))
    floor = int(request.form.get('floor'))
    condition = request.form.get('condition')
    furniture = int(request.form.get('furniture'))
    has_parking = int('hasParking' in request.form)
    has_bars = int('hasBars' in request.form)
    has_air_condition = int('hasAirCondition' in request.form)
    has_balcony = int('hasBalcony' in request.form)
    has_mamad = int('hasMamad' in request.form)

    data=finall(floor,condition,type1,city,street,room_number,area,furniture,num_of_images,has_parking,
                has_bars,has_air_condition,has_balcony,has_mamad)

    reshaped_data = np.array(list(data.values())).reshape(1, -1)

    # Make the prediction
    prediction = rf_model.predict(reshaped_data)

    return render_template('index.html', prediction_text='{}'.format(prediction))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)
