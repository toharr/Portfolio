from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import ElasticNet
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def for_floor(date_string):
    if date_string=="קומת קרקע" or date_string=="קומת מרתף":
        return int(0)
    elif type(date_string)is float:
        return (np.nan)
    else:
        if (re.findall("[0-9]+",date_string))==[]:
            return int(date_string)
        else:
            return int(re.findall("[0-9]+",date_string)[0])


def for_floor_out_of(date_string):
    if date_string=="קומת קרקע" or date_string=="קומת מרתף":
        return int(0)
    elif type(date_string)is float:
        return (np.nan)
    else:
        if (re.findall("[0-9]+",date_string))==[]:
            return int(date_string)
        elif len(re.findall("[0-9]+",date_string))==1:
            return int(0)
        else:
            return int(re.findall("[0-9]+",date_string)[1])
def to_date(date_string):
    if type(date_string) is float:
        return "not_defined"
    else:
        if str(date_string)=="מיידי":
            return "not_defined"
        elif "גמיש" in str(date_string):
            return "flexible"
        elif str(date_string)=="לא צויין":
            return "not_defined"
        else:
            date_format = "%d/%m/%Y"

            datetime_obj = datetime.strptime(date_string, date_format).date()
            date_now=datetime.today().date()
            gap=(datetime_obj-date_now).days
            if gap<=182:
                return "less_than_6 months"
            elif gap>=365:
                return "above_year"
            else:
                return "months_6_12"    
def for_area1(for_street):
    if type(for_street)is float:
        return(np.nan)
    else:
        if (re.findall(",(.*)",for_street))==[]:
            return(for_street)
        else:
            return(re.findall(",(.*)",for_street.get_text())[0])
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
def for_rooms(for_room):
    if type(for_room) is float:
        return(for_room)#np.nan
    else:
        try:
            return(re.findall("([0-9.]+)",for_room)[0])#להוריד גרש אחרי המספר
        except:
            try:
                return(re.findall("([0-9.]+)",for_room)[0])#[0-9]+.[0-9]*
            except:
                return(for_room)
def for_area(for_area):
    if type(for_area) is float:
        return(np.nan)
    else:
        try:
            return(re.findall("([0-9]+.[0-9]).*",for_area)[0])
        except:
            try:
                return(re.sub(r'[^\d]','',for_area))
            except:
                return(for_area)

def prepare_data():
    url = 'C:\\Users\\User\\Downloads\\output_all_students_Train_v10.csv'
    df = pd.read_csv(url)
    df["Street"]=df["Street"].apply(lambda x:for_street1(x))
    df["city_area"]=df["city_area"].apply(lambda x:for_area1(x))
    df["price"]=df["price"].apply(lambda x:for_price(x))
    df["entranceDate "]=df["entranceDate "].apply(lambda x:to_date(x))
    df["room_number"]=df["room_number"].apply(lambda x:for_rooms(x))
    df["Area"]=df["Area"].apply(lambda x:for_area(x))
    df["floor"]=df["floor_out_of"].apply(lambda x:for_floor(x))
    df["floor_of"]=df["floor_out_of"].apply(lambda x:for_floor_out_of(x))
    
    df[['hasElevator ','hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ','hasMamad ','handicapFriendly ']] = df[['hasElevator ','hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ','hasMamad ','handicapFriendly ']].replace({True: 1, False: 0})
    words_to_no = ["FALSE",'אין', 'no',"False", "לא"]
    df[['hasElevator ','hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ','hasMamad ','handicapFriendly ']] = df[['hasElevator ','hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ','hasMamad ','handicapFriendly ']].replace(words_to_no, 1, regex=True)
    words_to_yes = ['יש', 'yes',"TRUE","True", "כן"]
    df[['hasElevator ','hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ','hasMamad ','handicapFriendly ']] = df[['hasElevator ','hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ','hasMamad ','handicapFriendly ']].replace(words_to_yes, 0, regex=True)
    df['description '] = df['description '].str.replace('[,"!''?/״]', '')
    df['Street'] = df['Street'].str.replace('[,"!''?/״]', '')
    df=df.dropna(subset=['hasElevator ','hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ','hasMamad ','handicapFriendly '])
    df.dropna(subset=["price"])
    df= df[~df["price"].apply(lambda x: isinstance(x, float))]
    df["price"]=df["price"].astype(int)
    df["type"]=df["type"].astype(str)
    
    df=df.dropna(subset=["price"])
    df=df.drop(["number_in_street","city_area","floor_out_of","description ","publishedDays "],axis=1)
    
    #fill floor whith the mean un the same city and type
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    df['floor'] = df.groupby(['City', 'type'])['floor'].transform('mean').fillna(method='ffill')
    df['floor'] = df['floor'].astype(int)
    #fill floor_of whith the mean un the same city and type
    df['floor_of'] = pd.to_numeric(df['floor_of'], errors='coerce')
    df['floor_of']=df.groupby(["City","type"])["floor_of"].transform('mean').fillna(method='ffill')
    df['floor_of']=df['floor_of'].astype(int)
    df=df.drop("floor_of",axis=1)
    df= df[~df["Street"].apply(lambda x: isinstance(x, float))]
    
    df['num_of_images'] = pd.to_numeric(df['num_of_images'], errors='coerce')
    df['num_of_images']=df.groupby(["City","type","room_number"])["num_of_images"].transform('mean').fillna(method='ffill')
    df['num_of_images']=df['num_of_images'].astype(int)
    df["room_number"] = pd.to_numeric(df["room_number"], errors='coerce')
    df["Area"] = pd.to_numeric(df["Area"], errors='coerce')
    
    df["Area"] = df.groupby(["City","type"])["Area"].transform(lambda x: x.fillna(x.mean()))
    df=df.dropna(subset=["Area"])
    df["condition "].replace('חדש', 'משופץ', inplace=True)
    df["condition "].replace('FALSE', 'לא צויין', inplace=True)
    
    data = df.copy()
    data=data[data['condition '].isna()==False]
    # Separate the features (price and street) from the target variable (condition)
    X = data[['price']]
    y = data['condition ']
    scaler = MinMaxScaler()
    X['price'] = scaler.fit_transform(X['price'].values.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    
    data = df.copy()
    train_data = data[data['condition '].notnull()]
    test_data = data[data['condition '].isnull()]
    X_train = train_data[['price']]
    y_train = train_data['condition ']
    X_test = test_data[['price']]
    scaler = MinMaxScaler()
    X_train['price'] = scaler.fit_transform(X_train['price'].values.reshape(-1, 1))
    X_test['price'] = scaler.transform(X_test['price'].values.reshape(-1, 1))
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    predicted_conditions = knn.predict(X_test)
    test_data['condition '] = predicted_conditions
    filled_data = pd.concat([train_data, test_data])
    df=filled_data.copy()

    df_ohe = df[["condition "]]
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder = one_hot_encoder.fit(df_ohe)
    ohelabels = one_hot_encoder.transform(df_ohe).toarray()
    df_ohe = pd.DataFrame(ohelabels, columns=["Requires_renovation","old","Not_mentioned","renovation","save"])
    
    df["Requires_renovation"]=list(df_ohe["Requires_renovation"])
    df["old"]=list(df_ohe["old"])
    df["Not_mentioned"]=list(df_ohe["Not_mentioned"])
    df["renovation"]=list(df_ohe["renovation"])
    df["save"]=list(df_ohe["save"])
    
    furniture_data = df[['furniture ']]
    encoder = LabelEncoder()
    furniture_encoded = encoder.fit_transform(furniture_data.values.ravel())
    furniture_encoded[furniture_data.values.ravel() == 'אין'] = 0
    furniture_encoded[furniture_data.values.ravel() == 'לא צויין'] = 1
    furniture_encoded[furniture_data.values.ravel() == 'חלקי'] = 2
    furniture_encoded[furniture_data.values.ravel() == 'מלא'] = 3
    df['furniture_encoded'] = furniture_encoded
    
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
    df[["city_avg","furniture_encoded","save","renovation","Not_mentioned","old","Requires_renovation","floor","handicapFriendly ","hasMamad ","hasBalcony ","hasAirCondition ","hasStorage ","hasBars ","hasParking ","hasElevator ","num_of_images","type_avg"]]=df[["city_avg","furniture_encoded","save","renovation","Not_mentioned","old","Requires_renovation","floor","handicapFriendly ","hasMamad ","hasBalcony ","hasAirCondition ","hasStorage ","hasBars ","hasParking ","hasElevator ","num_of_images","type_avg"]].astype(float)
    
   
    columns_for_vif = ["city_avg","furniture_encoded","save","renovation","Not_mentioned","old","Requires_renovation","floor","handicapFriendly ","hasMamad ","hasBalcony ","hasAirCondition ","hasStorage ","hasBars ","hasParking ","hasElevator ","num_of_images","type_avg"]
    vif_data = df[columns_for_vif]
    vif = pd.DataFrame()
    vif["Column"] = columns_for_vif
    vif["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    vif_corr = vif_data.corrwith(df['price']).abs().sort_values(ascending=False)
    selected_columns = vif_corr[vif_corr > 0.5].index
    
    df=df.drop("Requires_renovation",axis=1)
    df['City']=df['City'].astype("category").cat.codes
    df["type"]=df["type"].astype("category").cat.codes
    df["room_number"] = pd.to_numeric(df["room_number"], errors='coerce')
    
    data =df[["Area","City","type","room_number"]]#,"type"
    k = 3  # מספר השכנים הקרובים שישומו
    imputer = KNNImputer(n_neighbors=k)
    data_imputed = imputer.fit_transform(data)
    df_new = pd.DataFrame(data_imputed,columns=["Area","City","type","room_numbers"])#[:, 1]
    df["room_number"]=data_imputed[:, 3]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    model = SVC()
    model.fit(X_train_normalized, y_train)
    y_pred = model.predict(X_test_normalized)
    accuracy = accuracy_score(y_test, y_pred)
    # Print the accuracy

    grouped_data3 =df.groupby(["Street","City"])[["price"]].mean()
    average_of_averages3 = grouped_data3.mean()
    df_merged3 = pd.merge(df, grouped_data3, on='Street', suffixes=('', '_avg3')).reset_index()
    df_merged3=df_merged3.drop(["index"],axis=1)
    df=df_merged3.copy()
    df["street_avg"]=df["price_avg3"].apply(lambda x: x/average_of_averages3)
    
    df["price_val"]=df["price"]
    
        # Select the columns for VIF calculation

    
    scaler = MinMaxScaler()
    reshaped_prices = df["Area"].values.reshape(-1, 1)
    scaler.fit(reshaped_prices)
    scaled_prices = scaler.transform(reshaped_prices)
    df["Area_val"] = scaled_prices
    df=df.drop(["hasElevator ","handicapFriendly ","hasStorage "],axis=1)
    df=df[["floor","old","save","Not_mentioned","renovation","furniture_encoded",'room_number', 'Area_val','city_avg',"type_avg","street_avg","num_of_images","hasParking ","hasBars ","hasAirCondition ","hasBalcony ","hasMamad ","price"]]

    return (df)
