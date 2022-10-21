#import the libraries
from cmath import nan
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split


#import the data
path = 'C:/Users/33646/Documents/GitHub/Portefolio/Personnal-Project/Car_Price_prediction/cars_data.csv'
df = pd.read_csv(path)

df = df.dropna()
df2 = pd.get_dummies(df, columns=['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style'])
X = df2.drop(['MSRP'], axis=1)
y = df2['MSRP']
X = np.array(X)
y = np.array(y)

#create input dictionary and list for categorical variables :
#for the variable Make
make_unique = df['Make'].unique()
make_dict = dict(zip(make_unique, range(len(make_unique))))
make_list = list(make_dict.keys())

#for the variable Model
model_unique = df['Model'].unique()
model_dict = dict(zip(model_unique, range(len(model_unique))))
model_list = list(model_dict.keys())

#for the variable Engine Fuel Type	
fuel_unique = df['Engine Fuel Type'].unique()
fuel_dict = dict(zip(fuel_unique, range(len(fuel_unique))))
fuel_list = list(fuel_dict.keys())


#for the variable Transmission Type
transmission_type_unique = df['Transmission Type'].unique()
transmission_type__dict = dict(zip(transmission_type_unique, range(len(transmission_type_unique))))
transmission_type_list = list(transmission_type__dict.keys())

#for the variable Driven_Wheels
driven_wheels_unique = df['Driven_Wheels'].unique()
driven_wheels_dict = dict(zip(driven_wheels_unique, range(len(driven_wheels_unique))))
driven_wheels_list = list(driven_wheels_dict.keys())

#for the variable Market Category
market_category_unique = df['Market Category'].unique()
market_category_dict = dict(zip(market_category_unique, range(len(market_category_unique))))
market_category_list = list(market_category_dict.keys())

#for the variable Vehicle Size
vehicle_size_unique = df['Vehicle Size'].unique()
vehicle_size_dict = dict(zip(vehicle_size_unique, range(len(vehicle_size_unique))))
vehicle_size_list = list(vehicle_size_dict.keys())

#for the variable Vehicle style
vehicle_style_unique = df['Vehicle Style'].unique()
vehicle_style_dict = dict(zip(vehicle_style_unique, range(len(vehicle_style_unique))))
vehicle_style_list = list(vehicle_style_dict.keys())


#create a function for filtering the model name correspond to it brand
def filter_model(make):
    model = df[df['Make'] == brand]['Model'].unique()
    return list(model)

#setting the streamlit app
import streamlit as st
#create the title
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")


#create a title of the streamlit app in the center
st.title("🚗 Car Price Prediction 🚗")

col1, col2 = st.columns(2)

#start taking inputs
brand_input = col1.selectbox("Select Brand", make_list,help = "From which brand the car is made")
#create a list of brand models with the brand selected as 1, and the other elements as 0

brand = make_dict[brand_input]
print(brand)
#for the variable Model with a loop and a if statement
model_input = col2.selectbox("Select Model", tuple(df.loc[df["Make"] == brand_input]['Model'].unique()), help = "Which model of the car is it")
model = model_dict[model_input]

#input for year
year = col1.slider("Select Year", min_value=1980, max_value=2020, value=2020, step=1)

#input for engine hp
engine_hp = col2.slider("Select Engine Horse Power", min_value=50, max_value=1000, value=50, step=1)

#input for engine cylinders
engine_cylinders = col1.slider("Select Engine Cylinders", min_value=1, max_value=16, value=4, step=1)

#input for number of doors
number_of_doors = col2.slider("Select Number of Doors", min_value=2, max_value=4, value=4, step=1)

#input for highway mpg
highway_mpg = col1.slider("Select Highway MPG", min_value=10, max_value=354, value=30, step=1)

#input for city mpg
city_mpg = col2.slider("Select City MPG", min_value=7, max_value=140, value=30, step=1)

#input for popularity
popularity = col1.slider("Select Popularity", min_value=1, max_value=6000, value=50, step=1)


fuel_input = col2.selectbox("Select Fuel Type", fuel_list, help = "Select the fuel type of the car")
fuel_type = fuel_dict[fuel_input]

transmission_type_input = col2.selectbox("Select Transmission Type", transmission_type_list, help = "Select the transmission type of the car")
transmission_type = transmission_type__dict[transmission_type_input]

driven_wheels_input = col2.selectbox("Select Driven Wheels", driven_wheels_list, help = "Select the driven wheels of the car")
driven_wheels = driven_wheels_dict[driven_wheels_input]


market_category_input = col1.selectbox("Select Market Category", market_category_list, help = "Select the market category of the car")
market_category = market_category_dict[market_category_input]

vehicle_size_input = col2.selectbox("Select Vehicle Size", vehicle_size_list, help = "Select the vehicle size of the car")
vehicle_size = vehicle_size_dict[vehicle_size_input]

vehicle_style_input = col1.selectbox("Select Vehicle Style", vehicle_style_list, help = "Select the vehicle style of the car")
vehicle_style = vehicle_style_dict[vehicle_style_input]

algorithm = st.selectbox("Select algorithm",["Linear Regression", "Decision Tree", "Random Forest Model","XG-Boost"], help = "Select the algorithm to use")




df = pd.get_dummies(df, columns=['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style'])
print(df)
#create an input array for prediction with the same size of X
#X has 15 features, but LinearRegression is expecting 873 features as input.
#So, we need to add dummy variables for the other 786 features.



#predict the price
if algorithm == "Linear Regression":
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(input_array)
elif algorithm == "Decision Tree":
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(X, y)
    pred = model.predict(input_array)
elif algorithm == "Random Forest Model":
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X, y)
    pred = model.predict(input_array)
elif algorithm == "XG-Boost":
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.fit(X, y)
    pred = model.predict(input_array)
else:
    st.write("Please select an algorithm")

#create a boutton for prediction
Predict = st.button("Predict")

if Predict:
    #pred = model.predict(input_array)
    if pred < 0:
        st.error("The input values must be irrelevant, try again by giving different values")
    pred = round(float(pred), 3)
    write = st.write("The predicted price is : ", pred)
    st.success(write)
    st.balloons()

st.header("🧭 Some infos about the project")
st.write("This project is a prediction of the car price based on the inputs given by the user.")