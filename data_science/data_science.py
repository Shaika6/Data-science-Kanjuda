import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from patsy import dmatrices
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df_data = pd.read_excel('airlines.xlsx')
df_data.dropna(inplace=True)
print(df_data.isnull().sum())

df_data['Journey_Date'] = pd.to_datetime(df_data['Journey_Date'], format='%d/%m/%Y')
df_data['Dep_Time'] = pd.to_datetime(df_data['Dep_Time']).dt.time
df_data['Arrival_Time'] = pd.to_datetime(df_data['Arrival_Time']).dt.time

df_data['Journey_day'] = df_data['Journey_Date'].dt.day
df_data['Journey_month'] = df_data['Journey_Date'].dt.month
df_data['Journey_year'] = df_data['Journey_Date'].dt.year
df_data['Dep_Time_hour'] = df_data['Dep_Time'].apply(lambda x: x.hour)
df_data['Dep_Time_minute'] = df_data['Dep_Time'].apply(lambda x: x.minute)
df_data['Arrival_Time_hour'] = df_data['Arrival_Time'].apply(lambda x: x.hour)
df_data['Arrival_Time_minute'] = df_data['Arrival_Time'].apply(lambda x: x.minute)

def dep_description(hour):
    if 4 <= hour < 8:
        return "Early Morning"
    elif 8 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 16:
        return "Noon"
    elif 16 <= hour < 20:
        return "Evening"
    elif 20 <= hour < 24:
        return "Night"
    else:
        return "Late Night"

df_data['dep_description'] = df_data['Dep_Time_hour'].apply(dep_description)


df_data['dep_description'].value_counts().plot(kind='bar')
plt.xlabel('Dep Description')
plt.ylabel('Number of Flights')
plt.title('Number of Flights by Departure Description')
plt.show()


df_data['Duration'] = pd.to_timedelta(df_data['Arrival_Time'].astype(str)) - pd.to_timedelta(df_data['Dep_Time'].astype(str))
df_data['Duration_hours'] = df_data['Duration'].dt.components.hours
df_data['Duration_mins'] = df_data['Duration'].dt.components.minutes
df_data['Duration_total_mins'] = df_data['Duration'].dt.total_seconds() / 60


x = df_data['Duration_total_mins']
y = df_data['Price']
plt.scatter(x, y)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')
plt.xlabel('Duration (minutes)')
plt.ylabel('Price')
plt.title('Duration vs Price with Regression Line')
plt.show()


sns.scatterplot(data=df_data, x='Duration_total_mins', y='Price', hue='Total_Stops')
plt.xlabel('Duration (minutes)')
plt.ylabel('Price')
plt.title('Duration vs Price by Number of Stops')
plt.show()


jet_airways = df_data[df_data['Airline'] == 'Jet Airways']
most_used_routes = jet_airways['Route'].value_counts().head(10)
most_used_routes.plot(kind='bar')
plt.xlabel('Routes')
plt.ylabel('Number of Routes')
plt.title('Most Used Routes by Jet Airways')
plt.show()
print(f"Most used route by Jet Airways: {most_used_routes.idxmax()}")


y, X = dmatrices('Price ~ Airline * Source * Destination * Total_Stops * dep_description * Journey_month * Journey_weekday', data=df_data, return_type='dataframe')
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R2 score for the model: {r2}")


y, X = dmatrices('Price ~ Duration_total_mins + Airline + Source + Destination', data=df_data, return_type='dataframe')
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2_new = r2_score(y, y_pred)
print(f"R2 score for the new model: {r2_new}")
