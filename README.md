#Project Name: Analyzing Covid-19 Trends Using plots


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
  
plt.style.use('ggplot') 
start_date = '2020-01-01'  
num_days = 365
dates = pd.date_range(start_date, periods=num_days)  
cases = np.random.randint(100, 5000, size=num_days)
df = pd.DataFrame({"Date": dates, "Cases": cases})  
start = pd.to_datetime(start_date)
df['Days'] = (df['Date'] - start).dt.days  
train_split_date = '2020-09-01'
train = df[df['Date'] < train_split_date]
test = df[df['Date'] >= train_split_date]
X_train = train['Days'].values.reshape(-1,1)   
y_train = train['Cases'].values 
model = LinearRegression()
model.fit(X_train, y_train)
X_test = test['Days'].values.reshape(-1,1)   
pred = model.predict(X_test)
plt.plot(test['Date'], test['Cases'], label='Actual')
plt.plot(test['Date'], pred, label='Predicted')  
plt.xticks(rotation=90)
plt.legend()  
plt.show()

print("Executed Successfully!")
