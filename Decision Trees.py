import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

path = r'C:\Users\santo\Desktop\ML Course\Part 2 - Regression\Section 8 - Decision Tree Regression\Python\Position_Salaries.csv'

df = pd.read_csv(path)
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
# no feature scaling needed

dec = DecisionTreeRegressor(random_state=0)
dec = dec.fit(x,y)

Rand= RandomForestRegressor(n_estimators= 10, random_state=0)
Rand = Rand.fit(x,y)


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(-1,1)
y_pred = dec.predict(x_grid)
y_rand = Rand.predict(x_grid)
plt.scatter(x,y, color= 'red')
plt.plot(x_grid,y_rand, color = 'blue')
plt.plot(x_grid,y_pred, color= 'green')
plt.title('Decision Tree vs Random Forrest')
plt.show()

