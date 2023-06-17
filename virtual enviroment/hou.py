import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

df = pd.read_csv(
    r'housing_test.csv')

columns = ['Neighborhood', 'Condition1', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr']
df = df[columns]

X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

pickle.dump(gbr, open('gbr.pkl', 'wb'))