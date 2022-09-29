import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


df = pd.read_csv("Retention_CP.csv")

encoder = OrdinalEncoder()
final_data = encoder.fit_transform(df.drop(columns='cost'))

X = final_data
y = df['cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=1)

# Prediction
rfr_model = RandomForestRegressor()
rfr_model.fit(X_train, y_train)
y_pred = rfr_model.predict(X_test)

# Checking Accuracy
print("r2 score", r2_score(y_test, y_pred))

pickle.dump(rfr_model, open('rf_model.pkl', 'wb'))
pickle.dump(encoder, open('encoder.pkl', 'wb'))
