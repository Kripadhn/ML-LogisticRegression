import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("fraud_data.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3, random_state=42)

# Train the model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict the fraud status
y_pred = clf.predict(X_test)
print("Predicted fraud status:", y_pred)
