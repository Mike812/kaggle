import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from alternate_dimension.preprocessing import Preprocessing

# Read data from kaggle as dataframes
train_val_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
# Prepare cross validation of model predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
counter = 1
mse_results = []
xgb_models = []
acc_results = []
# Start cross validation
for train_index, val_index in (kf.split(train_val_data)):
    print(f'Fold: {counter}')
    train_data = train_val_data.loc[train_index]
    val_data = train_val_data.loc[val_index]
    print(f'Train data: {train_data.shape}')
    print(f'Validation data: {train_data.shape}')
    # Start preprocessing of data needed for modeling
    x_train, y_train = Preprocessing(df=train_data, test=False).start()
    x_val, y_val = Preprocessing(df=val_data, test=False).start()
    # Define and train model
    xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.1)
    xgb_model.fit(x_train, y_train, early_stopping_rounds=5, eval_set=[(x_val, y_val)], verbose=False)
    # Predict and evaluate model quality
    y_predictions = xgb_model.predict(x_val)
    mse = mean_squared_error(y_predictions, y_val)
    acc = accuracy_score(y_val, y_predictions)
    print("Mean squared error: " + str(mse))
    print("Model accuracy: " + str(acc))
    mse_results.append(mse)
    acc_results.append(acc)
    xgb_models.append(xgb_model)
    counter += 1

print("Final model:")
# Start preprocessing of test data
x_test = Preprocessing(df=test_data, test=True).start()
# Pick best model from cross validation
best_model_index = mse_results.index(min(mse_results))
print("Mean squared error of best model: " + str(mse_results[best_model_index]))
print("Accuracy of best model: " + str(mse_results[best_model_index]))
# Predict if passenger will be transported or not
y_test = xgb_models[best_model_index].predict(x_test)
print(y_test)
