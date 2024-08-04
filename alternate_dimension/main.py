import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from alternate_dimension.preprocessing import Preprocessing

train_val_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
counter = 1
mae_results = []
xgb_models = []
for train_index, val_index in (kf.split(train_val_data)):
    print(f'Fold:{counter}')
    train_data = train_val_data.loc[train_index]
    val_data = train_val_data.loc[val_index]
    x_train, y_train = Preprocessing(df=train_data, test=False).start()
    x_val, y_val = Preprocessing(df=val_data, test=False).start()

    xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.1)
    xgb_model.fit(x_train, y_train, early_stopping_rounds=5, eval_set=[(x_val, y_val)])
    predictions = xgb_model.predict(x_val)
    # TypeError: 'numpy.float64' object is not callable
    mean_absolute_error = mean_absolute_error(predictions, y_val)
    print("Mean Absolute Error: " + str(mean_absolute_error))
    mae_results.append(mean_absolute_error)
    xgb_models.append(xgb_model)
    counter += 1

x_test = Preprocessing(df=test_data, test=True).start()
# merge models or take best
y_test = xgb_models[0].predict(x_test)
print(y_test)
