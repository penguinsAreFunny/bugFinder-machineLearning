import joblib
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

def model(x_train, y_train):
   print("Grid search: finding best hyperparameter for DecisionTreeClassifier...")
   # Grid search: finding best hyperparameters
   parameters = {
       "C": np.logspace(-3,3,7), 
       "penalty":["l1","l2"], #, "none", "elasticnet"],
       "max_iter": [100, 1000, 2500, 5000]
   }
   clf = LogisticRegression()
   grid_cv = GridSearchCV(clf, parameters, cv=10, n_jobs=-1)
   grid_cv.fit(x_train, y_train)
   print(f"Parameters of best model: {grid_cv.best_params_}")
   print(f"Score of best model: {grid_cv.best_score_}")

   # Use best hyperparameters to train model
   clf = LogisticRegression(C=grid_cv.best_params_["C"], penalty=grid_cv.best_params_["penalty"], max_iter=grid_cv.best_params_["max_iter"]) 
   clf.fit(x_train, y_train)

   model_name = "LogisticRegression" 
   joblib.dump(clf, pathlib.Path(__file__).parent.joinpath("trained_model_dumps").joinpath(model_name + ".model").resolve())
   return clf