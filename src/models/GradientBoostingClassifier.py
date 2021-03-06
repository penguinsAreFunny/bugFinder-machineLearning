import joblib
import pathlib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

def model(x_train, y_train):
   print("Grid search: finding best hyperparameter for DecisionTreeClassifier...")
   # Grid search: finding best hyperparameters
   parameters = {
       "loss": ["deviance", "exponential"],
       'n_estimators': [40, 80, 160],
       'criterion': ['friedman_mse'],
       'max_depth': [None, 10]
   }
   clf = GradientBoostingClassifier(verbose=1)
   grid_cv = GridSearchCV(clf, parameters, cv=10, n_jobs=-1)
   grid_cv.fit(x_train, y_train)
   print(f"Parameters of best model: {grid_cv.best_params_}")
   print(f"Score of best model: {grid_cv.best_score_}")

   # Use best hyperparameters to train model
   clf =  GradientBoostingClassifier(n_estimators=grid_cv.best_params_["n_estimators"], criterion=grid_cv.best_params_["criterion"], max_depth=grid_cv.best_params_["max_depth"]) 
   clf.fit(x_train, y_train)

   model_name = "GradientBoostingClassifier" 
   joblib.dump(clf, pathlib.Path(__file__).parent.joinpath("trained_model_dumps").joinpath(model_name + ".model").resolve())
   return clf