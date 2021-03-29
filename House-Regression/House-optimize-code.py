# Author: Mohammadreza Ebrahimi
# Email: m.reza.ebrahimi1995@gmail.com

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# %% Reading Data

House = pd.read_csv('C:/Users/Al-Mahdi/Dropbox/Python/housing.csv')
House['income_ctg'] = pd.cut(House['median_income'],
                             bins=[0, 1.5, 3, 4.5, 6, np.inf],
                             labels=[1, 2, 3, 4, 5])
# %% Splitting data

st_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in st_split.split(House, House['income_ctg']):
    Train_set = House.loc[train_index]
    Test_set = House.loc[test_index]
# %%
for set_ in (Train_set, Test_set):
    set_.drop("income_ctg", axis=1, inplace=True)

# %% data cleaning
Housing = Train_set.drop('median_house_value', axis=1, inplace=False)
Housing_label = Train_set['median_house_value'].copy()

# %%
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class \
     CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            np.c_[rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(Housing.values)

# %%
Housing_num = Housing.drop('ocean_proximity', axis=1, inplace=False)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

Housing_num_tr = num_pipe.fit_transform(Housing_num)

num_attribs = list(Housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipe, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])

Housing_prepared = full_pipeline.fit_transform(Housing)
# %% Training model
lin_reg = LinearRegression()
lin_reg.fit(Housing_prepared, Housing_label)

# %%
House_prediction = lin_reg.predict(Housing_prepared)
lin_mse = mean_squared_error(House_prediction, Housing_label)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# %% Alternate model ... DecisionTree

tree_reg = DecisionTreeRegressor()
tree_reg.fit(Housing_prepared, Housing_label)

House_prediction = tree_reg.predict(Housing_prepared)
tree_mse = mean_squared_error(House_prediction, Housing_label)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# %% a method for testing train data in the model ( Linear or DecisionTree ) is Cross Validation (cv) and get the
# "mean_squared_error"  for every fold (cv)

scores = cross_val_score(tree_reg, Housing_prepared, Housing_label,
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_score = np.sqrt(-scores)


def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())


display_scores(tree_rmse_score)
# %%
lin_scores = cross_val_score(lin_reg, Housing_prepared, Housing_label,
                             scoring='neg_mean_squared_error', cv=10)
lin_rmse_score = np.sqrt(-lin_scores)
display_scores(lin_rmse_score)

# %% third model; RandomForest

forest_reg = RandomForestRegressor()
forest_reg.fit(Housing_prepared, Housing_label)

House_prediction = forest_reg.predict(Housing_prepared)
forest_mse = mean_squared_error(House_prediction, Housing_label)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# %%

forest_scores = cross_val_score(forest_reg, Housing_prepared, Housing_label,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# %%
# The Worst Model
from sklearn.preprocessing import PolynomialFeatures

Housing_poly = PolynomialFeatures(2)
Housing_prepared2 = Housing_poly.fit_transform(Housing_prepared, Housing_label)
lin_reg.fit(Housing_prepared2, Housing_label)
poly_scores = cross_val_score(lin_reg, Housing_prepared2, Housing_label,
                              scoring="neg_mean_squared_error", cv=10)
poly_scores_rmse = np.sqrt(-poly_scores)
display_scores(poly_scores_rmse)

# %% Among these Four models (Linear, Tree, poly, Forest) the later provides us best
# fitted results with the lowest error,
# so we choose it as our model and evaluate the hyperparameters (Fine-Tune)
# In order to get more precise prediction, it is needed to Fine-Tune model by using hyperparameters (Tuning the
# hyperparameters  is the most important part of training model)
# Hyperparameters for RandomForest model


param_grid = [
    # Try 20 (4*5) combination of hyperparameters
    {'n_estimators': [3, 10, 30, 40], 'max_features': [2, 3, 4, 5, 6]},
    # Try bootstrap as False with 12 combination
    {'bootstrap': [False], 'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4, 5]},
]
forest_reg = RandomForestRegressor(random_state=42)
# Train across 5 fold, that is a total of (16 + 12)*5 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)
grid_search.fit(Housing_prepared, Housing_label)

# %%
print(grid_search.best_params_)
# print(grid_search.best_estimator_)

# %%
cvres = grid_search.cv_results_
for mean_scores, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_scores), params)

# %%
cvres.keys()

# %%
final_model = grid_search.best_estimator_
X_test = Test_set.drop('median_house_value', axis=1, inplace=False)
y_test = Test_set['median_house_value'].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_prediction = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(final_prediction, y_test)
final_rmse = np.sqrt(final_mse)
final_rmse
# %%
