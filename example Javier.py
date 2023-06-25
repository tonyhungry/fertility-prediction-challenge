#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Classifier imports
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay


import pylab as plt
import pandas as pd
import numpy as np

from joblib import dump, load


# # 1. Read data

# In[4]:


# Import paths
path_train = "~/Downloads/Example data/LISS_example_input_data.csv"
path_outcome = "~/Downloads/Example data/LISS_example_groundtruth_data.csv"


# In[5]:


# Read data
original_data = pd.read_csv(path_train, encoding="cp1252", low_memory=False)
outcome = pd.read_csv(path_outcome, encoding="cp1252", low_memory=False)

# Drop observations where the outcome is missing
y_isna = outcome['new_child'].isnull()
data = original_data.loc[~y_isna]
outcome = outcome.loc[~y_isna]


# # 2. Split data into train and test
# First thing always, otherwise you risk overfitting

# In[6]:


# Select predictors: education, year of birth, gender, number of children in the household 
# You can do this automatically (not necessarily better): https://scikit-learn.org/stable/modules/feature_selection.html
keepcols = ['leeftijd2019','aantalki2019','partner2019','burgstat2019','woonvorm2019','woning2019','belbezig2019','oplmet2019','sted2019','brutohh_f2019','geslacht','ch19l004','ch19l018', 'ch19l021', 'ch19l022', 'ch19l126','ch19l133','ch19l159','ch19l160','ch19l161','ch19l162','ch19l163','ch19l229','cp19k010','cp19k026','cv19k012','cv19k053','cv19k101','cv19k125','cv19k126','cv19k130','cv19k140','cr19l089','cr19l134','cs19l079','cs19l105','cs19l436','cs19l435','cf19l014','cf19l025','cf14g034','cf19l136','cf19l131','cf19l129','cf19l130','cf19l133','cf19l134','cf19l183','cf19l198']
#a = ['apple']

data = data.loc[:, keepcols]


X_train, X_test, y_train, y_test = train_test_split(data,outcome,test_size=0.2, random_state=2023)
y_train = y_train["new_child"]
y_test = y_test["new_child"]


# # 3. Pre-process and model
# You may not want to include the preprocessing in the pipeline if it becomes too cumbersome
# 
# Make sure to use the scoring that you want to optimize in the search space

# In[7]:


# An example of a preprocessing apart from the pipeline
dict_kids = {'None': 0, 'One child': 1, 'Two children': 2, 'Three children': 3, 'Four children': 4, 'Five children': 5, 'Six children': 6}
X_train["aantalki2019"] = X_train["aantalki2019"].map(dict_kids)

# Create transformers
# Imputer are sometimes not necessary
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50))])

numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="mean")),('scaler', StandardScaler())])

# Use ColumnTransformer to apply the transformations to the correct columns in the dataframe
preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, selector(dtype_exclude=object)(X_train)),
        ('cat', categorical_transformer, selector(dtype_include=object)(X_train))])


# In[8]:

from sklearn.ensemble import GradientBoostingClassifier

# Create pipeline
model = Pipeline([
               ("preprocess", preprocessor),
               ("classifier", GradientBoostingClassifier())
               ]) 
                      
# Define the hyperparameters, this can include several classifiers, but will make it slow
# You can see different classifiers here: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
parameters = [
    {
        'classifier': [GradientBoostingClassifier()],
    }
    
]

# Perform hyperparameter tuning using cross-validation: https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers
# Scoring metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
# f1 = f1 of the class labeled as 1 (i.e. kids)
grid_search = GridSearchCV(model, parameters, cv=5, n_jobs=-1, scoring="f1", verbose=9) #n_jobs=-1 allows for multiprocessing
grid_search.fit(X_train, y_train)

# Keep best model (or define it from scratch with the best coefficients found)
best_model = grid_search.best_estimator_

best_model


# In[9]:


#Variable names in the data
best_model["preprocess"].get_feature_names_out()


# # Evaluate the model
# 
# Note: The results below are not for LogisticRegression, are for a different model

# In[10]:


X_test["aantalki2019"] = X_test["aantalki2019"].map(dict_kids)


# In[11]:


# Print ROC curve, it tells you how well you can balance false and true positives
RocCurveDisplay.from_predictions(
    y_test,
    best_model.predict_proba(X_test)[:, 1],
    color="cornflowerblue",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# In[12]:


# Create predictions
y_pred = best_model.predict(X_test)

# Report classification table
print(classification_report(y_test, y_pred))


# # Save models 

# In[13]:


import os
os.makedirs("../models", exist_ok=True)

# Dump model (don't change the name)
dump(best_model, "../models/model.joblib")

#### RUN UNTIL HERE!! ####

# # How the submission would look like, 

# In[14]:


def predict_outcomes(df):
    """Process the input data and write the predictions."""
    # Dictionary used
    dict_kids = {'None': 0, 'One child': 1, 'Two children': 2, 'Three children': 3, 'Four children': 4, 'Five children': 5, 'Six children': 6}
    
    # Keep 
    keepcols =  ['oplmet2019', 'gebjaar', 'geslacht', 'aantalki2019']
    results = df[["nomem_encr"]]
    
    df = df.loc[:, keepcols]
    df["aantalki2019"] = df["aantalki2019"].map(dict_kids)
                            
    # Load your trained model from the models directory
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
    model = load(model_path)

    # Use your trained model for prediction
    results.loc[:, "prediction"] = model.predict(df)

    #If you use predict_proba to get a probability and a different threshold
    #df["prediction"] = (df["prediction"] >= 0.5).astype(int)
    return results


# In[15]:


__file__ = './' #this is not needed outside juypter notebooks
predict_outcomes(original_data)


# # Interpretable AI
# 
# Note: Again, the result below are not for the LogisticRegression
# 
# There are other methods 
#     - tree-based models have an argument "best_model.feature_importances_"
#     - Other libraries: shap, lime, eli5
#     - Other plots, such as partial dependence plots: https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py

# In[16]:


from sklearn.inspection import permutation_importance

def print_sorted_importance(importances, columns, codebook=None):
    for imp, var in sorted(zip(importances, columns))[::-1]:
        if codebook is not None:
            print(f"{imp:2.3f}: {codebook[var]:50.50s}")
        else:
            print(f"{imp:2.3f}: {var:50.50s}")
        
r = permutation_importance(best_model, X_test, y_test,
                            n_repeats=100,
                            random_state=0)

print_sorted_importance(r["importances_mean"], X_test.columns)


# In[ ]:




