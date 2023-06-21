import pandas as pd

data = pd.read_csv("datasets/LISS_example_input_data.csv", encoding='cp1252', low_memory=False)
outcome = pd.read_csv("datasets/LISS_example_groundtruth_data.csv")

selected_columns = ['leeftijd2019','lftdcat2019', 'lftdhhh2019', 'aantalki2019', 'partner2019', 'burgstat2019', 'woonvorm2019',
                   'woning2019', 'belbezig2019', 'brutoink2019', 'nettoink2019', 'brutocat2019', 'nettocat2019',
                   'oplzon2019', 'oplmet2019', 'oplcat2019', 'sted2019', 'werving2019']
features = data[selected_columns]

y_isna = outcome['new_child'].isnull()
X_isna = features.isnull().any(axis=1)
features = features.drop(features[y_isna | X_isna].index)
outcome = outcome.drop(outcome[y_isna | X_isna].index)

from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(features)
categorical_columns = categorical_columns_selector(features)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, outcome['new_child'], test_size=0.30, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

model.fit(X_train, y_train)
model.score(X_test, y_test)

from sklearn.model_selection import cross_validate
cv_result = cross_validate(model, features, outcome['new_child'], cv=5)
cv_result
