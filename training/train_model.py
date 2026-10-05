from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
import joblib

file_path = Path(__file__).resolve().parent.parent / "data" / "diabetes_data_upload.csv"
df = pd.read_csv(file_path)

clean =df.drop_duplicates().copy()

X = clean.drop(columns = ["class"])
y = clean["class"].map({
    "Negative" : 0,
    "Positive" : 1
})

X_train, X_test,Y_train, Y_test = train_test_split(X,y, test_size= 0.2, random_state =42 , stratify = y)

numeric_columns = ["Age"]
categorical_columns = X.columns.drop("Age").tolist()

numeric_pipeline = Pipeline ([
    ("fill_missing" , SimpleImputer(strategy = "median")),
    ("scale", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("fill_missing", SimpleImputer(strategy = "most_frequent")),
    ("encode", OneHotEncoder(handle_unknown = "ignore" , sparse_output = False ))    
])

preprocessor = ColumnTransformer([
    ("numbers", numeric_pipeline, numeric_columns),
    ("categories", categorical_pipeline, categorical_columns)
])



model = Pipeline([
    ("prepare", preprocessor),
    ("classifier", LogisticRegression(max_iter = 1000))
])

model.fit(X_train, Y_train)

predictions= model.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)

baseline = DummyClassifier(strategy = "most_frequent")
baseline.fit(X_train, Y_train)

baseline_predictions = baseline.predict(X_test)
accuracybase = accuracy_score(Y_test, baseline_predictions)

matrix = confusion_matrix(Y_test, predictions, labels =[0, 1])
# print("Confusion matrix")
# print(matrix)

tn, fp, fn , tp = matrix.ravel()

precision = precision_score(Y_test, predictions, zero_division = 0)
recall = recall_score(Y_test, predictions, zero_division=0)
f1 = f1_score(Y_test, predictions, zero_division=0)
f1_base = f1_score(Y_test, baseline_predictions, zero_division=0)

probabilities = model.predict_proba(X_test)
positive_probabilities = probabilities[:, 1]

comparison = pd.DataFrame({
    "actual" : Y_test.to_numpy(),
    "predicted" : predictions,
    "positive_probabitlity": positive_probabilities
})


cv = StratifiedKFold(n_splits =5, shuffle = True, random_state = 42)

scores = cross_val_score(model, X_train, Y_train, cv = cv, scoring = "f1")

tree_model = Pipeline([
    ("prepare", clone(preprocessor)),
    ("classifier", DecisionTreeClassifier(max_depth = 3 , random_state = 42))
])

tree_scores = cross_val_score(tree_model, X_train, Y_train, cv=cv, scoring = "f1")

param_grid = {
    "classifier__max_depth": [2,3,4,5],
    "classifier__min_samples_leaf": [1,3]
}

search = GridSearchCV(estimator = tree_model , param_grid = param_grid, scoring = "f1" , cv = cv, refit = True)
search.fit(X_train, Y_train)


final_model = search.best_estimator_

final_predictions = final_model.predict(X_test)


final_accuracy = accuracy_score(Y_test , final_predictions)
final_precision = precision_score(Y_test , final_predictions, zero_division = 0)
final_recall = recall_score(Y_test, final_predictions, zero_division = 0)
final_f1 = f1_score(Y_test , final_predictions, zero_division = 0)

final_matrix = confusion_matrix(Y_test, final_predictions , labels = [0,1])
final_report = classification_report(Y_test, final_predictions, labels =[0,1] ,target_names = ["Negative" , "Positive"], digits = 2 , zero_division = 0)


model_path = Path(__file__).resolve().parent.parent / "models" / "diabetes_pipeline.joblib"

model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, model_path)

reloaded_model = joblib.load(model_path)
reloaded_predictions = reloaded_model.predict(X_test)

same_predictions = np.array_equal(final_predictions, reloaded_predictions)

original_probabilities = final_model.predict_proba(X_test)
reloaded_probabilities = reloaded_model.predict_proba(X_test)

same_probabilities = np.allclose(original_probabilities , reloaded_probabilities)




print(f"Accuracy: {final_accuracy*100}%")
print(f"Precision: {final_precision*100}%")
print(f"Recall: {final_recall*100}%")
print(f"F1: {final_f1*100}%")
print(f"Confusion matrix: \n{final_matrix}")
print(final_report)
print(f"Model saved to: {model_path}")
print()
print("Predictions match after loading:", same_predictions)
print("Probabilities match after loading:", same_probabilities)





# print("Best settings:",search.best_params_)
# print(f"Best mean CV f1: {search.best_score_:.2%}")







# print(f"Logistic regression mean f1: {scores.mean():.2%}")
# print(f"Decision Tree mean f1: {tree_scores.mean():.2%}")






# print("F1 for each fold:", scores)
# print(f"Mean F1: {scores.mean():.2f}")
# print(f"Standard Deviation: {scores.std():.2f}")








# print(model.classes_)
# print(comparison.head(10))

# print(f"Accuracy: {accuracy*100}%")
# print(f"Baseline accuracy: {accuracybase*100}%")
# print(f"Precision: {precision*100}%")
# print(f"Recall: {recall*100}%")
# print(f"Model F1: {f1*100:.2f}%")
# print(f"Base F1: {f1_base*100:.2f}%")
# print()
# print(classification_report(Y_test, predictions, labels =[0, 1] , target_names = ["Negative" , "Positive"] , digits =2, zero_division = 0))
# print()
# print(classification_report(Y_test, baseline_predictions, labels =[0, 1] , target_names = ["Negative", "Positive"] , digits =2, zero_division = 0))









# print(tn + fp + fn + tn)
# print(fp + fn)

# print(f"Predicted: {predictions}")
# print(f"Accuracy: {accuracy*100}%")

# print(f"Baseline accuracy: {accuracybase*100}%")
# X_train_prepared = preprocessor.fit_transform(X_train)
# X_test_prepared = preprocessor.transform(X_test)
# print(preprocessor.get_feature_names_out())
# print(X_train_prepared.shape)
# print(X_test_prepared.shape)
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)
# print(Y_train.value_counts())
# print(Y_test.value_counts())
# print(numeric_columns)
# print(categorical_columns)
# print(X.shape)
# print(y.shape)
# print(y.value_counts())
# print(f"Shape: {clean.shape}")
# print(f"Duplicates: {clean.duplicated().sum()}")
# print(clean["class"].value_counts())
# print(f"First five rows: {df.head()}")
# print(f"Shape: {df.shape}")
# print(f"Column names: \n{df.columns.tolist()}")
# print(f"Data types: \n{df.dtypes}")
# print(f"Missing value count: \n{df.isna().sum()}")
# print(f"Duplicate rows: {df.duplicated().sum()}")
# print(df["class"].value_counts())
