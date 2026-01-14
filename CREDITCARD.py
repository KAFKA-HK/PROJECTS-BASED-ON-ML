import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score


df = pd.read_csv("creditcard.csv")
print(df.shape)
print(df.columns)
print(df["Class"].value_counts())

X = df.drop("Class",axis = 1)
y = df["Class"]

# preprocessing
Scaler = StandardScaler()
X['Amount'] = Scaler.fit_transform(X[['Amount']])
X['Time'] = Scaler.fit_transform(X[['Time']])
df = df.drop(["Time", "Amount"], axis=1)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify= y, random_state= 42)

# Apply SMOTE
print("Before SMOTE:", np.bincount(y_train))
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", np.bincount(y_train_res))

# model - XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nConfusion Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test,y_proba)
print("ROC AUC SCORE: ", roc_auc)
print("AVERAGE PRECISION (PR AUC) ", average_precision_score(y_test,y_proba))

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()





# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'max_depth': [3, 5, 10],
#     'min_samples_split': [2, 10, 20],
#     'criterion': ['gini', 'entropy']
# }
#
# grid = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), param_grid, cv=3, scoring='average_precision')
# grid.fit(X_train, y_train)
#
# best_model = grid.best_estimator_
# print("Best Parameters:", grid.best_params_)
