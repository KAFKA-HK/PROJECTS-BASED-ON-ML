import pandas as pd




# Load the dataset (adjust path as needed)
df = pd.read_csv('loan.csv', low_memory=False)


# Preview the dataset
print(df.shape)
df.head()


selected_cols = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
    'emp_length', 'home_ownership', 'annual_inc', 'purpose',
    'dti', 'delinq_2yrs', 'revol_util', 'loan_status'
]

df = df[selected_cols]

df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})


# Clean term: "36 months" → 36
df['term'] = df['term'].str.extract('(\d+)').astype(int)

# Clean int_rate and revol_util: "13.56%" → 13.56
df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)
df['revol_util'] = df['revol_util'].str.rstrip('%').astype(float)

# Clean emp_length: e.g., "< 1 year", "10+ years"
df['emp_length'] = df['emp_length'].replace({
    '10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6,
    '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1,
    '< 1 year': 0, 'n/a': 0
}).astype(float)

# Drop rows with missing values in the selected subset
df.dropna(inplace=True)
# print(df.shape)

df = pd.get_dummies(df, columns=['grade', 'home_ownership', 'purpose'], drop_first=True)


from sklearn.model_selection import train_test_split

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train the decision tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(tree_model, feature_names=X.columns, class_names=['Paid', 'Default'], filled=True, max_depth=3)
plt.title("Decision Tree (First 3 Levels)")
plt.show()


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = tree_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


from sklearn.metrics import roc_curve, roc_auc_score

y_proba = tree_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label='Decision Tree')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

print("ROC AUC Score:", roc_auc_score(y_test, y_proba))


