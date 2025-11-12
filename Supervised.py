import pandas as pan
import numpy as num
import seaborn as sea
import matplotlib.pyplot as plot
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

print("==Dataframe==")
dataframe = pan.read_csv("D:\Code stuff\MashineLearnin\loan.csv")
print(dataframe.head())
print(dataframe.info())

dataframe.dropna()
labEn = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Amount', 'Loan_Amount_Term', 'Loan_Status', 'Credit_History']:
    dataframe[col] = labEn.fit_transform(dataframe[col])

x = dataframe.drop('Loan_Status', axis=1)
y = dataframe["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


print("==Showing Confusion Matrix==")
ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap="Blues")
plot.title("Confusion Matrix")
plot.show()

print("==Showing Feature Importance==")
importance = model.feature_importances_
indices = num.argsort(importance)[::-1]
feature = x.columns

plot.figure(figsize=(10, 6))
plot.barh(feature[indices], importance[indices])
plot.xlabel("Importance Score")
plot.ylabel(feature)
plot.title("Feature Importance")
plot.gca().invert_yaxis
plot.show()

print("==Showing Predicted Loan Eligiblity==")
sea.countplot(x=y_pred)
plot.title("Predicted Loan Eligibility")
plot.xlabel("Loan Status (0 = Not Eligible, 1 = Eligible)")
plot.ylabel("Count")
plot.show()

print("==Showing ROC Curve==")
ypred_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, ypred_prob)
rocauc = auc(fpr, tpr)
plot.plot(fpr, tpr, label=f"AUC = {rocauc:.2f}", color='blue')
plot.plot([0,1], [0,1], linestyle='--', color='gray')
plot.title("ROC Curve")
plot.xlabel("False Positive Rate")
plot.ylabel("True Positive Rate")
plot.legend()
plot.show()
