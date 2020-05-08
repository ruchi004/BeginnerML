import pandas as pd
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score

dataset = pd.read_csv("C:\\Users\\RUCHI\\Desktop\\train.csv")

features = ['Pclass' , 'Age' , 'Sex']
X = dataset[features]
y = dataset[['Survived']]

values = X['Age'].mean()
X['Age'] = X['Age'].fillna(values)

label_encoder = LabelEncoder()
transform_sex = array(X['Sex'])
fit_sex = label_encoder.fit_transform(transform_sex)
X['Sex'] = fit_sex

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

C_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
Accuracy = accuracy_score(y_test, y_pred)

print(C_matrix)
print(report)
print(Accuracy)

# Precision = tp/tp+fp (indicates low # of fp)
# Recall = tp/tp+fn (indicates low # of fn)
# F-measures = 2*R*P/R+P (F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more.)
# Accuracy = tp + tn / tp + tn +fp +fn