import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#===================================================================================#
train = pd.read_csv("landmarks_train_v2.csv") # -> renomear para arquivo.csv correto
test = pd.read_csv("landmarks_test_v2.csv")   # -> renomear para arquivo.csv correto
#===================================================================================#
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())
print(test.head())

X_train = train.drop(columns=["label"]).values
y_train = train["label"].values

X_test = test.drop(columns=["label"]).values
y_test = test["label"].values

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Precisão:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "model/classifier_v2.pkl")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(xticks_rotation=45)
plt.tight_layout()
#===================================================================================#
plt.savefig("confusion_matrix_v2.png") # -> renomear para arquivo.png desejado
#===================================================================================#
