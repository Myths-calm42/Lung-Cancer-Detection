from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_model(model_name="svm"):
if model_name == "svm":
return SVC(kernel="linear", probability=True)
elif model_name == "rf":
return RandomForestClassifier(n_estimators=100)
elif model_name == "logreg":
return LogisticRegression(max_iter=1000)
else:
raise ValueError("Unknown model")
