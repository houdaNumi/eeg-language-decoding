import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np










phrases= np.load("phrases.npy")
labels= np.load("labels.npy")
print(phrases.shape)
print(labels.shape)


print(np.isnan(phrases).sum())

phrases = np.nan_to_num(phrases, nan=0.0)
#split
# 1. split
x_train, x_test, y_train, y_test = train_test_split(phrases, labels, test_size=0.2, random_state=42)



# 2. feature selection
selector = SelectKBest(f_classif, k=100)
x_train_selected = selector.fit_transform(x_train, y_train)
x_test_selected = selector.transform(x_test)




#norm
scaler = StandardScaler()
x_train_scaler= scaler.fit_transform(x_train_selected)
x_test_scaler= scaler.transform(x_test_selected)


svm= SVC(class_weight='balanced')
svm.fit(x_train_scaler, y_train)
y_pred_svm= svm.predict(x_test_scaler)

knn = KNeighborsClassifier()
knn.fit(x_train_scaler, y_train)
y_pred_knn= knn.predict(x_test_scaler)

print(accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print(accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

