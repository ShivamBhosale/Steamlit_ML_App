import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.title("Streamlit Machine Learning Application")
st.write(""" ## Created By: Shivam Bhosale """)

dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest","Decision Tree"))


def get_dataset(name):
    data = None  
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()


    X = data.data
    Y = data.target
    return X,Y

X,Y = get_dataset(dataset_name)
st.write("Shape of the dataset", X.shape)
st.write("Number of classes", len(np.unique(Y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth",2,15)
        params["max_depth"] = max_depth
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["n_estimators"] = n_estimators
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        
    return params

params=add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],random_state=4)
    else:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=params["max_depth"])
    
    return clf


clf = get_classifier(classifier_name,params)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(Y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=Y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)