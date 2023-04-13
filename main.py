import streamlit as st

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import numpy as np

import matplotlib.pyplot as plt

st.title("StreamLit Machine Learning Example")

st.write("""
# Explore different classifier
## Uses wonderful visualizations from streamlit with Scikit-learn datasets""")
selected = "Selected"


dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
st.success(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    
    x = data.data
    y = data.target
    return x, y

x, y = get_dataset(dataset_name)
shape_of_dataset = x.shape
number_of_unique_classes = len(np.unique(y))

st.write("shape_of_dataset", shape_of_dataset)
st.write("number_of_classes", number_of_unique_classes)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        leaf_size = st.sidebar.slider("leaf_size", 30, 40)
        params["K"] = K
        params["leaf_size"] = leaf_size
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        degree = st.sidebar.slider("degree", 3, 7)
        params["C"] = C
        params["degree"] = degree
    else:
        max_depth = st.sidebar.slider("max_depth", 3, 20)
        n_estimators = st.sidebar.slider("n_estimators", 3, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)



# clf means classifier
def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"], leaf_size = params["leaf_size"])
    elif classifier_name == "SVM":
        clf = SVC(C = params["C"], degree = params["degree"])
    else:
        clf = RandomForestClassifier(n_estimators= params["n_estimators"], max_depth = params["max_depth"], random_state = 1234)
    return clf

clf = get_classifier(classifier_name, params)

# Classification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =12345)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

# Plotting
# Feature reduction from higher dimension to lower dimension using PCA for plotting
# PCA = Principal Component Analysis
# Feature reduction 2 Dimenion 
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c = y, alpha= 0.8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()


st.pyplot(fig)