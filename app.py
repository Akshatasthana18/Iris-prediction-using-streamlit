import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load('iris_model.pkl')

# App title
st.title("🌸 Iris Flower Prediction App")

st.write("Enter the features below to classify the type of Iris flower.")

# User inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
class_names = ['Setosa', 'Versicolor', 'Virginica']
st.subheader(f"🌼 Prediction: **{class_names[prediction]}**")

# Load dataset for visualization
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Visualization
st.subheader("📊 Sepal Length vs Sepal Width")
fig, ax = plt.subplots()
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=df, palette='viridis', ax=ax)
plt.scatter(sepal_length, sepal_width, c='red', s=100, label='Your Input')
plt.legend()
st.pyplot(fig)


st.subheader("📊 Petal Length vs Petal Width")
fig, ax = plt.subplots()
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='species', data=df, palette='viridis', ax=ax)
plt.scatter(petal_length, petal_width, c='red', s=100, label='Your Input')
plt.legend()
st.pyplot(fig)
