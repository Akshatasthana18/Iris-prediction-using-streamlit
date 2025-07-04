# 🌸 Iris Flower Classification Web App using Streamlit

This project is a simple and interactive web application built using **Streamlit** that predicts the species of an Iris flower based on user-provided input features. It uses a trained **Random Forest Classifier** on the popular **Iris dataset**.

---

## 🚀 Features

- 📥 Interactive sliders to input sepal and petal measurements
- 🧠 Real-time prediction of Iris species (`Setosa`, `Versicolor`, `Virginica`)
- 📊 Visualization of user input on top of real dataset using Seaborn
- 💾 Machine Learning model saved and loaded using `joblib`
- 🌐 Easy deployment with Streamlit Cloud



---

## 🧠 Model Overview

- **Dataset:** Iris dataset (from `sklearn.datasets`)
- **Model:** Random Forest Classifier (`sklearn.ensemble`)
- **Training Script:** `train_model.py`
- **Model Output:** `iris_model.pkl`

To retrain the model:

```bash
python train_model.py
