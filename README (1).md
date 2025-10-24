# 🧠 NeoCare: Deep Learning Model for Fetal Health Prediction

NeoCare is a professional deep learning project designed to predict **fetal health conditions** using **Cardiotocogram (CTG)** data.  
It leverages an advanced neural network architecture to classify fetal states into three categories:

- **Normal**
- **Suspect**
- **Pathological**

This model provides AI-powered insights that assist healthcare professionals in **prenatal monitoring** and **early detection of fetal risks**.

---

## 🚀 Project Overview

The NeoCare system integrates **data preprocessing**, **feature scaling**, and a **deep neural network model** (built using TensorFlow/Keras) to achieve high accuracy in fetal health classification.  
It also includes a **Gradio-based web interface** for real-time predictions.

---

## ✨ Features

- Deep Neural Network trained on fetal CTG data
- Automated preprocessing and feature scaling
- Real-time predictions via Gradio web interface
- Visualization of training metrics (Accuracy, Loss)
- Confusion matrix and classification report
- Model and scaler saving for deployment

---

## 🧩 Tech Stack

| Category        | Technologies                |
| --------------- | --------------------------- |
| Language        | Python 3.10+                |
| Frameworks      | TensorFlow, Keras           |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization   | Matplotlib, Seaborn         |
| Deployment      | Gradio                      |

---

## 📊 Model Performance

| Metric                | Score |
| :-------------------- | :---- |
| Accuracy              | 94%   |
| Macro Avg F1-Score    | 0.89  |
| Weighted Avg F1-Score | 0.94  |

---

## 🧠 Model Architecture

```text
Input Layer (19 features)
↓
Dense(128, activation='relu')
↓
Dropout(0.2)
↓
Dense(64, activation='relu')
↓
Dropout(0.2)
↓
Dense(3, activation='softmax')
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/NeoCare-Fetal-Health.git
cd NeoCare-Fetal-Health
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
python app.py
```

---

## 🌐 Gradio Web Interface

The project includes an interactive **Gradio web app** that allows users to:

- Input CTG parameters manually
- Get instant AI predictions for fetal health
- View the model’s confidence across all classes

---

## 🧬 Dataset

The dataset used is the **Fetal Health Classification Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Fetal+Health).  
It contains medical measurements derived from cardiotocograms of pregnant women.

---

## 🧾 Example Output

**Confusion Matrix**

```
[[340   5   2]
 [ 11  62   4]
 [  2   8  45]]
```

**Classification Report**

```
              precision    recall  f1-score   support
Health 1          0.96      0.98      0.97       347
Health 2          0.82      0.78      0.80        77
Health 3          0.90      0.82      0.86        55
```

---

## 💾 Model Saving

The trained model and scaler are both saved for deployment:

```python
model.save('/content/fetal_health_model.h5')

import joblib
joblib.dump(sc, '/content/fetal_scaler.pkl')
```

---

## 📈 Future Enhancements

- Integration with hospital monitoring systems
- Explainability using SHAP for model interpretation
- Cloud deployment on Hugging Face Spaces or Streamlit
- REST API for clinical use

---

## 👨‍💻 Author

**Developed by [Islam Abdul Rahim]**  
AI & Data Science Enthusiast | Specializing in Healthcare Machine Learning

📫 _Connect on [LinkedIn](https://www.linkedin.com/in/islam-a-mohamed-16159a246?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) or check out more projects on [GitHub](https://github.com/Islam-ML/NeoCare-Deep-Learning-Model-for-Fetal-Health-Prediction)_

---

## 🏷️ License

This project is licensed under the **MIT License** – free for personal and commercial use.
