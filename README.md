# ❤️ HeartGuard — Heart Disease Risk Prediction System

A machine learning-powered web application that predicts the likelihood of heart disease based on patient health parameters. Built with Python, scikit-learn, and Flask.

🔗 **Live Demo:** [https://heartguard-fgbb.onrender.com](https://heartguard-fgbb.onrender.com)

---

## 📌 About The Project

Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve treatment outcomes. This project uses machine learning to analyze patient health data and predict whether a person is at risk of heart disease.

The system trains three classification models on the **UCI Cleveland Heart Disease Dataset** (920 patients, 4 hospitals) and serves predictions through an interactive web interface where users can input health parameters and receive instant risk assessments.

### Key Features

- **User Authentication** — Register and login to track your predictions
- **3 ML Models** — Logistic Regression, Decision Tree, and Random Forest — choose any model or compare all three
- **Instant Predictions** — Enter health data, get a risk assessment with probability score
- **Explainable AI** — See which health factors most influenced the prediction
- **Health Dashboard** — View prediction statistics, risk distribution, model performance comparison, and feature importance charts
- **Prediction History** — Track all past predictions with full details
- **Deployed Online** — Accessible publicly via Render

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Machine Learning | scikit-learn, pandas, numpy |
| Backend | Flask (Python) |
| Frontend | HTML, CSS (custom design) |
| Data Analysis | Jupyter Notebook, matplotlib, seaborn |
| Deployment | Render, Gunicorn |
| Version Control | Git, GitHub |

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository — Heart Disease Dataset (Cleveland, Hungary, Switzerland, VA Long Beach)

| Property | Value |
|----------|-------|
| Total Patients | 920 |
| Features | 13 medical parameters |
| Target | Binary (Heart Disease: Yes/No) |
| Hospitals | 4 (Cleveland, Hungary, Switzerland, VA Long Beach) |

### Features Used

| Feature | Description |
|---------|-------------|
| age | Patient's age in years |
| sex | Gender (Male/Female) |
| cp | Chest pain type (4 categories) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol level (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalch | Maximum heart rate achieved |
| exang | Exercise-induced chest pain |
| oldpeak | ST depression from exercise stress test |

### Data Preprocessing

- Dropped columns with >30% missing values (`ca`, `thal`, `slope`)
- Replaced impossible values (0 blood pressure, 0 cholesterol) with median
- Encoded categorical text columns to numeric values
- Converted multi-class target (0-4 severity) to binary (0 = healthy, 1 = disease)
- Filled remaining missing values with column median
- Scaled features using StandardScaler

---

## 🤖 Model Performance

Three models were trained and evaluated on a 80/20 train-test split:

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | 80.4% | 84.1% | 82.6% |
| **Decision Tree** | **83.2%** | **85.5%** | **86.2%** |
| Random Forest | 81.0% | 84.9% | 82.6% |

**Decision Tree performed best** with 83.2% accuracy, 85.5% precision, and 86.2% recall.

---

## 📁 Project Structure

```
HEART-DISEASE-PREDICTOR/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version for deployment
├── render.yaml                 # Render deployment config
├── decision_tree.pkl           # Trained Decision Tree model
├── logistic_regression.pkl     # Trained Logistic Regression model
├── random_forest.pkl           # Trained Random Forest model
├── scaler.pkl                  # Feature scaler (StandardScaler)
├── features.pkl                # Feature names list
├── model_performance.pkl       # Saved model evaluation metrics
├── Untitled.ipynb              # Jupyter Notebook (data analysis + model training)
└── templates/
    ├── base.html               # Base template (navbar, styling)
    ├── login.html              # Login page
    ├── register.html           # Registration page
    ├── predict.html            # Prediction form + results
    ├── dashboard.html          # Health dashboard with charts
    └── history.html            # Prediction history table
```

---

## 🚀 How To Run Locally

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/WeendFlo/HEART-DISEASE-PREDICTOR.git
   cd HEART-DISEASE-PREDICTOR
   ```

2. **Install dependencies**
   ```bash
   pip install flask scikit-learn pandas numpy
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

5. **Register an account** and start making predictions!

---

## 📸 Screenshots

### Login Page
Clean authentication interface with register/login functionality.

### Prediction Form
User-friendly form with plain English labels — no medical jargon. Choose from 3 ML models.

### Prediction Results
- Risk assessment (High/Low) with probability score
- All 3 models compared side by side
- Feature importance breakdown showing which factors influenced the prediction

### Health Dashboard
- Prediction statistics and risk distribution
- Model performance comparison table
- Interactive feature importance charts with tabs for each model
- Recent prediction history

---

## 📓 Jupyter Notebook

The `Untitled.ipynb` notebook contains the complete data science pipeline:

1. **Data Loading** — Loading the UCI heart disease dataset
2. **Exploratory Analysis** — Dataset shape, column descriptions, basic statistics
3. **Data Quality Checks** — Missing values, duplicates, impossible values
4. **Data Preprocessing** — Encoding, cleaning, scaling
5. **Visualization** — Target distribution, age histograms, correlation heatmap, boxplots
6. **Model Training** — Logistic Regression, Decision Tree, Random Forest
7. **Model Evaluation** — Accuracy, precision, recall, confusion matrices
8. **Model Saving** — Exporting trained models as .pkl files

---

## 🔮 How Prediction Works

1. User submits health parameters through the web form
2. Flask backend receives the form data
3. Input is converted to a DataFrame and scaled using the saved StandardScaler
4. The selected ML model (loaded from .pkl file) runs `model.predict()` and `model.predict_proba()`
5. All 3 models generate predictions for comparison
6. Feature importance is calculated to explain the prediction
7. Results are displayed with risk level, probability, model comparison, and explanations

---

## ⚠️ Disclaimer

This application is a **screening tool built for educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical decisions.

---

## 👤 Author

WeendFlo


---

## 📄 License

This project is open source and available for educational use.
