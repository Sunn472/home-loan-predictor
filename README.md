# 🏠 Home Loan Predictor

## 📌 Objective
The objective of this project is to build a predictive model that determines whether a loan application will be approved based on applicant details such as income, credit history, employment status, and other demographic/financial factors.  
This helps banks and financial institutions **reduce risk, automate loan approval, and improve decision-making efficiency**.

---

## 🚀 Features
- Predicts loan approval status.
- Handles missing values and categorical data.
- Provides a user-friendly interface (Flask/Web App).
- End-to-end pipeline: Data preprocessing → Model training → Prediction.
- Supports deployment on cloud platforms (Render/Heroku).

---

## 📂 Project Structure
├── artifacts/ # Saved models and preprocessors
├── data/ # Dataset (raw & processed)
├── notebooks/ # Jupyter notebooks for EDA & experiments
├── src/ # Source code
│ ├── pipeline/ # Training & prediction pipeline
│ ├── components/ # Data transformation, model trainer, etc.
│ ├── utils/ # Utility functions
│ ├── logger.py # Logging
│ └── exception.py # Custom exceptions
├── templates/ # HTML files for Flask app
├── app.py # Flask application
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ Setup Instructions

1. **Create project folder**
   ```bash
   mkdir home-loan-predictor
   cd home-loan-predictor

2. Create virtual environment
    python -m venv venv

3. Install dependencies
    pip install -r requirements.txt

4. Run Flask app
    python app.py

🧪 Training the Model
    python src/pipeline/train_pipeline.py

📊 Dataset

Uses loan application data (ApplicantIncome, LoanAmount, CreditHistory, Gender, Education, Employment, etc.) from Kaggle/UCI.

📌 Tech Stack
    Python, Flask
    Pandas, NumPy, Scikit-learn
    Machine learning algorithm
    HTML

👤 Author

Sunny Bharti
📧 sb3225796@gmail.com
<<<<<<< HEAD

=======
>>>>>>> 0d70e80731a867d38106baa4f5f6dcacc62c5511
🔗 GitHub (https://github.com/Sunn472)
 | [Linkedin] (https://www.linkedin.com/in/sunny2003/)
