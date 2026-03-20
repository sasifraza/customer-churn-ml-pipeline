**Customer Churn Prediction — End-to-End ML Pipeline
**
**Overview**

This project demonstrates a production-style machine learning pipeline for predicting customer churn.
It includes model training, API deployment, and an interactive web interface.

The system allows users to input customer attributes and receive a churn prediction along with the probability.

The goal of this project is to illustrate how a machine learning model moves from training → deployment → user interface, which is a core workflow in ML Engineering and Applied AI systems.

⸻

**Architecture**

User Interface (Streamlit)
↓
Prediction API (FastAPI)
↓
Prediction Module (src/predict.py)
↓
Trained Model (models/model.pkl)
↓
Training Pipeline (src/train.py)

⸻

**Features**
	•	End-to-end machine learning pipeline
	•	Model training and evaluation
	•	Saved model artifact
	•	REST API for predictions
	•	Interactive Streamlit interface
	•	Modular project structure
	•	Reproducible Python environment

⸻

**Tech Stack**

Python
Scikit-learn
Pandas
FastAPI
Uvicorn
Streamlit
Git / GitHub

⸻

**Project Structure**
customer-churn-ml-pipeline
│
├── app
│   ├── api.py              # FastAPI inference service
│   └── streamlit_app.py   # Streamlit UI
│
├── src
│   ├── train.py           # Model training pipeline
│   └── predict.py         # Prediction logic
│
├── models
│   └── model.pkl          # Saved trained model
│
├── data
│   └── churn.csv          # Dataset
│
├── tests
│   └── test_train.py
│
├── requirements.txt
├── README.md
└── .gitignore

**Model Performance**

Example evaluation results:

ROC-AUC: 0.84

**Installation**

Clone the repository:

git clone https://github.com/YOUR_USERNAME/customer-churn-ml-pipeline.git
cd customer-churn-ml-pipeline

Create virtual environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

**Train the Model**

python src/train.py
This will train the model and save:
models/model.pkl

**Run the Prediction API**
uvicorn app.api:app --reload
API documentation available at:
http://127.0.0.1:8000/docs

**Run the Web Interface
**
streamlit run app/streamlit_app.py

Open in browser:
http://localhost:8501

**Example Prediction**

Input customer attributes through the UI and receive:
	•	Churn prediction
	•	Probability score

⸻

**Future Improvements**

Possible enhancements:
	•	Feature engineering pipeline
	•	Model monitoring
	•	Docker containerization
	•	Cloud deployment (Azure / AWS)
	•	Experiment tracking with MLflow
	•	Automated CI/CD pipeline

⸻

**Author**

Syed Asif Raza
PhD — Operations Research / Analytics
Machine Learning & AI Systems

