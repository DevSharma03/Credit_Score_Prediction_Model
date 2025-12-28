# Credit Score Prediction Model

## 🚀 Introduction

The **Credit Score Prediction Model** is a machine learning–based solution designed to predict an individual’s credit score using financial and demographic attributes.  
This project demonstrates a complete end-to-end ML workflow, covering data preprocessing, model training, API-based inference, and Dockerized deployment.

---

## ⭐ Features

- 📊 Data preprocessing and feature engineering  
- 🤖 Supervised machine learning model for credit score prediction  
- 🧠 REST API for real-time credit score prediction  
- 🐳 Dockerized API for consistent deployment  
- 🧪 Reproducible and modular project structure  
- 📈 Model evaluation and performance analysis  

---

## 🛠 Tech Stack

| Category | Tools & Technologies |
|--------|----------------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| API Framework | FastAPI |
| Visualization | Matplotlib, Seaborn |
| Environment | Docker |
| Dependency Management | requirements.txt |

---

## 📁 Project Structure

```
Credit_Score_Prediction_Model/
├── data/                   # Raw and processed datasets
├── model/                  # Saved / trained model artifacts
├── src/                    # Training, prediction logic, utilities
├── api/                    # FastAPI application for inference
├── Dockerfile              # Docker configuration for API
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧰 Setup Instructions

### Prerequisites

- Python 3.8+
- Git
- Docker (for containerized API)

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/DevSharma03/Credit_Score_Prediction_Model.git
cd Credit_Score_Prediction_Model
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 🔹 1. Train the Model (Local)

Train the machine learning model and save it to the `model/` directory:

```bash
python src/Training Pipeline/credit_analysis_pipeline.ipynb
```

---

### 🔹 2. Run Prediction API (Local)

Start the FastAPI server for prediction:

```bash
py -m uvicorn api.main:app --reload
```

- API will be available at: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

---

### 🔹 3. Run Prediction API using Docker

#### Build Docker Image

```bash
docker build -t credit-score-api .
```

#### Run Docker Container

```bash
docker run -p 8000:8000 credit-score-api
```

- API available at: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

---

## 🛟 Troubleshooting

### Common Issues & Fixes

#### ❌ ModuleNotFoundError
- Ensure the virtual environment is activated
- Reinstall dependencies:
```bash
pip install -r requirements.txt
```

---

#### ❌ FileNotFoundError (Data or Model)
- Verify dataset exists inside `data/`
- Ensure model file exists inside `model/`
- Run training before starting the API:
```bash
python src/train.py
```

---

#### ❌ API Not Starting
- Ensure FastAPI and Uvicorn are installed
- Check correct module path:
```bash
uvicorn api.main:app --reload
```

---

#### ❌ Docker Build Fails
- Ensure Docker is running
- Clear cache and rebuild:
```bash
docker build --no-cache -t credit-score-api .
```

---

#### ❌ Docker Container Exits Immediately
- Check logs:
```bash
docker logs <container_id>
```
- Ensure model files are copied correctly in Dockerfile

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

**Devashish Sharma**  
📧 Email: work.devashishsharma09@gmail.com  
🔗 GitHub: https://github.com/DevSharma03  

---

⭐ If you find this project useful, consider starring the repository!
