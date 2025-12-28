# Credit Score Prediction Model

## 🚀 Introduction

The **Credit Score Prediction Model** is a machine learning–based solution designed to predict an individual’s credit score using financial and demographic attributes.  
This project demonstrates a complete end-to-end ML workflow, covering data preprocessing, model training, evaluation, and deployment readiness.

The repository is structured to be clean, modular, and scalable—making it suitable for academic projects, fintech prototypes, and real-world credit risk assessment systems.

---

## ⭐ Features

- 📊 Data preprocessing and feature engineering  
- 🤖 Supervised machine learning model for credit score prediction  
- 🧠 Model persistence for reuse and inference  
- 🧪 Reproducible and modular project structure  
- 🐳 Dockerized setup for consistent environment and deployment  
- 📈 Evaluation metrics for model performance analysis  

---

## 🛠 Tech Stack

| Category | Tools & Technologies |
|--------|----------------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Environment | Docker |
| Dependency Management | requirements.txt |

---

## 📁 Project Structure

```
Credit_Score_Prediction_Model/
├── data/                   # Raw and processed datasets
├── model/                  # Saved / trained model artifacts
├── src/                    # Source code (training, prediction, utilities)
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧰 Setup Instructions

### Prerequisites

- Python 3.8+
- Git
- Docker (optional)

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DevSharma03/Credit_Score_Prediction_Model.git
cd Credit_Score_Prediction_Model
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the Model

```bash
python src/train.py
```

### Make Predictions

```bash
python src/predict.py
```

---

## 🛟 Troubleshooting

- **ModuleNotFoundError**: Ensure virtual environment is activated  
- **FileNotFoundError**: Check dataset paths inside `data/`  
- **Model not found**: Run training before prediction  
- **Docker issues**: Ensure Docker is running and rebuild image  

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

