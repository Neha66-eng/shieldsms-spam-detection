
# ShieldSMS: ML-Based SMS Spam Detection

## 1. Project Description

ShieldSMS is a machine learning web application that classifies SMS messages as Spam or Ham (legitimate). It uses TF-IDF text features and a Multinomial Naive Bayes classifier. The Streamlit interface allows users to enter a message and view the predicted class and estimated spam probability.

## 2. Technologies and Tools Used

- Python
- pandas
- scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- joblib
- Streamlit
- pytest
- Git and GitHub
- Docker

## 3. Dataset

The project uses the UCI SMS Spam Collection dataset.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## 4. Project Features

- SMS spam and legitimate-message classification
- Text preprocessing and TF-IDF feature extraction
- Machine learning model training and evaluation
- Precision, recall, F1-score and confusion matrix
- Estimated spam probability
- Interactive Streamlit web interface

## 5. Project Structure

```text
SmartSMS_Shield_MLOps_starter/
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── Dockerfile
├── data/
├── models/
├── reports/
└── tests/
```

## 6. Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip

### Install dependencies

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Prepare the dataset

Download the UCI SMS Spam Collection dataset and place the extracted `SMSSpamCollection` file inside the project's `data/` directory.

### Train the model

```bash
python train.py
```

### Test an SMS message in the terminal

```bash
python predict.py
```

### Run the web application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually http://localhost:8501.

## 7. Model Evaluation

The initial evaluation produced these results:

- Spam precision: 100%
- Spam recall: 73.44%
- Spam F1-score: 84.68%

These results are specific to the evaluation dataset. The model may make incorrect predictions on new messages.

## 8. Deployment URLs

- AWS deployment URL: Pending deployment
- Google Cloud Platform (GCP) deployment URL: Pending deployment

## 9. Author

Student project developed for practical learning in machine learning, MLOps and cloud deployment.
