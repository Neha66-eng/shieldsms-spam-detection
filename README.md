# SmartSMS Shield: An MLOps-Enabled SMS Spam Detection System Using Multinomial Naive Bayes

## Project overview
SmartSMS Shield classifies an SMS message as **Spam** or **Ham (legitimate)** using a scikit-learn pipeline with TF-IDF features and Multinomial Naive Bayes. It includes data preparation, model evaluation, a Streamlit UI, automated tests, and deployment guidance.

## Technologies
- Python 3.10+
- pandas, scikit-learn, joblib
- Streamlit
- pytest
- Git and GitHub
- Optional deployment: AWS or Google Cloud

## Dataset
Use the UCI SMS Spam Collection dataset:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Download the dataset from UCI and place the extracted `SMSSpamCollection` file in `data/SMSSpamCollection`.
The training script also accepts a tab-separated file with two columns: label (`ham`/`spam`) and message text.

## Project structure
```text
SmartSMS_Shield_MLOps/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── tests/
│   └── test_predict.py
└── README.md
```

## Run locally (Windows CMD)
1. Install Python 3.10 or newer.
2. Open CMD in this project folder.
3. Create and activate a virtual environment:
   ```bat
   py -m venv .venv
   .venv\\Scripts\\activate
   ```
4. Install dependencies:
   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Download/extract the UCI dataset and place `SMSSpamCollection` in the `data` folder.
6. Train and evaluate:
   ```bat
   python train.py
   ```
   This creates `models/sms_spam_pipeline.joblib` and `reports/metrics.json`.
7. Start the app:
   ```bat
   streamlit run app.py
   ```
8. Run tests:
   ```bat
   pytest -q
   ```

## Model and evaluation
The pipeline uses `TfidfVectorizer` plus `MultinomialNB`. The training script creates a stratified train/test split and reports accuracy, precision, recall, F1-score, and a confusion matrix. Do not claim results until you run the script on the selected dataset.

## MLOps practices demonstrated
- Reproducible split with a fixed random seed
- Training and evaluation script separated from the UI
- Saved model pipeline for inference
- Automated unit tests
- Git-based version control
- Containerization with Docker
- Deployment documentation

## GitHub
Create a new repository named `SmartSMS-Shield-MLOps`, then run:
```bash
git init
git add .
git commit -m "Initial SmartSMS Shield MLOps project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/SmartSMS-Shield-MLOps.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username.

## Deployment URLs
- **AWS deployment URL:** Add the public URL after deploying the app to an AWS service (for example, App Runner or an EC2 instance).
- **GCP deployment URL:** Add the public URL after deploying the app to Google Cloud Run.

Do not invent URLs. If your faculty expects both URLs, ask whether both cloud deployments are compulsory or whether one live deployment plus a documented second-cloud plan is acceptable. Cloud services may require billing setup.

## Limitations
This is a student demonstration, not a production-grade spam filter. It may misclassify messages, especially messages in languages or styles poorly represented in the dataset. Avoid entering private or sensitive SMS messages into a public demo.
