# medical-no-show-predictor

Predicting whether a patient will show up for their medical appointment. Built this using the Kaggle [No-show appointments dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments) with about 110k records from Brazil.

The problem is pretty common in healthcare — missed appointments waste staff time and block other patients from getting care. I wanted to see if you could predict no-shows ahead of time well enough to make targeted reminders worthwhile.

## Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 87.2% | 0.84 | 0.91 |
| XGBoost | 86.8% | 0.83 | 0.90 |
| Logistic Regression | 82.4% | 0.80 | 0.86 |

Random Forest came out on top. Class imbalance was a real issue (most patients do show up), so I used SMOTE to balance the training data.

## Features used

- Patient age and gender
- Days between scheduling and appointment
- Whether an SMS reminder was sent
- Previous no-show history
- Day of week and neighborhood

The most predictive feature ended up being prior no-show history, which makes sense.

## Tech stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost, imbalanced-learn
- Flask (REST API + frontend)
- Matplotlib, Seaborn, Plotly

## Setup

```bash
git clone https://github.com/abdul-aziz-md/medical-no-show-predictor.git
cd medical-no-show-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the dataset from Kaggle and put it in `data/`.

## Usage

Train the model:
```bash
cd scripts
python train_model.py
```

Run the web app:
```bash
cd frontend
python app.py
# http://localhost:5000
```

Or explore the notebooks for EDA:
```bash
jupyter notebook
```

## Project structure

```
medical-no-show-predictor/
├── data/
├── notebooks/
├── scripts/
│   └── train_model.py
├── frontend/
│   └── app.py
├── models/
└── requirements.txt
```

## What I'd do differently

- Try SHAP values for better explainability
- Build an automated retraining pipeline
- Look into threshold tuning (optimizing for recall over accuracy given the use case)

## License

MIT
