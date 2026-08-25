# TruckGuard AI — Production AI/ML Platform for Trucking HOS Risk



TruckGuard AI is a production-style AI/ML application that evaluates trucking Hours-of-Service risk using a rules engine, custom machine-learning classifier, FastAPI, Streamlit, SQLite logging, MLflow tracking, and automated tests.



This project connects transportation domain knowledge with applied AI/ML engineering.



## What It Does



A dispatcher, safety manager, or driver enters a driver's current HOS clock situation:



- Driving hours today

- Hours inside the 14-hour duty window

- Driving hours since last qualifying break

- Cycle hours used

- 60/70-hour cycle limit

- Consecutive off-duty hours



TruckGuard AI returns:



- Final risk level: LOW, MEDIUM, or HIGH

- Whether the driver can continue driving

- Remaining drive hours

- Remaining duty-window hours

- Remaining cycle hours

- Break-required status

- 34-hour restart eligibility

- ML prediction

- ML confidence score

- Plain-English explanation

- Recommended action

- Database log ID



## Project Status



Completed:



- HOS rules engine

- Synthetic HOS data generation

- Custom Naive Bayes ML risk classifier

- MLflow experiment tracking with SQLite backend

- Saved model artifact

- Prediction service

- Explanation service

- FastAPI API

- SQLite prediction logging

- Prediction history endpoint

- Streamlit dashboard

- Automated pytest test suite



## Tech Stack



Python, FastAPI, Streamlit, pandas, NumPy, SQLAlchemy, SQLite, MLflow, joblib, Pydantic, pytest, custom Naive Bayes classifier, Git/GitHub.



## Run Locally



Create and activate a virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1


