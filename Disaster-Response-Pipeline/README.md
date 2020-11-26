# Disaster Response Pipeline Project

## Project Overview
In this project, I will apply data engineering to analyze disaster data from Figure 8 to create a model for an API that classifies disaster messages.
Data directory contains a data set which are real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that appropriate disaster relief agency can be reached out for help.

This code is supposed to run a web app that classify a new disaster text messages into several categories, The web app will also display visualizations of the data.

## File Description:
* `process_data.py`A ETL Pipeline that: Loads the messages and categories datasets, Merges the two datasets, Cleans the data, Stores it in a SQLite database.
* `train_classifier.py` A Machine Learning Pipeline that: Loads data from the SQLite database, Splits the dataset into training and test sets, Builds a text processing and machine learning pipeline, Trains and tunes a, Model using GridSearchCV, Outputs results on the test set, Exports the final model as a pickle file This folder contains sample messages and categories datasets in csv format.
* `run.py` A Flask Web App that visualizes the results.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## GitHub link:
   - https://github.com/shahadl/Disaster-response-pipeline

## Results:
The main observations of the trained classifier can be seen by running the application.
