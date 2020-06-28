# Disaster_response_pipeline
## Objective:
              This project is to classify disaster response messages through Machine learning.
              
## Table of Contents:
### Description
The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim's to build a NLP modelto categorize messages on a real time basis.

This project is divided in the following key sections:

1. Process data, build an ETL pipeline to extract data from source, clean the data and save them in Sqlite db
2. Build a machine learning pipeline to train which then classifies text messages in various categories.
3. Run the app that shows model results in real time.

### Important Files
 - app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

### Execution
1. Run the following commands in the projects'directory to set the database, train model and save model.
  - Data pipelines:
    - Run the ETL pipeline to clean data and store the processed data in the database
       `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Run ML pipeline that trains classifier and saves
       `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

### Additional
You can find 2 jupyter notebooks in the folders: **data** and **models**.
1. **data**
      - **ETL Pipeline** :It takes data from disaster_categories.csv and disaster_messages.csv and reads, cleans and stores data in a SQL database.
2. **models**
      - **ML Pipeline**: Loads the data to sql DB, tronsforms it using NLP and ML algorithm.

### Author
* [Shruti ND](https://github.com/deshpande-shruti)
