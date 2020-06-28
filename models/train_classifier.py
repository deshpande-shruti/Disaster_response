import sys
import nltk
import numpy as np
import pandas as pd
nltk.download('punkt')
nltk.download('wordnet')
import re
import pickle
import os

from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    ''' Load the data from database function
        Arguments:
            database_filepath = a path to SQLite destination db (Disaster_response_db)
        Output:
               X : Dataframe containing features
               Y : Dataframe containing labels
               category_names : list of categories_names
     '''
     # load data from database
    
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    Disasters =  pd.read_sql_table('Disasters', engine)
    print(Disasters.head())
    
    X = Disasters['message']
    y= Disasters[Disasters.columns[4:]]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    '''Normalize and tokenize the text string
        Args:
            text : A string that contains messages for processing
            
        Returns:
                clean_tokens: tokens extracted from the text 
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #Extract all the url's from the text
    detect_url = re.findall(url_regex, text)
    for url in detect_url:
        text = text.replace(url, 'urlplaceholder')
        
    #convert text to lowercase and remove punctuations
    tokenx = RegexpTokenizer(r'\w+')
    token = tokenx.tokenize(text)
    
    #lemmatize
    lemmatizer = WordNetLemmatizer()
   
    #clean tokens
    clean_to = []
    for t in token:
        clean_t = lemmatizer.lemmatize(t).lower().strip()
        clean_to.append(clean_t)
    return clean_to
    


def build_model():
    mol = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__max_depth' : [10, 50, None],
                 'clf__estimator__min_samples_leaf' : [2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    ''' Print model results
	    input:
		    model : required, estimator object
		    X_test : required
		    y_test : required 
		    category_names = required, list of category strings
	    output: none
    '''
    Y_pred = model.predict(X_test)
    #print classification report
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for c in Y_test.columns:
        print(classification_report(Y_test[c], Y_pred[c]))
        print('Performance with category: {}'.format(c))
       

def save_model(model, model_filepath):
    """
    Save the model into a pkl file
    Args: model, path where to save the file
    return: None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
        
        

def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()