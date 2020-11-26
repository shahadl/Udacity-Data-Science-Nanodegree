# Import necessary libraries
import sys
import pandas as pd
import numpy as np
import re
# Importing libraries necessary for machine learning
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import TruncatedSVD
import pickle
import warnings
# SQL
from sqlalchemy import create_engine
# Import nltk
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):
    """
    Load and merge datasets
    input: database filepath
    outputs: X: the message column
             y: the categories
             category_name: the names of the all categories
    """
    # Load data from database 'DisasterResponse'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Assign the required columns to the variables X,Y
    X = df.message.values
    Y = df.iloc[:, 4:].values

    category_names = (df.columns[4:]).tolist()
    return X, Y, category_names 

def tokenize(text):
    """
    Tokenize input a text and return a clean tokens after normalization, lemmatization, and removing stopwords
    Inputs: a piece of text
    Outputs: a clean list of tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
     # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Take out all punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    Build model with GridSearchCV
    Outputs: Trained model after performing grid search
    """
    # Build a model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
    ])

    # Set parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }

    # Optimize model
    model = GridSearchCV(pipeline, param_grid=parameters,
                         cv=2, verbose=1)
    return model
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Shows model's performance
    Input: model: trained model
           X_test: Test features
           Y_test: Test targets
           category_names: Target labels
    Output: Prints the Classification report
    """
    # Predict categories of messages.
    Y_pred = model.predict(X_test)

    print("----Classification Report per Category:\n")
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file    
    Inputs: model: The trained model
            model_filepath: a place to save the model
    """
    # Export the model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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