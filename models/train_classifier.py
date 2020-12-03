import sys
import nltk
nltk.download(['punkt','wordnet'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier


def load_data(database_filepath):
	'''
	INPUT: filepath of the Disaster database

	OUTPUT: message dataframe, categories dataframe, list of categories

	TASK: retrieve the messages table from the database and split into 
	input and output dataset for machine learning; get list of categories
	'''
	# load data from database
	engine = create_engine('sqlite:///{}'.format(database_filepath))
	df = pd.read_sql_table('messages', engine)
	X = df['message'].values
	Y = df[['related', 'request', 'offer',
	       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
	       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
	       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
	       'infrastructure_related', 'transport', 'buildings', 'electricity',
	       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
	       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
	       'other_weather', 'direct_report']].values

	#message categories
	categories = list(df.columns.values)
	categories = categories[4:]

	return df, categories


def tokenize(text):
    '''
    INPUT : message text
    OUTPUT: tokenized words
    TASK: function tokenizes message text
    '''
    #tokenize text
    tokens = word_tokenize(text)
    
    #perform lemmatization
    clean_tokens = []
    for tok in tokens:
        clean_tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    '''
    TASK : build a machine learning pipeline
    OUTPUT: pipeline
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize, max_df=0.8, ngram_range=(1,2))),
        ('tfidf',TfidfTransformer(use_idf=False)),
        ('clf', OneVsRestClassifier(LinearSVC()))
    ])
   
    
    return pipeline


#def evaluate_model(model, X_test, Y_test, category_names):
 #   pass
def evaluate_model(model, y_test, test_message,category_names):
	'''
	INPUT: model, test_x data, test_y data

	TASK: predict and evaluate prediction on test data

	'''

	for category in category_names:
		print("**Processing {}".format(category))
		#predict using model on test data
		prediction = model.predict(test_message)
		print(metrics.classification_report(y_test[category], prediction, labels=np.unique(prediction)))


def save_model(model, model_filepath):

	#filename = 'finalmodel.sav'
	filename = model_filepath
	pickle.dump(model, open(filename,'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df, category_names = load_data(database_filepath)
        train, test = train_test_split(df, test_size=0.2)

        train_message = train['message']
        test_message = test['message']

        y_train = train.drop(labels=['id','message','original','genre'], axis=1)
        y_test = test.drop(labels=['id','message','original','genre'], axis=1)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        for category in category_names:
        	model.fit(train_message, y_train[category])
        
        print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)
        evaluate_model(model, y_test,test_message,category_names)

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