import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
  '''
  INPUT: message csv and categories csv

  TASK: merge the two datasets

  OUTPUT: combined dataset
  '''
  #read in data
  messages = pd.read_csv(messages_filepath)
  categories = pd.read_csv(categories_filepath)
  # merge datasets
  df = messages.merge(categories, on="id")

  return df



def clean_data(df):
  '''
  INPUT: dataframe

  TASK: preprocess the categories column, split string and separate into
  individual columns; drop duplicates and rows with category value of 2

  OUTPUT: clean dataframe
  '''
  # create a dataframe of the 36 individual category columns
  categories = df["categories"].str.split(";", expand=True)

  #name the categories column
  # select the first row of the categories dataframe
  row = categories.iloc[0]
  #extract column names
  category_colnames = row.apply(lambda x: x[:-2]).tolist()
  categories.columns = category_colnames

  #convert category values to just numbers 0 or 1
  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype('int64')

    #replace categories column in df with the new category columns
    # drop the original categories column from `df`
  df.drop(["categories"], axis=1, inplace=True)
  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df, categories], axis=1)

  #drop rows that have value 2 in the related column of df
  df = df[df.related !=2]


  #remove duplicates
  df.drop_duplicates(inplace = True)

  return df



def save_data(df, database_filename):
  '''
  INPUT: dataframe, database name

  TASK: create a database and store the dataframe as a table
  '''
  #create a database
  engine = create_engine('sqlite:///{}'.format(database_filename))
  #create a table in the database and store the clean dataset
  df.to_sql('messages', engine, index=False, if_exists='replace')



def main():
  '''
  TASK: retrieve arguments from command line and call functions to preprocess data;
         create a 'messages' table in a database 'DisasterResponsedb'
  '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]


        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()