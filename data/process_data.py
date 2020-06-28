import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' Load Message data with categries function
       Arguments:
            message_filepath -> Path to the CSV file containing messages
            categories_filepath -> Path to the CSV file containing categories
        Output:
            df -> Combined data containing messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df
    
def clean_data(df):
    categories = df['categories'].str.split(pat=';', expand=True)
   
    
    '''Clean data included in the DataFrame and transform categories part
    INPUT
    df -- type pandas DataFrame
    OUTPUT
    df -- cleaned pandas DataFrame
    '''
    row = categories.iloc[[1]]
    category_colnames = [cat_name.split('-')[0] for cat_name in row.values[0]]
    print(category_colnames)
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)
    
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)
    df.drop_duplicates(inplace=True)
   
    return df
 
def save_data(df, database_filename):
    ''' Saves dataframe (df) to database path'''
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('Disasters', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
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