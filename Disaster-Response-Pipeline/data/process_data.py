# import necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    input:
        messages_filepath: The path of messages csv dataset.
        categories_filepath: The path of categories csv dataset.
    output:
        df: A dataframe of messages and categories
    '''
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath) 
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
    '''
    input:
        df: The merged dataset before cleaning
    output:
        df: Dataset after cleaning.
    '''
    
    # Divide ['categories'] into Separate Category Columns.
    categories = df.categories.str.split(';', expand = True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0] 
   
    # Cut the last character for each category
    category_colnames =  row.apply(lambda x: x[:-2]).values.tolist()

    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # Convert column from string to int
        categories[column] = categories[column].astype(int)

    # Drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1, join='inner')

    # Drop the duplicates.
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    input:
        df:Dataframe containing cleaned version of merged message and 
        categories data.
    output:
        no output
    '''
    engine = create_engine('sqlite:///' + database_filename)
    # Sql table name: DisasterResponse
    df.to_sql('DisasterResponse', engine, index=False)
        
def main():
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