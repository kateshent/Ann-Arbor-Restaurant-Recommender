# embeddings_to_csv.py

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.append('../')
from embedding import *



def load_dataset_csv(file_path):
    """Load dataset from a CSV file."""
    dish_df = pd.read_csv(file_path, delimiter=',')
    dish_df.columns = ['Restaurant Name', 'Menu Item', 'Item Description']
    return dish_df



def get_food_embeddings(dish_df):
    """
    Get and store BERT embeddings for each menu item in a dataframe.

    Args:
    - dish_df (pandas DataFrame): DataFrame containing menu item and item description columns.

    Returns:
    - None
    """
    # Join menu item and menu description into one column
    dish_df['index'] = dish_df.index
    dish_df['Food Item'] = dish_df['Menu Item'].astype(str)+" "+ dish_df["Item Description"]

    # Food Data into list format to input into model
    food_data = dish_df['Food Item'].tolist()

    # Add index number to dataframe, so the restaurant name can be retrieved later
    Indexes = dish_df['index'].tolist()

    # Initialize an empty list for data_embeddings
    data_embeddings = []

    # Specify the chunk size
    chunk_size = 1000

    # Loop over the food_data list in chunks of chunk_size
    for i in range(0, len(food_data), chunk_size):
        # Calculate the start and end indices for the current chunk
        start = i
        end = min(i + chunk_size, len(food_data))
        print(start," to ", end)

        # Encode the current chunk of food_data and append the embeddings to data_embeddings
        embeddings = encode(food_data[start:end])
        data_embeddings.append(embeddings)

    # Write the updated dataframe to a CSV file
    data_embeddings = torch.cat(data_embeddings, dim=0)
    dish_df['BERT embeddings'] = data_embeddings.tolist()
    dish_df.to_csv('food_embeddings_full.csv', mode='a', header=False, index=False)

def get_restaurant_embeddings(restaurant_df):
    """
    Get and store BERT embeddings for each restaurant in a dataframe.

    Args:
    - restaurant_df (pandas DataFrame): DataFrame containing restaurant name, menu item, and item description columns.

    Returns:
    - None
    """
    # Joining all information from each restaurant together
    restaurant_df['Food Item'] = restaurant_df['Menu Item'].astype(str)+" "+ restaurant_df["Item Description"]
    restaurant_df = restaurant_df.groupby('Restaurant Name').agg({'Food Item': ''.join}).reset_index()

    # Add index number to dataframe, so the restaurant name can be retrieved later
    restaurant_df['index'] = restaurant_df.index
    Indexes = restaurant_df['index'].tolist()

    # Food Data into list format to input into model
    food_data = restaurant_df['Food Item'].tolist()

    # Initialize an empty list for data_embeddings
    data_embeddings = []

    # Specify the chunk size
    chunk_size = 1000

    # Loop over the food_data list in chunks of chunk_size
    for i in range(0, len(food_data), chunk_size):
        # Calculate the start and end indices for the current chunk
        start = i
        end = min(i + chunk_size, len(food_data))
        print(start," to ", end)

        # Encode the current chunk of food_data and append the embeddings to data_embeddings
        embeddings = encode(food_data[start:end])
        data_embeddings.append(embeddings)

    # Write the updated dataframe to a CSV file
    data_embeddings = torch.cat(data_embeddings, dim=0)
    restaurant_df['BERT embeddings'] = data_embeddings.tolist()
    restaurant_df.to_csv('restaurant_embeddings_full.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    # Load Dataset csv
    dish_df = load_dataset_csv('menu_data.csv')
    print(dish_df.head())

    # Get and store BERT embeddings for each dish
    get_food_embeddings(dish_df)

    # Get and store BERT embeddings for each restaurant
    get_restaurant_embeddings(dish_df)
