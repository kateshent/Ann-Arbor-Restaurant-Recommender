# evaluate_queries.py


import os
import numpy as np
import torch as torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
from embedding import *

# Load Food dataset
def load_food_dataset():
    dish_df = pd.read_csv(
        'food_embeddings_full.csv')
    dish_df.columns = ['Restaurant Name', 'Menu Item',
                       'Item Description', 'Index', 'Food Item', 'BERT embeddings']
    return dish_df


def load_restaurant_dataset():
    restaurant_df = pd.read_csv('restaurant_embeddings_full.csv')
    return restaurant_df


# Load Test query dataset
def load_query_dataset():
    query_df = pd.read_csv(
        'test_queries.csv')
    return query_df

# Compute cosine similarity between query and all rows in dataset


def get_cosine_similarity_dishes(query_embedding, dish_df):
    cosine_similarities = []
    for dish in dish_df['BERT embeddings']:
        array_elements = dish.strip("[]").split(", ")
        array_list = [float(element) for element in array_elements]
        dishes = torch.tensor(array_list)
        cosine_similarities.append(cosine_similarity(query_embedding, dishes))
    Indexes = dish_df['Index'].tolist()
    dish_score_pairs = list(zip(Indexes, cosine_similarities))
    restaurant_score_pairs = sorted(
        dish_score_pairs, key=lambda x: x[1], reverse=True)
    top_5_pairs = restaurant_score_pairs[:5]
    for i, pair in enumerate(top_5_pairs):
        first_item, second_item = top_5_pairs[i]
        first_item = dish_df.iloc[top_5_pairs[i][0]]['Menu Item']
        top_5_pairs[i] = (first_item, second_item)
    return top_5_pairs


def get_cosine_similarity_restaurants(query_embedding, restaurant_df):
    #Compute cosine similarity between query and all rows in dataset
    cosine_similarities = []
    for restaurant in restaurant_df['BERT embeddings']:

        array_elements = restaurant.strip("[]").split(", ")
        # Convert string elements to floats and create a list
        array_list = [float(element) for element in array_elements]
        # Convert list to torch tensor
        dishes = torch.tensor(array_list)

        # Perform cosine similarity calculation
        cosine_similarities.append(cosine_similarity(query_embedding, dishes))
    Indexes = restaurant_df['index'].tolist()

    #Combine restaurant index & cosine similarity scores
    restaurant_score_pairs = list(zip(Indexes, cosine_similarities))

    #Sort by decreasing score
    restaurant_score_pairs = sorted(restaurant_score_pairs, key=lambda x: x[1], reverse=True)
    # Get the top 5 pairs
    top_5_pairs = restaurant_score_pairs[:5]

    for i, pair in enumerate(top_5_pairs):
        #top_5_pairs[i][0] = dish_df.iloc[restaurant_score_pairs[0][0]]['Menu Item']

        # Unpack the tuple
        first_item, second_item = top_5_pairs[i]

        # Update the value of the second item
        first_item = restaurant_df.iloc[top_5_pairs[i][0]]['Restaurant Name']

        # Create a new tuple with the updated value
        top_5_pairs[i] = (first_item, second_item)

    return top_5_pairs


# Get BERT embeddings for queries
def get_query_embeddings(query_df):
    queries = query_df['Query'].tolist()
    query_embeddings = encode(queries)
    query_df['BERT embeddings'] = query_embeddings.tolist()
    query_embeddings = query_df['BERT embeddings'].tolist()
    return query_df, query_embeddings


# Save query results to a CSV file
def save_query_results(query_df):
    query_df.to_csv('query_results.csv',
                    mode='a', header=False, index=False)


def dish_evaluations_csv(query_df, column_name):    
    dish_df = load_food_dataset()
    query_df, query_embeddings = get_query_embeddings(query_df)
    query_df[column_name] = None
    for i, query in enumerate(query_embeddings):
        query_df.at[i, column_name] = get_cosine_similarity_dishes(
            query, dish_df)
    return query_df


def restauarant_evaluations_csv(query_df, column_name):
    restaurant_df = load_restaurant_dataset()
    query_df, query_embeddings = get_query_embeddings(query_df)
    query_df[column_name] = None
    for i, query in enumerate(query_embeddings):
        query_df.at[i, column_name] = get_cosine_similarity_restaurants(
            query, restaurant_df)
    return query_df



def main():
    query_df = load_query_dataset()

    # Change column name when trying different metrics, record in readMe test metrics table
    # Changes will need to be made in embedding.py to change model and pooling method
    #call different similarity function in this file

    #query_df = dish_evaluations_csv(query_df, 'Test1 dishes']
    query_df = restauarant_evaluations_csv(query_df, 'Test1 restaurants')
    save_query_results(query_df)


if __name__ == "__main__":
    main()
