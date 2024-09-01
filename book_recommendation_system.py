import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
books = pd.read_csv("Books.csv", low_memory=False)
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

# Check for null values and duplicates in datasets
print("Null values in books:", books.isnull().sum())
print("Null values in users:", users.isnull().sum())
print("Null values in ratings:", ratings.isnull().sum())
print("Duplicate entries in books:", books.duplicated().sum())
print("Duplicate entries in users:", users.duplicated().sum())
print("Duplicate entries in ratings:", ratings.duplicated().sum())

# Merge ratings and books datasets
ratings_books = ratings.merge(books, on='ISBN')

# Convert 'Book-Rating' to float, remove invalid entries
ratings_books = ratings_books[pd.to_numeric(ratings_books['Book-Rating'], errors='coerce').notna()]
ratings_books['Book-Rating'] = ratings_books['Book-Rating'].astype(float)

# Popularity-Based Recommendation System

# Calculate the number of ratings per book
num_rating_df = ratings_books.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

# Calculate the average rating per book
avg_rating = ratings_books.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating.rename(columns={'Book-Rating': 'Avg-Rating'}, inplace=True)

# Merge number of ratings and average ratings dataframes
popular_books = num_rating_df.merge(avg_rating, on='Book-Title')

# Filter books with more than 250 ratings and sort by average rating
popular_books = popular_books[popular_books['num_ratings'] >= 250].sort_values('Avg-Rating', ascending=False).head(15)

# Add author and image URL, remove duplicates
popular_books = popular_books.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'Avg-Rating']]

print(popular_books)

# Collaborative Filtering Recommendation System

# Filter users who have rated more than 200 books
x = ratings_books.groupby('User-ID').count()['Book-Rating'] > 200
wanted_users = x[x].index

# Filter ratings by selected users
filtered_ratings = ratings_books[ratings_books['User-ID'].isin(wanted_users)]

# Filter books with at least 50 ratings
y = filtered_ratings.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

# Final ratings dataframe with filtered users and books
final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]

# Create pivot table for collaborative filtering
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Compute cosine similarity between books
similarity_score = cosine_similarity(pt)

# Recommendation function
def recommend(book_name):
    # Fetch index of the book
    index = np.where(pt.index == book_name)[0][0]
    
    # Get indices of items with sorted similarity scores
    similarity_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:7]
    
    # Prepare recommendation data
    data = []
    for i in similarity_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    
    return data

# Example usage: recommend('1984')

# Save data using pickle
pickle.dump(popular_books, open('popular.pkl', 'wb'))
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_score, open('similarity_score.pkl', 'wb'))
