import pandas as pd
import numpy as np
books=pd.read_csv("Books.csv", low_memory=False)
users=pd.read_csv("Users.csv")
ratings=pd.read_csv("Ratings.csv")
books.isnull().sum()
users.isnull().sum()
ratings.isnull().sum()
books.duplicated().sum()
users.duplicated().sum()
ratings.duplicated().sum()
##Popularity based Recommendation System
ratings_books=ratings.merge(books,on='ISBN')
ratings_books = ratings_books[pd.to_numeric(ratings_books['Book-Rating'], errors='coerce').notna()]
ratings_books['Book-Rating'] = ratings_books['Book-Rating'].astype(float)
num_rating_df = ratings_books.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
#print(num_rating)
avg_rating= ratings_books.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating.rename(columns={'Book-Rating':'Avg-Rating'},inplace=True)
#print(avg_rating)
popular_books=num_rating_df.merge(avg_rating,on='Book-Title')
popular_books = popular_books[popular_books['num_ratings']>=250].sort_values('Avg-Rating',ascending=False).head(15)
popular_books = popular_books.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','Avg-Rating']]
print(popular_books)
##Collaborative based Recommendation System
x=ratings_books.groupby('User-ID').count()['Book-Rating']>200
wanted_users=x[x].index
filtered_ratings=ratings_books[ratings_books['User-ID'].isin(wanted_users)]
y=filtered_ratings.groupby('Book-Title').count()['Book-Rating']>=50
#print(y[y])
famous_books=y[y].index
final_ratings=filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]
pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)
from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(pt)
def recommend(book_name):
    #fetch index
    index=np.where(pt.index==book_name)[0][0]
    #get index of item with sorted similarity
    similarity_items=sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:7]
    data=[]
    for i in similarity_items:
        item=[]
        temp_df=books[books['Book-Title']]==pt.index[i[0]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    return data
#recommend('1984')
    
import pickle
pickle.dump(popular_books,open('popular.pkl','wb'))
pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_score,(open('similarity_score.pkl','wb')))