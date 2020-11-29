#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']


# In[5]:


print(ratings.shape)
print(list(ratings.columns))


# In[11]:


plt.rc("font", size=15)
ratings.bookRating.value_counts(sort=False).plot(kind='line')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('system1.png', bbox_inches='tight')
plt.show()


# In[12]:


print(books.shape)
print(list(books.columns))


# In[13]:


print(users.shape)
print(list(users.columns))


# Removing users with less than 50 ratings and Books with Less than 300 Ratings

# In[14]:


counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 50].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 300].index)]


# Using colaborative filtering model to reccomend books based on knn (A machine lerning algorithm which makes reccomendation based on nearest k neighbour

# In[31]:


combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
combine_book_rating.head()


# Grouping by book titles and creating a new column for data count

# In[18]:


combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

book_ratingCount = (combine_book_rating.
     groupby(by = ['bookTitle'])['bookRating'].
     count().
     reset_index().
     rename(columns = {'bookRating': 'totalRatingCount'})
     [['bookTitle', 'totalRatingCount']]
    )
book_ratingCount.head()


# In[32]:


rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
rating_with_totalRatingCount.head()


# In[20]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_ratingCount['totalRatingCount'].describe())


# In[30]:


print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))


# In[33]:


popularity_threshold = 75
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_book.head()


# In[34]:


rating_popular_book.shape


# # Since the data is particularly large, i am going to filter it to users of australia 

# In[36]:


combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')

australia_user_rating = combined[combined['Location'].str.contains("australia")]
australia_user_rating.head()


# Implementing KNN

# In[48]:


from scipy.sparse import csr_matrix
australia_user_rating = australia_user_rating.drop_duplicates(['userID', 'bookTitle'])
australia_user_rating_pivot = australia_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
australia_user_rating_matrix = csr_matrix(australia_user_rating_pivot.values)

from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(australia_user_rating_matrix)


# In[51]:


australia_user_rating_pivot.iloc[query_index,:].values.reshape(1,-1)


# In[64]:


query_index = np.random.choice(australia_user_rating_pivot.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(australia_user_rating_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)


# In[52]:


australia_user_rating_pivot.index[query_index]


# In[60]:


for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(australia_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, australia_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


# In[ ]:




