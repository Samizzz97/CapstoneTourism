# %% [markdown]
# # Import all the datasets

# %%
import pandas as pd

places_df = pd.read_csv('tourism_with_id.csv')
users_df = pd.read_csv('user.csv')
ratings_df = pd.read_csv('tourism_rating.csv')

# %% [markdown]
# # Display dataframes created 

# %%
places_df.head(5)

# %%
users_df.head(2)

# %%
ratings_df.head(3)

# %% [markdown]
# # Preliminary inspection

# %%
print(places_df.info())

# %%
print(users_df.info())

# %%
print(ratings_df.info())

# %% [markdown]
# # Check for nulls and duplicates

# %%
print(places_df.isnull().sum())
print(users_df.isnull().sum())
print(ratings_df.isnull().sum())

# %%
print(places_df.duplicated().sum())
print(users_df.duplicated().sum())
print(ratings_df.duplicated().sum())

# %% [markdown]
# # 1b. Remove any anomalies found in the data 

# %%
places_df.drop(columns=['Time_Minutes', 'Unnamed: 11','Unnamed: 12','Coordinate','Lat','Long'],inplace=True)

# %%
places_df

# %% [markdown]
# Explore the data in depth to understand the tourism patterns. 
# a. Explore the user group that provides the tourism ratings by answering the following questions:
# i. The age distribution of users visiting the places and giving the ratings.
# ii. Where are most of these users (tourists) coming from?
# 

# %%
users_df['Age']

# %%
users_df['Location']

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(users_df['Age'], kde=True, bins=20)
plt.title('Age Distribution of Tourists')

# %%
print(users_df['Age'].mode())

# %%
print(users_df['Age'].mean())

# %%
users_df['Location'].value_counts()

# %%
sns.countplot(data=users_df, y='Location', order=users_df['Location'].value_counts().index)
plt.title('Where most of the tourists are from')

# %% [markdown]
# b. Explore the locations and categories of tourist spots by answering the following questions:
# i. What are the different categories of tourist spots?
# 

# %%
print(places_df['Category'].value_counts())

# %%
sns.countplot(data=places_df, y='Category', order=places_df['Category'].value_counts().index)
plt.title('Number of Places by Category')

# %% [markdown]
# ii. What kind of tourism is each city or location most famous or suitable for?

# %%
pd.crosstab(places_df['City'], places_df['Category']).plot(
    kind='bar',
    stacked=True,
    figsize=(10, 5)
)
plt.title('Tourism by City')
plt.tight_layout()
plt.show()


# %%
category_counts = pd.crosstab(places_df['City'], places_df['Category'])
print(category_counts)

# %% [markdown]
# iii. Which city would be best for a nature enthusiast to visit?

# %%
nature_spots = places_df[places_df['Category'].str.contains('Cagar Alam')]
print(nature_spots['City'].value_counts())

# %%
plt.figure(figsize=(10, 5))
sns.countplot(data=nature_spots, x='City', order=nature_spots['City'].value_counts().index)
plt.title('Number of Cagar Alam Spots per City')
plt.ylabel('Count')
plt.xlabel('City')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# iv. What is the average price or cost of these places?

# %%
city_prices = places_df.groupby('City')['Price'].mean().sort_values()

# %%
city_prices

# %%
sns.barplot(x=city_prices.index,y=city_prices.values)

# %% [markdown]
# c. Create combined data with places and their user ratings.
# 

# %%
ratings_df.head(2)

# %%
places_df

# %%
merged_df = ratings_df.merge(places_df, on='Place_Id')
merged_df

# %% [markdown]
# d. Use this data to figure out the spots that the tourists most love. Which city has the most loved tourist spots?
# 

# %%
top_places = merged_df.groupby(['Place_Name'])['Rating'].mean().sort_values(ascending=False).head(10)
print(top_places)

# %%
most_loved_city = merged_df.groupby('City')['Rating'].mean().sort_values(ascending=False).head(10)

# %%
most_loved_city

# %%
most_loved_cat = merged_df.groupby('Category')['Rating'].mean().sort_values(ascending=False)

# %%
most_loved_cat

# %% [markdown]
# 3. Build a Recommendation model for the tourists.
# a.  Use the above data to develop a collaborative filtering model for recommendation. 
# Use that to recommend other places using the current tourist location (place name).
# 

# %%
merged_df.head(10)

# %%
from sklearn.metrics.pairwise import cosine_similarity

# Create User-Item matrix
user_item_matrix = merged_df.pivot_table(index='User_Id', columns='Place_Name', values='Rating')

user_item_matrix.head(5)

# %%
user_item_matrix_filled = user_item_matrix.fillna(user_item_matrix.mean())

user_item_matrix_filled.head(5)

# %% [markdown]
# Item Comparison for Item-Based
# 

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

user_item_matrix_scaled = pd.DataFrame(scaler.fit_transform(user_item_matrix_filled.T).T,columns=user_item_matrix_filled.columns)

# %%
# Compute cosine similarity between items (places)
item_similarity = cosine_similarity(user_item_matrix_scaled.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix_filled.columns, columns=user_item_matrix_filled.columns)

# %%
item_similarity_df.head(10)

# %% [markdown]
# Item-Based (location based)

# %%
def get_similar_places(Place_Name, top_n=5):

    Place_Name = Place_Name.strip().title()
    
    if Place_Name not in item_similarity_df.columns:
        return "Place not found."
    
    # Get the similarity scores
    similar_scores = item_similarity_df[Place_Name].sort_values(ascending=False)
    
    # Exclude the place itself
    similar_scores = similar_scores.drop(Place_Name)
    
    # Get top N similar places
    top_places = similar_scores.head(top_n)

    recommendations = []
    for place, score in top_places.items():
        location = merged_df.loc[merged_df['Place_Name'] == place, 'City'].values
        location = location[0] if len(location) > 0 else "Unknown"
        recommendations.append((place, score, location))
    
    # Create a DataFrame
    rec_df = pd.DataFrame(recommendations, columns=['Place_Name', 'Similarity_Score', 'Location'])

    return rec_df

# Prompt user
user_input = input("\nEnter the name of a destination you're interested in: ")
results = get_similar_places(user_input)

# Show results
if isinstance(results, list):
    print("\nYou might also enjoy visiting:")
    for name, score, location in results:
        print(f"- {name} (Similarity: {score:.2f}, Location: {location})")
else:
    print(f"\n{results}")


# %%



