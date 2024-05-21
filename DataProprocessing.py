import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def merge_csv_files():
    file1 = pd.read_csv('ml-latest-small/links.csv')
    file2 = pd.read_csv('ml-latest-small/movies.csv')
    file3 = pd.read_csv('ml-latest-small/ratings.csv')
    file4 = pd.read_csv('ml-latest-small/tags.csv')

    merged_data = pd.merge(file1, file2, on='movieId', how='outer')
    merged_data = pd.merge(merged_data, file3, on='movieId', how='outer')
    merged_data = pd.merge(merged_data, file4, on=['movieId', 'userId'], how='left')

    return merged_data


def print_DataSet(dataSet):
    pd.options.display.max_columns = None
    pd.options.display.expand_frame_repr = False
    pd.options.display.width=None
    print(dataSet.head())


def missing_data(dataSet):
    missing_data = dataSet.isnull().sum()
    print('Dataset null entries: \n' + missing_data.to_string() + '\n')
    print('Dataset entries amount: ' + str(len(dataSet)) + '\n')


def extract_and_drop(Data, list):
    Data = Data.drop(list, axis=1)
    Data['release_year'] = Data['title'].str.extract(r'\((\d{4})\)')
    return Data


def clean_and_replace(field, value, type, DataS):
    DataS[field] = DataS[field].fillna(value)
    DataS[field] = DataS[field].astype(float) #In case it is a string, I want the cast to be done correctly
    DataS[field] = DataS[field].astype(type)


def genres_separation(dataSet):
    dataSet['genres'] = dataSet['genres'].str.split('|')
    unique_lists = dataSet['genres'].explode().unique()
    print(str(unique_lists) + '\n')
    return dataSet


def high_rated_generes_per_year(dataSet):
    filtered_dataSet = dataSet[dataSet['genres'].apply(lambda X: '(no genres listed)' not in X)]

    sampled_years = filtered_dataSet['release_year'].sample(frac=0.0001, random_state=42)

    filtered_dataSet = filtered_dataSet.loc[dataSet['release_year'].isin(sampled_years)]
    filtered_dataSet = filtered_dataSet.explode('genres')

    high_rated_genres_per_year = filtered_dataSet.groupby(['release_year', 'genres'])['rating'].mean().reset_index(). \
        groupby('release_year', as_index=False).apply(lambda x: x.loc[x['rating'].idxmax()])

    genre_list = high_rated_genres_per_year['genres'].tolist()

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=filtered_dataSet, x='release_year', y='rating', hue='genres', errorbar=None)
    plt.title('Rating per Genre over Sampled Release Years')
    plt.xlabel('Release Year')
    plt.ylabel('Rating')
    plt.xticks(rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

    for year, genre in zip(sampled_years.unique(), genre_list):
        print(f'{year}: {genre}')


def rating_for_specific_year(dataSet, year):
    data_2005 = dataSet[dataSet['release_year'] == year]
    data_2005 = data_2005[data_2005['genres'] != '(no genres listed)']
    data_2005 = data_2005.explode('genres')
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=data_2005, x='genres', y='rating', errorbar=None)
    plt.title('Rating per Genre in 2005')
    plt.xlabel('Genre')
    plt.ylabel('Rating')
    
    plt.xticks(rotation=45) 
    
    plt.tight_layout()
    plt.show()


def convert_genres(dataset):
    labels = convert_genres_golumn(dataset['genres'])
    dataset = pd.merge(dataset, labels, left_index=True, right_index=True)
    dataset = dataset.drop('genres', axis=1)
    return dataset


def convert_genres_golumn(column):
    labels = column.str.join(sep='*').str.get_dummies(sep='*')
    return labels


def remove_missing_values(dataset, value):
    rows_to_drop = dataset[dataset.eq(value).any(axis=1)].index
    filtered_dataset = dataset.drop(rows_to_drop)
    return filtered_dataset


def reduce_dataSet(dataset, columns_to_keep = ['userId'] ):
    columns_to_drop = [col for col in dataset.columns if col not in columns_to_keep]
    reducedDataSet = dataset.drop(columns_to_drop, axis=1)
    return reducedDataSet


def get_average_ratings(dataset):
    average_ratings = dataset.groupby('movieId')['rating'].mean().reset_index()
    merged_df = pd.merge(dataset.drop(columns=['rating', 'userId', 'timestamp_x', 'imdbId',  'tmdbId']), average_ratings, on='movieId', how='inner')
    merged_df.drop_duplicates(subset='movieId', inplace=True)
    final_dataset = merged_df[['movieId', 'rating', 'release_year', 'genres']]
    final_dataset.columns = ['movieId', 'average_rating', 'release_year', 'genres']
    final_dataset.loc[:, 'average_rating'] = final_dataset['average_rating'].round(1)
    return final_dataset


def remove_rows_with_zero(dataset, genre):
    condition = (dataset[genre] != 0)
    filtered_dataset = dataset[condition]
    return filtered_dataset