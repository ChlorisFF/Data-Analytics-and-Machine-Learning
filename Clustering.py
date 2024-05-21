from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import DataProprocessing as dp


# region Release_year AND Rating
def kmeans_func_1(dataset, clusters): 
    Test1=dp.reduce_dataSet(dataset, ['release_year', 'rating'])
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(Test1)
    gfg = sns.scatterplot(data = Test1, x = 'release_year', y = 'rating', hue = kmeans.labels_)
    gfg.set_xlim(1905, 2024)
# endregion
    
# region Release_year AND userId
def kmeans_func_2(dataset, clusters): 
    Test2=dp.reduce_dataSet(dataset, ['release_year', 'userId'])
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(Test2)
    gfg = sns.scatterplot(data = Test2, x = 'release_year', y = 'userId', hue = kmeans.labels_)
    gfg.set_xlim(1905, 2024)
# endregion
    
 # region Release_year AND Rating for specific genre
def kmeans_Genre(dataset, clusters, genre): 
    Test1=dp.reduce_dataSet(dataset, ['userId', 'rating', genre])
    Test1=dp.remove_rows_with_zero(Test1,genre)
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(Test1)
    gfg = sns.scatterplot(data = Test1, x = 'userId', y = 'rating', hue = kmeans.labels_)
# endregion
    
# region Release_year AND Rating
def dbscan_func_1(dataset, distance = 0.5, minsamples = 5): 
    Test1=dp.reduce_dataSet(dataset, ['release_year', 'rating'])
    dbscan = DBSCAN(eps = distance, min_samples = minsamples, metric='manhattan')
    dbscan.fit(Test1)
    gfg = sns.scatterplot(data = Test1, x = 'release_year', y = 'rating', hue = dbscan.labels_)
    gfg.set_xlim(1905, 2024)
# endregion

# region Release_year AND Rating
def compare_func(dataset, distance = 0.5, minsamples = 5, clusters = 4): 

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    dbscan_func_1(dataset, distance , minsamples)
    plt.title('DBSCAN')
    
    plt.subplot(1, 2, 2)
    kmeans_func_1(dataset, clusters)
    plt.title('KMEANS')

    plt.tight_layout()
    plt.show()  
# endregion  