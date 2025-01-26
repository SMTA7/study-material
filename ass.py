import pandas as pd
import numpy as np
from AutoClean import AutoClean
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage , dendrogram
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.cluster import AgglomerativeClustering
from clusteval import clusteval
import sweetviz
import matplotlib.pyplot as plt
df=pd.read_csv(r"D:/DATA SCIENCE/Assignment/Data Set (5)/AirTraffic_Passenger_Statistics.csv")
df1 = df[["Operating Airline", "GEO Region", "Passenger Count"]]
airline_count = df1["Operating Airline"].value_counts()
airline_count.sort_index(inplace=True)
passenger_count = df1.groupby("Operating Airline").sum()["Passenger Count"]
passenger_count.sort_index(inplace=True)
df2 = pd.concat([airline_count, passenger_count], axis=1)





my_report.show_html('Report.html')

cols=list(df.columns)
df1=df['Passenger Count']
print(df_clean.columns)
cleaned_pipeline=AutoClean(df.iloc[:,6:],mode='manual',missing_num='auto',outliers='winz',encode_categ='auto')
df_clean=cleaned_pipeline.output
df_clean.drop(['Year', 'Month', 'Month_lab'],axis=1,
              inplace=True)

cols=list(df_clean.columns)
pipe=make_pipeline(MinMaxScaler())
df_pipe=pd.DataFrame(pipe.fit_transform(df_clean),columns=cols,index=df_clean.index)

plt.figure(1,figsize=(16,8))

tree_plot = dendrogram(linkage(df_pipe,method='complete'))
plt.title("Hierarchical clustering")
plt.xlabel("Index")
plt.ylabel('Euclidean Distance')
plt.show()

ce = clusteval(evaluate='silhouette')
df_array = np.array(df_pipe)
ce.fit(df_array)
ce.plot()
plt.show()

hc1=AgglomerativeClustering(n_clusters=2, metric='euclidean',linkage='ward')
y_hc1=hc1.fit_predict(df_pipe)
hc1.labels_
cluster_labels = pd.Series(hc1.labels_) 
df_clust = pd.concat([cluster_labels,df_clean] ,axis=1)
df_clust = df_clust.rename(columns={0: 'cluster'})

metrics.silhouette_score(df_pipe, cluster_labels)
df_clust.iloc[:, 1:].groupby(df_clust.cluster).mean()
final = pd.concat([df["Operating Airline"], df_clust], axis=1)


print(df.columns)
