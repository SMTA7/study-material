import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import sweetviz
from AutoClean import AutoClean
from scipy.cluster.hierarchy import linkage , dendrogram
from sklearn.cluster import AgglomerativeClustering
from clusteval import clusteval
from sqlalchemy import create_engine , text
import pymysql
from sklearn import metrics
from  urllib.parse import quote
import dtale
univ = pd.read_excel(r"D:/DATA SCIENCE/Study Material/University_Clustering.xlsx")

user = 'root'
pw = quote('smta')
db = 'univ'
server='localhost'
engine=create_engine(f'mysql+pymysql://{user}:{pw}@{server}/{db}')

univ.to_sql('univ_tbl',con=engine,if_exists='replace',chunksize=1000,index=False)
sql="select * from univ_tbl;"

df=pd.read_sql_query(sql,engine)

univ.drop('UnivID',axis=1,inplace=True)
df.info()

my_report=sweetviz.analyze([df,'df'])
my_report.show_html('Report.html')


d=dtale.show(df)
d.open_browser()

univ.plot(kind='box',subplots=True,sharey=False,figsize=(15,8))
plt.subplots_adjust(wspace=0.9)
plt.show()

cleaned_pipeline=AutoClean(univ.iloc[:,1:],mode='manual',missing_num='auto',outliers='winz',encode_categ='auto')
df=cleaned_pipeline.output

df.drop('State',axis=1,inplace=True)

cols=list(df.columns)
pipe=make_pipeline(MinMaxScaler())
df_pipe=pd.DataFrame(pipe.fit_transform(df),columns=cols,index=df.index)


## model building

plt.figure(1,figsize=(16,8))

tree_plot = dendrogram(linkage(df_pipe,method='complete'))
plt.title("Hierarchical clustering")
plt.xlabel("Index")
plt.ylabel('Euclidean Distance')
plt.show()

hc1=AgglomerativeClustering(n_clusters=4, metric='euclidean',linkage='single')
y_hc1=hc1.fit_predict(df_pipe)
hc1.labels_
cluster_labels = pd.Series(hc1.labels_) 
df_clust = pd.concat([cluster_labels,df] ,axis=1)
df_clust.columns
df_clust = df_clust.rename(columns={0: 'cluster'})

## cluster evaluation

metrics.silhouette_score(df_pipe, cluster_labels)


ce = clusteval(evaluate='silhouette')
df_array = np.array(df_pipe)
ce.fit(df_array)
ce.plot()
plt.show()
# hyperparameter change  if n_cluster=2 ,linkage='single'

hc2=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='single')
y_hc2=hc2.fit_predict(df_pipe)
hc2.labels_
cluster_labels2 = pd.Series(hc2.labels_)
df_clust2=pd.concat([cluster_labels2,df],axis=1)
df_clust2=df_clust2.rename(columns={0:'cluster'})

metrics.silhouette_score(df_pipe, cluster_labels2)

# hyperparameter change  if n_cluster=2 ,linkage='complete'


hc3=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='complete')
y_hc3=hc3.fit_predict(df_pipe)
cluster_labels3=pd.Series(hc3.labels_)
metrics.silhouette_score(df_pipe, cluster_labels3)
df_label3=pd.concat([cluster_labels3,df],axis=1)
df_label3=df_label3.rename(columns={0:'cluster'})

# hyperparameter change  if n_cluster=2 ,linkage='average'
hc4=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='average')
y_hc4=hc4.fit_predict(df_pipe)
cluster_labels4=pd.Series(hc4.labels_)
metrics.silhouette_score(df_pipe, cluster_labels4)

# hyperparameter change  if n_cluster=2 ,linkage='centriod'


hc5=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward')
y_hc5=hc5.fit_predict(df_pipe)
cluster_labels5=pd.Series(hc5.labels_)
metrics.silhouette_score(df_pipe, cluster_labels5)

df_label3.iloc[:, 1:7].groupby(df_label3.cluster).mean()
final = pd.concat([univ.Univ,univ.State,df_label3], axis=1)
