import pandas as pd
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

os.chdir("C:\\Users\\mcast_000\\Documents\\Projects\\Health")

### WHO dataset on child mortality
child_mortality = pd.read_csv("child_mortality_country_who.csv")

### various mortality stats
print(child_mortality.Indicator.value_counts())
print(child_mortality.Sex.value_counts())
print(child_mortality.Year.value_counts()) ### we should use 2009 for year of reference

neonatal = child_mortality[child_mortality.Indicator.str.contains("Neonatal")]
underfive = child_mortality[child_mortality.Indicator.str.contains("Under-five")]
infant = child_mortality[child_mortality.Indicator.str.contains("Infant")]
### only dataset w/one obs per country
stillbirth = child_mortality[child_mortality.Indicator.str.contains("Stillbirth")]

neonatal = pd.DataFrame(neonatal,
                        columns = ['WHO region', 'World Bank income group', 'Country', 'Year', 'Numeric']).rename(columns = {'Numeric' : 'neonatal'})
underfive = pd.DataFrame(underfive,
                         columns = ['WHO region', 'World Bank income group', 'Country', 'Year', 'Numeric']).rename(columns = {'Numeric' : 'underfive'})
infant = pd.DataFrame(infant,
                      columns = ['WHO region', 'World Bank income group', 'Country', 'Year', 'Numeric']).rename(columns = {'Numeric' : 'infant'})
stillbirth = pd.DataFrame(stillbirth,
                          columns = ['WHO region', 'World Bank income group', 'Country', 'Year', 'Numeric']).rename(columns = {'Numeric' : 'stillbirth'})

neonatal_2009 = neonatal[neonatal.Year == 2009]
underfive_2009 = underfive[underfive.Year == 2009]
infant_2009 = infant[infant.Year == 2009]

mortality = neonatal_2009.merge(underfive_2009).merge(infant_2009).merge(stillbirth)

mort_numeric = pd.DataFrame(mortality, columns = ['neonatal', 'underfive', 'infant', 'stillbirth'])

### perform hierarchical clustering
ac = AgglomerativeClustering(n_clusters = 16, linkage = 'ward')
hclus = ac.fit_predict(mort_numeric)

### perform kmeans clustering
km = KMeans(n_clusters = 16)
kclus = km.fit_predict(mort_numeric)

mortality['hclus'] = hclus.tolist()
mortality['kclus'] = kclus.tolist()

### see hierarchical clustering results
for i in range(16):
    print(mortality[mortality.hclus == i])
### see kmeans clustering results
for i in range(16):
    print(mortality[mortality.kclus == i])