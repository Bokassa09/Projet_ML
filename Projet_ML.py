# étape 1 : Lecture des données
# importer la lib pandas pour le traitement sur le jeux de données
import pandas as pd

# read the csv-formatted data file into a pandas dataframe
df=pd.read_csv('diabetes.csv')
# get shape of data frame
print('Shape (n_rows,n_columns) of dataframe:',df.shape)
# print top 5 rows of data frame
print(df.head(5))
