import pandas as pd
import numpy as np
import math
import operator


from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler 

import os
import gensim
from gensim import models
from gensim.models import Word2Vec, KeyedVectors
from gensim import corpora

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#### Start of STEP 1
# Importing data 
with open('myrecipe.csv', 'r') as csv_file:
    data = pd.read_csv(csv_file)
#### End of STEP 1

ingredients2 = ['xyz']
ingredients = data["Ingredients"].tolist()
for i in range(12351):
    ingredients1 = ingredients[i].split(',')
    ingredients2.extend(ingredients1)
ingredients2.remove('xyz')

mylist = list(dict.fromkeys(ingredients2))
mylist.sort()


resultrecipe = []
for item in mylist:
  if mylist.index(item) > mylist.index('almond'):
    resultrecipe.append(item)
resultrecipe.append('almond')
resultrecipe.sort()
#print(len(resultrecipe))

'''
for i in range(437):
    if data['Ingredients'].str.contains(resultrecipe[i]) is False:
        data.drop(indexNames , inplace=True)
print(data)'''

recipe = data["Recipe_Name"].tolist()
'''
#print (data[['Recipe_Name', 'Ingredients']])
#print(data.loc[data['Ingredients'].str.contains("egg") , 'Recipe_Name'])

d = {}
array=list
for i in range(943):
    array = data.loc[data['Ingredients'].str.contains(mylist[i]) , 'Recipe_Name']
    d.setdefault(mylist[i], []).append(array)
#print(d) 
drecipe = {}
array1=list
for i in range(12351):
    array1 = data.loc[data['Recipe_Name'] == recipe[i], 'Ingredients']
    ingr= array1[i].split(',')
    drecipe.setdefault(recipe[i], []).append(ingr)
#print(drecipe)


review = data["Review_Count"].tolist()
author = data["Author"].tolist()
prepare = data["Prepare_Time"].tolist()
cook = data["Cook_Time"].tolist()
total = data["Total_Time"].tolist()
directions = data["Directions"].tolist()

#print(tuple(d.items())[14])
'''
drecipe = {}
array1=list
for i in range(12351):
    array1 = data.loc[data['Recipe_Name'] == recipe[i], 'Ingredients']
    ingr= array1[i].split(',')
    drecipe.setdefault(recipe[i], []).append(ingr)
#print(drecipe)
from sklearn.preprocessing import MultiLabelBinarizer
# Instantiate the binarizer
mlb = MultiLabelBinarizer()

df = data.apply(lambda x: x["Ingredients"].split(","), axis=1)
# Transform to a binary array
array_out = mlb.fit_transform(df)

df_out = pd.DataFrame(data=array_out, columns=mlb.classes_)

#print(df_out)
#merged = pd.concat([data['Recipe_Name'] , df_out], axis='columns')
#print(mylist)

out = ['xyz']
for item in mylist:
  if mylist.index(item) < mylist.index('almond'):
    str = item.split(',')
    out.extend(str)
out.remove('xyz')
#print(len(out))
#print(len(mylist))

ml=df_out.iloc[:, 0:506]
final = df_out.drop(ml, axis='columns' , inplace=True)


merged = pd.concat([data['Recipe_Name'] , df_out], axis='columns') # recipe and all ingredients dataframe


#finalmerged = merged.drop('Ingredients', axis='columns' )
#print(merged.tail())


new_ing = [ 'potato','onion','tomato','broccoli']
length = len(new_ing)
naya = 437 - length

list1 = [None] * naya        #populate list, length n with n entries "None"

for i in range(length):
  list1.insert(i,new_ing[i])         #redefine list as the last n elements of list
#print(len(list1))


array = [None] * 437 
for i in range(len(resultrecipe)):
  #print(i)
  if list1[i] != None:
    stri = list1[i]
    location = resultrecipe.index(stri)
  if stri in resultrecipe:
    array[location] = 1
  else:
    array[location] = 0


for i in range(437):
  if array[i] == None:
    array[i] = 0
#print(array)   #final vector form of given ingredients

#x1 = merged.iloc[:, merged.columns != 'Recipe_Name'].as_matrix()

#print(array)
#print(x1[11])

def cos_sim(a , b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

#print(cos_sim(array, x1[11]))
forcalc = []
resultmaybe = []
z=[]
for i in range(12351):
  forcalc = merged.iloc[i, ~merged.columns.isin(['Recipe_Name'])]
  z1 = cos_sim(array, forcalc)
  z.append(z1)
  

cloud = dict( zip( recipe, z))
#print( recipe)
from collections import OrderedDict
d_sorted_by_value = OrderedDict(sorted(cloud.items(), key=lambda x: x[1], reverse=True))


for key, value in dict(d_sorted_by_value).items():
        if value == 0.0:
            del d_sorted_by_value[key]

#print( d_sorted_by_value)  #dictionary with similar recipe and similarity
#print( len(d_sorted_by_value))
review = []
author = []
lastrecipe = []
for key, value in dict(d_sorted_by_value).items():
    lastrecipe.append(key)
    review.append(data.loc[data['Recipe_Name'] == key, 'Review_Count'])
    author.append(data.loc[data['Recipe_Name'] == key, 'Author'])
    
cloudlasts = []
for n, j, s in zip(lastrecipe, author, review):
  cloudlast = { 'lastrecipe': n, 'author': j, 'review': s }
  cloudlasts.append(cloudlast)
print(cloudlasts)