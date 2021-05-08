import numpy as np
import pandas as pd
#from sklearn import neighbors
#from sklearn.model_selection import train_test_split


# For your assignment, you have implement function "train_test_split" and
# module "neighbors" on your own. This means you will have to make the code working
# without any modification of lines 14-30.

df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

h = neighbors.KNeighborsClassifier(n_neighbors=7, p=2)
h.fit(x_train, y_train)

new_x = np.array([4,6,5,6,7,8,4,9,1])
result = h.predict(new_x.reshape(1, -1))
print(result)

print(h.score(x_test, y_test))
