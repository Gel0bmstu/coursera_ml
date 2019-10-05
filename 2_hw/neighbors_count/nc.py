import numpy as np
import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Читам файл с данными о вине в dataFrame 'wine'
wine = pd.read_csv('wine.csv', names=['WineClass', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline'])

# Разделяем dataFrame на объекты 'WineClass' и признаки объектов 'WineSign'
wineClass = wine['WineClass']
wineSign = wine.loc[:, wine.columns != 'WineClass']

# Нормализуем матрицу признаков объектов таким образом, чтобы
# каждый ее столбец имеел нулевое среднее значение и
# единичное стандартное отклонение.
# Раскимментить для выполения 3-го и 4-го заданий
# wineSign = sk.preprocessing.scale(wineSign)

maxAccuracy = -1
maxAccuracyNeigbourhoodsCout = -1

# Перебираем количество соседей, чтобы получить максимальную точность.
for i in range(1, 50):
    # Создаем классификатор, задаем для него количество соседей.
    classifier = KNeighborsClassifier(i)
    # Создаем генератор разбиений.
    generator = KFold(n_splits = 5, shuffle=True, random_state=42)
    # Определяем качество алгоритма для данного классификатора.
    score = cross_val_score(X = wineSign, 
                            y = wineClass, 
                            cv = generator, 
                            estimator = classifier)

    # Находим максимальную точность алгоритма и число соседей 
    # классфикатора, при котором достигается эта точность.
    if (np.mean(score) > maxAccuracy):
        maxAccuracy = np.mean(score)
        maxAccuracyNeigbourhoodsCout = i
            
print(maxAccuracy, maxAccuracyNeigbourhoodsCout)        
