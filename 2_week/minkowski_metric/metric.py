import numpy as np
import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor

# Загружаем выборку Boston
boston = datasets.load_boston()

# Приведем признаки в выборке к одному масштабу
bostonNorm = sk.preprocessing.scale(boston.data)

maxAccuracy = -199999999999999999999999
maxAccuracyP = -1999999999999999999999999999

# Создаем генератор разбиений.
generator = KFold(n_splits = 5, shuffle=True, random_state=42)

rangeArr = np.linspace(1,10,200)

# Перебираем количество соседей, чтобы получить максимальную точность.
for i in rangeArr:

    # Создаем классификатор
    classifier = KNeighborsRegressor(n_neighbors=5, weights='distance', p=i, metric='minkowski')

    # Определяем качество алгоритма для данного классификатора.
    score = (cross_val_score(X = bostonNorm, 
                            y = boston.target, 
                            cv = generator, 
                            scoring='neg_mean_squared_error',
                            estimator = classifier)).mean()

    # Находим максимальную точность алгоритма и коэффициент p 
    # классфикатора, при котором достигается эта точность.
    if (np.mean(score) > maxAccuracy):
        maxAccuracy = np.mean(score)
        maxAccuracyP = i
            
print(maxAccuracy, maxAccuracyP)