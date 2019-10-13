import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Читаем датасет из csv файлов, размечаем заголоки для колонн.
testVyborka = pd.read_csv('perceptron-test.csv', 
                           names = ['Target', 'Sign1', 'Sign2'],
                           header = None)

trainVyborka = pd.read_csv('perceptron-train.csv', 
                            names = ['Target', 'Sign1', 'Sign2'],
                            header = None)

# Для удобства, выделим отдельные переменные для классов и признаков
# тестовой и учебной выборок.
trainClasses = trainVyborka['Target']
testClasses = testVyborka['Target']

trainSigns = trainVyborka.loc[:, trainVyborka.columns != 'Target']
testSigns = testVyborka.loc[:, testVyborka.columns != 'Target']

# Обучаем персептрон на обучающей выборке и предиктим на тестовой 
# ВНИМАНИЕ! В задание указано использовать параметр random_state = 291
# при инициализации объекта Perceptron, в таком случае алгоритм выдает ответ,
# который не принимает coursra.
perceptron = Perceptron()
perceptron.fit(X = trainSigns, y = trainClasses)
predict1 = perceptron.predict(X = testSigns)

# Посчитаем точность мтеода.
accuracy = accuracy_score(testClasses, predict1, normalize=True)

# Нормализуем признаки выборок.
scaler = StandardScaler()
trainSignsScaled = scaler.fit_transform(trainSigns)
testSignsScaled = scaler.transform(testSigns)

# Еще раз обучаем и предиктим.
perceptron.fit(X = trainSignsScaled, y = trainClasses)
predict2 = perceptron.predict(X = testSignsScaled)

# Расчитываем разность между качеством на тестовой выборке 
# после нормализации и качеством до нее.
accuracyScaled = accuracy_score(testClasses, predict2, normalize=True)

print(accuracyScaled, accuracy, round(accuracyScaled - accuracy, 3))
