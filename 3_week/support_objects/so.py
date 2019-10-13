import pandas as pd
from sklearn.svm import SVC

# Читаем csv файл.
sample = pd.read_csv('svm-data.csv',  
                      names = ['Target', 'Sign1', 'Sign2'],
                      header = None)

# Выделяем в отдельные переменные целевые переменные и признаки выборки.
svmTarget = sample['Target']
svmSigns = sample.loc[:, sample.columns != 'Target']

# Создаем классификатор с методом опорных векторов.
classifier = SVC(kernel='linear', C = 100000, random_state=241)

# Обучаем классификатор на выборке.
classifier.fit(y = svmTarget, X = svmSigns)

# ВНИМАНИЕ! Индексация элементов опорных объектов не совпадает с
# индексацией, требуемой в курсе coursera. Объекты классификатора
# индексируются с 0, по заданию нужно указать индексы опорных объектов,
# начинающихся с 1, следоватьльно инкриментируем полученные номера
# индексов на 1.

# Выделим массив под ответы
answer = [] 

# Получаем коррекные индексы опорных объектов классификатора
for i in range(len(classifier.support_)):
    answer.append(classifier.support_[i] + 1)

print(answer)