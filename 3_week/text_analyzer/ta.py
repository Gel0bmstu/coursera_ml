from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
            )

# Выделяем в отдельные переменные масив текстов и номер класса.
targets = newsgroups.target
texts = newsgroups.data

# Создадим объект класса векторайзера и преобразуем обучающую выборку.
vectorizer = TfidfVectorizer()

# Найдем числовоее представление текстовых данных:
signsScaled = vectorizer.fit_transform(texts)

cv = KFold(n_splits=5, shuffle=True, random_state=241)
classifier = SVC(kernel='linear', random_state=241)

grid = {'C': np.power(10.0, np.arange(-5, 6))}

# Обучем SVM по всей выборке с оптимальным параметром C 
gs = GridSearchCV(classifier, grid, scoring='accuracy', cv=cv)
gs.fit(signsScaled, targets)

C = gs.best_params_['C']
print(C)

classifier = SVC(C=C, kernel='linear', random_state=241)
classifier.fit(signsScaled, targets)

feature_mapping = vectorizer.get_feature_names()

word_indicies = np.argsort(np.abs(np.asarray(classifier.coef_.todense())).reshape(-1))[-10:]
words = [feature_mapping[i].encode('utf-8') for i in word_indicies]

print("10 words with the highest absolute weight value:")
print(words.sort())