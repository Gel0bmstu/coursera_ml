import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Читаем csv файл
data = pd.read_csv('titanic.csv', index_col='PassengerId')
# Оставляем только 4 коллонки
vyborka = data.drop(columns = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'])
# Уаляем все строки, которые содержат NaN
vyborka = vyborka.dropna()

vyborka['Sex'] = vyborka['Sex'].str.replace('female', '0')
vyborka['Sex'] = vyborka['Sex'].str.replace('male', '1')

target = data.dropna()

Y = vyborka.Survived
X = vyborka[['Pclass', 'Fare', 'Age', 'Sex']]

clf = DecisionTreeClassifier()
clf.fit(X, Y)
importances = clf.feature_importances_

print (clf.feature_importances_)

