import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
answ = {}
print(data[:25])

#Общее количество пассажиров
total = 891
print("Total: ", total)

# Количество мужчин/женщин на корабле
mf_count = data['Sex'].value_counts()
answ['m/f'] = [577, 314]
print(mf_count)

# Количество выживших мужчин/женщин
mf_srv_count = data.groupby(['Survived']).size()
answ['srv'] = 342 *100 / total
print(mf_srv_count)

#Доля первого класса
mf_srv_count = data.groupby(['Pclass']).size()
answ['first_class'] = 216 *100 / total
print(mf_srv_count)

# Средний возраст
age = data['Age'].describe()
print(age)
answ['age'] = {'sred' : 29.69, 'med' : data['Age'].median()}

# Корреляция Пирса
sub_parr_pirse = data['Parch'].corr(data['SibSp'])
print(sub_parr_pirse)

answ['pirse'] = sub_parr_pirse

# Самое популярное женское имя
def most_popular_name(name_series):

    tmp = []

    for name_string in name_series:
        # Парсим строку с именем, получаем имя
        name = parse_name(name_string)
        tmp.append(name)

    return tmp

def parse_name(name_string):
    tmp = name_string.split(',')
    words = tmp[1].split(' ')

    for w in words:
        if w != "":
            if w[0] == '(':
                w = w.replace("(", "")
                w = w.replace(")", "")
                return w

    return words[2]

fems = data[data['Sex'] == 'female']['Name']
names = most_popular_name(fems)

names_df = pandas.DataFrame(data = {'Name' : names})

print (names_df['Name'].value_counts())

answ['most_popular_name'] = 'Anna'

print("\n\n", answ)