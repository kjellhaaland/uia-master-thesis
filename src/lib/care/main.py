import pandas as pd
import care

df1 = pd.read_csv('data.csv', delimiter=';')
df2 = pd.read_csv('data.csv', delimiter=';')
df3 = pd.read_csv('data.csv', delimiter=';')
df4 = pd.read_csv('data.csv', delimiter=';')

events = [
    {'normal': False, 'data': df1},
    {'normal': True, 'data': df2},
    {'normal': False, 'data': df3},
    {'normal': False, 'data': df4}
]

score = care.calc(events)
print(score)
