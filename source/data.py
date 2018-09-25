# coding: utf-8

import json as js
import os
import datetime


def get_relation_data():
    data = {}
    for i in range(1, 404):
        path = 'relation/user{0}.json'.format(i)
        if not os.path.exists(path):
            continue
        with open(path) as file:
            data['user{0}'.format(i)] = js.load(file)
    return data


def change_time(unixTime):
    _time = datetime.datetime.fromtimestamp(int(unixTime)/1000)
    weekday = _time.weekday()
    time = ((_time.hour + 20) % 24) * 6 + int(_time.minute/10)
    return [weekday, time]


reac = {}
for i in range(1, 403):
    reac['user{0}'.format(i)] = {}
    path = 'reaction/user{0}.json'.format(i)
    if not os.path.exists(path):
        continue
    with open(path) as file:
        reac['user{0}'.format(i)] = js.load(file)

rela = get_relation_data()
data = []
for key1 in reac:
    for key2 in reac[key1]:
        for n in reac[key1][key2]:
            weekday, time = change_time(n['post_time'])
            d_time = int((int(n['reaction_time']) -
                          int(n['post_time'])) / 1000 / 60 / 10)
            if key1 in rela and key2 in rela[key1]:
                mention, message, mention_all, reaction = [
                    rela[key1][key2][k] for k in rela[key1][key2]]
            else:
                continue
            data.append({'user': key1, 'weekday': weekday, 'time': time, 'mention': mention, 'message': message,
                         'mention_all': mention_all, 'reaction': reaction, 'd_time': d_time})
with open('data.json', 'w') as file:
    js.dump(data, file, indent=2)
