import os
import json
from flask import Flask, request
from dotenv import load_dotenv
import requests
from gremlin_python.driver import client, serializer
import sys
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np

# UNIX_TIME
from datetime import datetime, timezone, timedelta
JST = timezone(timedelta(hours=+9), 'JST')

app = Flask(__name__)
load_dotenv('.env')
env = os.environ

_gremlin_order = {}
g_dict = {}


def grem_order(order_num, input_name):
    global _gremlin_order
    if order_num == 0:
        _gremlin_order = {
            "order": "g.V().has('fullName','{0}').count()".format(input_name)
        }
    if order_num == 1:
        _gremlin_order = {
            "order": "g.V().has('fullName','{0}').values('label')".format(input_name)
        }
    if order_num == 2:
        _gremlin_order = {
            "order": "g.V().hasLabel('user').has('fullName','{0}').values('type')".format(input_name)
        }
    if order_num == 3:
        _gremlin_order = {
            "order": "g.V().hasLabel('user').has('fullName','{0}').values('name')".format(input_name)
        }
    if order_num == 4:
        _gremlin_order = {
            "order": "g.V().hasLabel('group').has('fullName','{0}').in('belongsTo').properties()".format(input_name)
        }
    return


def order_to_gremlin(order_num, x):
    grem_order(order_num, x)
    print(_gremlin_order)
    for result in get_callback():
        return result


def howistheName(companyId, groupId, _input):
    global g_dict
    g_dict = {}
    if order_to_gremlin(0, _input)[0] == 0:  # change 'usr' to "fullName"
        send_message(companyId, groupId, 'BOT: I cannot find him/her.')
        return 0
    else:
        if order_to_gremlin(1, _input)[0] == 'user':
            if order_to_gremlin(2, _input)[0] == 'bot':
                send_message(companyId, groupId, 'it\'s bot... (-_-#)')
                return 1
            else:
                usr_name = order_to_gremlin(3, _input)[0]
                g_dict[usr_name] = _input
                return [usr_name]
        else:
            g_dict = prop_to_dict(order_to_gremlin(4, _input))
            #関数(order_to_gremlin(4, _input))
            usrName_list = [k for k in g_dict]
            return usrName_list
    return


def prop_to_dict(prop):
    d = {}
    for i in range(len(prop)):
        if i % 3 == 0:
            d[prop[i]['value']] = prop[i+1]['value']
    return d


def get_relation(user1, user2):
    path = 'relation/{0}.json'.format(user1)
    if not os.path.exists(path):
        return [0, 0, 0, 0]
    with open(path) as file:
        data = json.load(file)
    return [data[user2][k] for k in data[user2]]


def change_time(unixTime):
    weekday = unixTime.weekday
    time = ((unixTime.hour + 20) % 24) * 6 + int(unixTime.minute / 10)
    return [weekday, time]


def run_query_log(client, _gremlin_order):

    for key in _gremlin_order:
        print("\t{0}:".format(key))
        print("\tRunning this Gremlin query:\n\t{0}".format(
            _gremlin_order[key]))
        callback = client.submitAsync(_gremlin_order[key])
        for result in callback.result():
            print(result)
            print("\n")


def get_query(client, _gremlin_order):
    ret = []
    for key in _gremlin_order:
        callback = client.submitAsync(_gremlin_order[key])
        ret.extend(callback.result())
    return ret


def get_callback():
    global _gremlin_order
    try:
        # User definition
        _client = client.Client(env['DATABASE_SERVER_URL'], 'g',
                                username=env['USERNAME'],
                                password=env['PASSWORD'],
                                message_serializer=serializer.GraphSONSerializersV2d0()
                                )

        return get_query(_client, _gremlin_order)

    except Exception as e:
        print('There was an exception: {0}'.format(e))
        # traceback.print_exc(file=sys.stdout)
        sys.exit(1)


def MLP(x):
    init = tf.variance_scaling_initializer()
    layer_1 = tf.layers.dense(
        x, 12, activation=tf.nn.relu, kernel_initializer=init)
    layer_2 = tf.layers.dense(
        layer_1, 12, activation=tf.nn.relu, kernel_initializer=init)
    out = tf.layers.dense(layer_2, 3, kernel_initializer=init)
    return out

# relation を読み込む必要   rel = [x1, x2, x3, x4]


def model(weekday, time, name, rel):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 6])
    y = MLP(x)
    path = './models/model_{0}/'.format(name)
    if not os.path.exists(path):
        path = './models/model0/'
    if not os.path.exists(path):
        print('Model file not found.')
        return [0.0, 0.0, 0.0]
    X = np.array([[weekday, time, rel[0], rel[1], rel[2], rel[3]]])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(sess, path)
        Y = y.eval(feed_dict={x: X}, session=sess)[0]
        Y = [max(0, Y[0]), max(0, Y[1]), max(0, Y[2])]
        Y = [min(1, Y[0]), min(1, Y[1]), min(1, Y[2])]
        return Y
    #     print(1)

    # print(" ##output from model\nname:{0}\ntime:{1}\nweekday:{2}\nrelation:".format(
    #     name, time, weekday)+str(rel))
    # y = list([0.0, 0.0, 0.0])
    # return y


def get_answer(user1, user2s, post_time):
    _time = datetime.fromtimestamp(int(post_time)/1000)
    weekday = _time.weekday()
    time = ((_time.hour + 20) % 24) * 6 + int(_time.minute/10)
    d = {}
    for u in user2s:
        d[u] = model(weekday, time, u, get_relation(u, user1))
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)[:3]
    r = {i[0]: i[1] for i in d}
    return r


@app.route('/message', methods=['POST'])
def messages():
    if is_request_valid(request):
        body = request.get_json(silent=True)
        companyId = body['companyId']
        msgObj = body['message']
        userName = msgObj['createdUserName']
        groupId = msgObj['groupId']
        # recieveName = msgObj['text']
        # recieve_name = howistheName(companyId, groupId, recieveName)
        origin = {}
        send_ori = ''
        with open('maskedFullNameToOriginName.json') as file:
            origin = json.load(file)
        message = msgObj['text'].split("-")
        if len(message) > 1:
            sendName = message[1]
            send_ori = origin[sendName]
            send_name = howistheName(companyId, groupId, sendName)[0]
        else:
            send_name = howistheName(companyId, groupId, userName)[0]
        recieveName = message[0]
        recieve_name = howistheName(companyId, groupId, recieveName)

        if type(recieve_name) is list:
            # model
            # result_dict = get_answer(
            #     userName, recieve_name, msgObj['createdAt'])
            result_dict = get_answer(
                send_name, recieve_name, msgObj['createdAt'])
            a = '{0}\n'.format(send_ori)
            for key in result_dict:
                a += "{0} さんの返信\n\t10分以内： {1:.0%}\n\t2時間以内： {2:.0%}\n\t1日以内： {3:.0%}\n".format(
                    origin[g_dict[key]], result_dict[key][0], result_dict[key][1], result_dict[key][2])
            send_message(companyId, groupId, a)
        return "OK"
    else:
        return "Request is not valid."

# Check if token is valid.


def is_request_valid(request):
    validationToken = env['CHIWAWA_VALIDATION_TOKEN']
    requestToken = request.headers['X-Chiwawa-Webhook-Token']
    return validationToken == requestToken

# Send message to Chiwawa server


def send_message(companyId, groupId, message):
    url = 'https://{0}.chiwawa.one/api/public/v1/groups/{1}/messages'.format(
        companyId, groupId)
    headers = {
        'Content-Type': 'application/json',
        'X-Chiwawa-API-Token': env['CHIWAWA_API_TOKEN']
    }
    content = {
        'text': message
    }
    requests.post(url, headers=headers, data=json.dumps(content))

# 基底例外クラス


class ChiwawaBaseException(Exception):
    pass


# Chiwawaサーバーから返ってきたエラーレスポンスをハンドリングする例外クラス
class ChiwawaResposeError(ChiwawaBaseException):
    def __init__(self, status_code, err_resp):
        super().__init__()
        self.status_code = status_code
        self.err_resp = err_resp

    def __str__(self):
        return json.dumps(self.err_resp)


# 知話輪クライアントクラス
class ChiwawaClient(object):
    def __init__(self, commpany_id, api_token, api_version='v1'):

        self.api_version = api_version
        self.commpany_id = commpany_id
        self.api_token = api_token
        self.base_url = \
            'http://{0}.chiwawa.one/api/public/{1}'.format(
                self.commpany_id, self.api_version)

    @staticmethod
    def _check_status_code(status_code, text):
        if status_code != 200:
            print(status_code)
            raise ChiwawaResposeError(status_code, text)

    # メッセージ投稿
    def post_message(self, group_id, text, *,
                     to=None, from_=None, to_all=False, attachments=None):
        # The 'from' is a reserved word for python, so specify it with 'from_'.

        data = {
            'text': text
        }

        if to is not None:
            data['to'] = to

        if from_ is not None:
            data['from'] = from_

        if to_all is not False:
            data['toAll'] = str(to_all).lower()

        if attachments is not None:
            data['attachments'] = attachments

        url = '{0}/groups/{1}/messages'.format(self.base_url, group_id)
        headers = {
            'Content-Type': 'application/json',
            'X-Chiwawa-API-Token': self.api_token
        }
        resp = requests.post(url, headers=headers, data=json.dumps(data))
        self._check_status_code(resp.status_code, resp.text)

        return resp.json()

    # メッセージ一覧取得
    def get_message_list(self, group_id, *,
                         created_at_from=0, created_at_to=None,
                         max_results=20):

        params = {
            'createdAtFrom': created_at_from,
            'maxResults': max_results,
        }

        if created_at_from is not None:
            params['createdAtTo'] = created_at_to

        url = '{0}/groups/{1}/messages'.format(self.base_url, group_id)
        headers = {
            'Content-Type': 'application/json',
            'X-Chiwawa-API-Token': self.api_token
        }
        resp = requests.get(url, headers=headers, params=params)
        self._check_status_code(resp.status_code, resp.text)

        return resp.json()

    # メッセージ情報取得
    def get_message_info(self, group_id, message_id):

        url = '{0}/groups/{1}/messages/{2}'.format(self.base_url, group_id,
                                                   message_id)
        headers = {
            'Content-Type': 'application/json',
            'X-Chiwawa-API-Token': self.api_token
        }
        resp = requests.get(url, headers=headers)
        self._check_status_code(resp.status_code, resp.text)

        return resp.json()

    # メッセージ削除（できれば使わないで）
    def delete_message(self, group_id, message_id):

        url = '{0}/groups/{1}/messages/{2}'.format(self.base_url, group_id,
                                                   message_id)
        headers = {
            'Content-Type': 'application/json',
            'X-Chiwawa-API-Token': self.api_token
        }
        resp = requests.delete(url, headers=headers)
        self._check_status_code(resp.status_code, resp.text)

        return True

    # メッセージ付加情報変更
    def update_message_attachments(self, group_id, message_id, attachments):

        url = '{0}/groups/{1}/messages/{2}/attachments'.format(self.base_url,
                                                               group_id,
                                                               message_id)
        data = {'attachments': attachments}
        headers = {
            'Content-Type': 'application/json',
            'X-Chiwawa-API-Token': self.api_token
        }
        resp = requests.put(url, headers=headers, data=json.dumps(data))
        self._check_status_code(resp.status_code, resp.text)

        return resp.json()

    # ファイル情報取得
    # レスポンスに含まれるダウンロードURL(downloadUrl)は60秒間のみ有効
    def get_file_info(self, group_id, message_id):

        url = '{0}/groups/{1}/files/{2}'.format(self.base_url, group_id,
                                                message_id)
        headers = {
            'Content-Type': 'application/json',
            'X-Chiwawa-API-Token': self.api_token
        }
        resp = requests.get(url, headers=headers)
        self._check_status_code(resp.status_code, resp.text)

        return resp.json()

    # ファイル投稿
    def post_file(self, group_id, file_type, file_path, *, file_name=None):

        if file_name is None:
            file_name = os.path.basename(file_path)

        files = {'file': (file_name, open(file_path, 'rb'), file_type)}
        data = {'fileName': file_name}

        url = '{0}/groups/{1}/files'.format(self.base_url, group_id)
        headers = {
            'X-Chiwawa-API-Token': self.api_token
        }
        resp = requests.post(url, headers=headers, files=files, data=data)
        self._check_status_code(resp.status_code, resp.text)

        return resp.json()

    # グループ所属ユーザ一覧取得
    def get_group_user_list(self, group_id):

        url = '{0}/groups/{1}/users'.format(self.base_url, group_id)
        headers = {
            'Content-Type': 'application/json',
            'X-Chiwawa-API-Token': self.api_token
        }
        resp = requests.get(url, headers=headers)
        self._check_status_code(resp.status_code, resp.text)

        return resp.json()
