'''
Generate embedding.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

from faker import Faker
from dataprep.clean import *
import time

fake = Faker()
Faker.seed(0)

def infer_types_BF_srs(srs, threshold=0.6):
    import dataprep.clean
    validate_lst = [x for x in dir(dataprep.clean) if x.startswith("validate")]

    n_row = len(srs)
    type_prob_dict = {}

    for foo_name in validate_lst:
        foo = getattr(dataprep.clean, foo_name)
        prob = foo(srs).sum()/n_row
        if prob >= threshold:
            type_prob_dict[foo_name[9:]] = prob

    return dict(sorted(type_prob_dict.items(), key=lambda x:x[1], reverse=True))

# 导入validate function所支持的标签
import dataprep.clean
validate_lst = [x for x in dir(dataprep.clean) if x.startswith("validate")]
features = [x[9:] for x in validate_lst]
features

## naive embedding
# 我们有多少个validate函数，这个vector就有多少维。
len(features) # 108维

# 生成"training data"
type_foo_dict = {"address": "address",
    "iban": "iban",
    "bic": "swift",
    "bic11": "swift11",
    "bic8": "swift8",
    "ean13": "ean13",
    "ean8": "ean8",
    "date": "date",
    "time": "time",
    "unix_time": "unix_time",
    "iso8601": "iso8601",
    "coordinate": "coordinate",
    "latitude": "latitude",
    "longitude": "longitude",
    "latlng": "latlng",
    "email": "email",
    "ipv4": "ipv4",
    "ipv6": "ipv6",
    "url": "url",
    # "url_schemeless": "url", # 若要传参，判断逻辑需要额外时间。暂时不该
    "isbn10": "isbn10",
    "isbn13":"isbn13",
    "ssn":"ssn",
    "phone":"phone_number",
    "currency":"pricetag",
    "license_plate":"license_plate",
    "aba":"aba",
    "bban":"bban",
    "color":"color",
    "rgb_color":"rgb_color",
    "credit_card_number":"credit_card_number",
    "iana_id":"iana_id",
    "mac_address":"mac_address",
    "port_number":"port_number",
    "ripe_id":"ripe_id",
}
len(type_foo_dict) # 34; 有一些是相同type的不同format，例如bic11和bic8

# 真实的classes数量只有22个
type_label_dict = {
    "address": "address",
    "iban": "iban",
    "bic": "swift",
    "bic11": "swift",
    "bic8": "swift",
    "ean13": "ean",
    "ean8": "ean",
    "date": "date",
    "time": "date",
    "unix_time": "date",
    "iso8601": "date",
    "coordinate": "lat_long",
    "latitude": "lat_long",
    "longitude": "lat_long",
    "latlng": "lat_long",
    "email": "email",
    "ipv4": "ip",
    "ipv6": "ip",
    "url": "url",
    "isbn10": "isbn",
    "isbn13":"isbn",
    "ssn":"ssn",
    "phone":"phone",
    "currency":"currency",
    "license_plate":"license_plate",
    "aba":"aba",
    "bban":"bban",
    "color":"color",
    "rgb_color":"color",
    "credit_card_number":"credit_card_number",
    "iana_id":"iana_id",
    "mac_address":"mac_address",
    "port_number":"port_number",
    "ripe_id":"ripe_id",
}
len(set(type_label_dict.values())) # 22


def generate_fake_data(type, type_foo_dict, n_row=300, n_col=20):
    _training_df = pd.DataFrame()
    foo = getattr(fake, type_foo_dict[type])
    for i in range(n_col):
        _training_df[type+"_"+str(i)] = [foo() for i in range(n_row)]
    return _training_df

# 结果扩充为矩阵
def turn_infered_results_to_matrix(infer_type_dicts_srs, features):
    '''
    infer_type_dicts_srs: a series, each word is the result of infer_types_BF_srs
    features: supported features in clean_module
    '''
    _df_mm = pd.DataFrame(index=infer_type_dicts_srs.index, columns=features)
    for type, dicts in infer_type_dicts_srs.items():
        for pretype, prob in dicts.items():
                _df_mm.loc[type, pretype] = prob
    return np.nan_to_num(_df_mm.to_numpy(dtype=float), nan=0)

## 测试
training_df = generate_fake_data("phone", type_foo_dict)
infer_type_dicts_srs = training_df.apply(infer_types_BF_srs)
# training_df.iloc[:, 0].tolist()
training_df_mm = turn_infered_results_to_matrix(infer_type_dicts_srs, features)
training_df_mm
# training_df_mm.shape # (20, 108) --> 20个instances
# training_df_mm.sum(axis=1)
# embedding = training_df_mm.mean(axis=0).tolist() # 综合成一个


# 对所有type求embedding
def cal_embedding(type_foo_dict, features, n_row=300, n_col=10):
    type_embedding = pd.DataFrame(columns=features)
    for x in type_foo_dict.keys():
        training_df = generate_fake_data(x, type_foo_dict, n_row, n_col)
        infer_type_dicts_srs = training_df.apply(infer_types_BF_srs)
        training_df_mm = turn_infered_results_to_matrix(infer_type_dicts_srs, features)
        embedding = training_df_mm.mean(axis=0).tolist()
        # print(embedding) # debug
        type_embedding.loc[x] = embedding
    return type_embedding # 34*108

# type_embedding = cal_embedding(type_foo_dict, features, 300, 20)
# type_embedding

# 这是每种type 10列，每列300行的结果. 经过测试，不会随着列数增加而变好了
# type_embedding.to_csv("temp_embedding.csv", index = False)
type_embedding = pd.read_csv("temp_embedding.csv")

type_embedding.to_numpy().sum(axis=1) # 有值

# 作图
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(type_embedding)
X_pca
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.show()

##################################
# 利用embedding测试
def generate_fake_df_in_module(n=100):
    """
    generate fake data for types that are included in current clean module.
    :param n: row number
    :return: fake_df
    """
    fake_df = pd.DataFrame({
        "address": [fake.address() for _ in range(n)],
        "iban": [fake.iban() for _ in range(n)],
        "bic": [fake.swift() for _ in range(n)],  # mix formats
        "bic11": [fake.swift11() for _ in range(n)],
        "bic8": [fake.swift8() for _ in range(n)],
        "ean13": [fake.ean13() for _ in range(n)],
        "ean8": [fake.ean8() for _ in range(n)],
        "date": [fake.date() for _ in range(n)],
        "time": [fake.time() for _ in range(n)],  # validate_date
        "unix_time": [fake.unix_time() for _ in range(n)],  # validate_date
        "iso8601": [fake.iso8601() for _ in range(n)],  # validate_date不支持的time
        "coordinate": [fake.coordinate() for _ in range(n)], # validate_lat_long
        "latitude": [fake.latitude() for _ in range(n)], # validate_lat_long
        "longitude": [fake.longitude() for _ in range(n)], # validate_lat_long
        "latlng": [fake.latlng() for _ in range(n)], # validate_lat_long
        "email": [fake.email() for _ in range(n)],
        "ipv4": [fake.ipv4() for _ in range(n)],
        "ipv6": [fake.ipv6() for _ in range(n)],
        "url": [fake.url() for _ in range(n)],
        # "url_schemeless": [fake.url(schemes=[]) for _ in range(n)], # validate_url 不支持
        "isbn10": [fake.isbn10() for _ in range(n)],
        "isbn13": [fake.isbn13() for _ in range(n)],
        "ssn": [fake.ssn() for _ in range(n)],
        "phone": [fake.phone_number() for _ in range(n)],
        "currency": [fake.pricetag() for _ in range(n)],  # 美元，如$7,604.87
    }
    )

    return fake_df

def generate_fake_df_outside_module(n=100):
    """
    generate fake data for types that are not included in current clean module.
    :param n:
    :return:
    """
    fake_df = pd.DataFrame({
        "license_plate": [fake.license_plate() for _ in range(n)], # automotive
        "aba": [fake.aba() for _ in range(n)], # ABA routing transit number. 不含在stdnum
        "bban": [fake.bban() for _ in range(n)], # Basic Bank Account Number. 不含在stdnum
        "color": [fake.color() for _ in range(n)], # hex
        "rgb_color": [fake.rgb_color() for _ in range(n)],
        "credit_card_number": [fake.credit_card_number() for _ in range(n)], # 混合型card_type
            # 包含formats:'amex', 'diners', 'discover', 'jcb', 'jcb15', 'jcb16', 'maestro', 'mastercard',
            # 'visa', 'visa13', 'visa16', and 'visa19'
        "iana_id": [fake.iana_id() for _ in range(n)], #IANA Registrar ID, 如'6463344'
        "mac_address": [fake.mac_address() for _ in range(n)],
        "port_number": [fake.port_number() for _ in range(n)],
        "ripe_id": [fake.ripe_id() for _ in range(n)],
    }
    )

    return fake_df

def generate_fake_df_complete(n=100):
    return generate_fake_df_in_module(n).join(generate_fake_df_outside_module(n))

fake_df = generate_fake_df_complete(300)
fake_type_dicts = fake_df.apply(infer_types_BF_srs)

_df_mm = pd.DataFrame(index=fake_type_dicts.index, columns=features)
for type, dicts in fake_type_dicts.items():
    for pretype, prob in dicts.items():
        _df_mm.loc[type, pretype] = prob
infered_mm = np.nan_to_num(_df_mm.to_numpy(dtype=float), nan=0)

turn_infered_results_to_matrix(fake_type_dicts, features) # 为啥不对？？

# 计算相似性
from sklearn.metrics.pairwise import cosine_similarity
sim_mm = cosine_similarity(infered_mm, type_embedding)
sim_mm.argmax(axis=1)

# 计算acc, pre, recall
n = sim_mm.shape[0]
label_ind = np.arange(n) # 特殊，正好都是
predicted = sim_mm.argmax(axis=1)

acc = (predicted==label_ind).sum()/n
acc # 0.558
from sklearn.metrics import precision_score, recall_score
precision_score(label_ind, predicted, average='macro') # 0.449
precision_score(label_ind, predicted, average='micro') # 0.558
recall_score(label_ind, predicted, average='macro') # 0.558
recall_score(label_ind, predicted, average='micro') # 0.558


##################################
## 尝试更换validate_time
from validator_collection import validators, checkers, errors

training_df = generate_fake_data("date", type_foo_dict) # is_date
training_df = generate_fake_data("time", type_foo_dict) # is_timedrelta
training_df = generate_fake_data("unix_time", type_foo_dict) # is_timezone, is_timedelta
training_df = generate_fake_data("iso8601", type_foo_dict) # is_timezone, is_datetime
# date, time, unix_time, iso8601
training_df.applymap(checkers.is_timedelta).sum()
training_df

# 随机integer？
df = pd.DataFrame(np.arange(300).reshape((50, 6)))
df.applymap(checkers.is_timedelta).sum()
df.shape # 只要支持unix_time，这个问题就无法避免
##################################
# 训练embedding
x = torch.rand(2, 2)

##################################
## 待处理问题
# argmax有多个结果?