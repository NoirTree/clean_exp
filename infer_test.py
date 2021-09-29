import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import string

from faker import Faker
from dataprep.clean import *

# 计时装饰器
import time
from functools import wraps
def timer_with_return_value(fun):
    def wrapper(*args, **kwargs):
        start_time = time.process_time()
        res = fun(*args, **kwargs)
        end_time = time.process_time()
        print("used time: %.9f"%(end_time - start_time))
        return res
    return wrapper


# zh_CN, en_US
fake = Faker()
Faker.seed(0)

# validate_num的问题
srs = pd.Series(np.random.randint(1, 1000000, size = 300), name="random_num")
validate_date(srs).sum()

# ①产生fake data
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
        "url_schemeless": [fake.url(schemes=[]) for _ in range(n)], # validate_url 不支持
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

## BF infer: 若超过<threshold>, 就加入可行解, 按概率降序输出
# @timer_with_return_value
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

# infer_types_BF_srs(fake_df["unix_time"])

# 模块内有的types：
fake_df_in_module = generate_fake_df_in_module(300)
type_dicts_in_module = fake_df_in_module.apply(infer_types_BF_srs)
for i, type_dict in type_dicts_in_module.items(): # 查看问题
    print(i, type_dict)

# speed: 6s, 300 rows
"""
start_time = time.process_time()
type_dicts = fake_df_in_module.apply(infer_types_BF_srs)
end_time = time.process_time()
print("used time: %.9f"%(end_time - start_time))
"""


# 导入validate function所支持的标签
import dataprep.clean
validate_lst = [x for x in dir(dataprep.clean) if x.startswith("validate")]
labels = [x[9:] for x in validate_lst]
labels
col_label_map = {"bic11": "bic", "bic8": "bic", "ean13": "ean", "ean8": "ean",
                 "time": "date", "unix_time": "date", "iso8601": "date",
                 "coordinate": "lat_long", "latitude": "lat_long",
                 "longitude": "lat_long", "latlng": "lat_long",
                 "ipv4": "ip", "ipv6": "ip", "url_schemeless": "url",
                 "isbn10": "isbn", "isbn13": "isbn",
                 "ssn": "us_ssn",} # column name -> label

##############################
## 模块内有的type
fake_df_in_module = generate_fake_df_in_module(300)
type_dicts_in_module = fake_df_in_module.apply(infer_types_BF_srs)
for i, type_dict in type_dicts_in_module.items(): # 查看问题
    print(i, type_dict)

type_dicts_in_module

# 结果扩充为矩阵
df_infer_in_module = pd.DataFrame(index=fake_df_in_module.columns, columns=labels)
# type_dicts_in_module
for type, dicts in type_dicts_in_module.items():
    pretype_prob = type_dicts_in_module[type] # still a dict
    for pretype, prob in pretype_prob.items():
            df_infer_in_module.loc[type, pretype] = prob
df_infer_in_module
predicted = np.nan_to_num(df_infer_in_module.to_numpy(dtype=float), nan=0).argmax(axis=1)
predicted

# 处理label list
label_ind = []
for type in df_infer_in_module.index:
    try: # col_name与label一致的
        label_ind.append(labels.index(type))
    except: # col_name与label不一致的，经过map再加入
        mapped_label = col_label_map[type]
        label_ind.append(labels.index(mapped_label))
label_ind # 每个type的真实label在labels（即columns）的第几列


# 评价
acc = (predicted==label_ind).sum()/len(label_ind)
acc # 0.48
from sklearn.metrics import precision_score, recall_score
precision_score(label_ind, predicted, average='macro') # 0.58
precision_score(label_ind, predicted, average='micro') # 0.48
recall_score(label_ind, predicted, average='macro') # 0.538
recall_score(label_ind, predicted, average='micro') # 0.48
# macro高而micro低，也说明了咱们对罕见类别比较拿手？

######################################
## 模块内没有的types：
fake_df_outside_module = generate_fake_df_outside_module(300)
type_dicts_outside_module = fake_df_outside_module.apply(infer_types_BF_srs)
for i, type_dict in type_dicts_outside_module.items(): # 一些bugs
    print(i, type_dict)

# 结果扩充为矩阵
df_infer_outside_module = pd.DataFrame(index=fake_df_outside_module.columns, columns=labels)
# type_dicts_in_module
for type, dicts in type_dicts_outside_module.items():
    pretype_prob = type_dicts_outside_module[type] # still a dict
    for pretype, prob in pretype_prob.items():
            df_infer_outside_module.loc[type, pretype] = prob
df_infer_outside_module

# 若非NaN的数量<labels的数量，说明被错判成某种type，为FP
predicted_out = df_infer_outside_module.isna().sum(axis=1)==len(labels)
label_out = [False]*len(predicted_out)
# 评价
precision_score(label_out, predicted_out, average='macro') # 0.5
precision_score(label_out, predicted_out, average='micro') # 0.5
recall_score(label_out, predicted_out, average='macro') # 0.25
recall_score(label_out, predicted_out, average='micro') # 0.5
# 太惨了，还不如瞎猜呢



##################################
## softmax(手动)
df_infer_in_module
df_infer_in_module.applymap(np.exp).sum(axis=1)
# df_softmax=df_infer_in_module.applymap(np.exp)/df_infer_in_module.applymap(np.exp).sum(axis=1)[:, np.newaxis]
numerator=df_infer_in_module.applymap(np.exp).to_numpy()
denominator=df_infer_in_module.applymap(np.exp).sum(axis=1).to_numpy()
df_softmax_np=numerator/denominator[:, np.newaxis] # broadcast (25,)->(25,118)
df_softmax = pd.DataFrame(df_softmax_np, index=fake_df_in_module.columns, columns=labels)
df_softmax.sum(axis=1) # =0的说明什么也没预测出来

# 处理label list
label_ind # 每个type的真实label在labels（即columns）的第几列

## cross-entropy
ai = np.expand_dims(np.array(label_ind), axis=1)
np.take_along_axis(df_softmax_np, ai, axis=1)
css = np.take_along_axis(df_softmax_np, ai, axis=1)
np.nan_to_num(css, nan=1e-15) # 用1e-15代替0
bf_res = -np.divide(np.log(np.nan_to_num(css, nan=1e-15)).sum(), len(css))
bf_res # 10.01

############################################
## confidence 实现

# 试验：计算validate_date的confidence
_df_for_cal_confidence = fake_df_in_module
# 有点问题：应该是一个validate只能对应一个（或不同format的若干个），而不是这里产生的fake data
df_pred_prob = _df_for_cal_confidence.apply(validate_date).sum(axis=0)/len(_df_for_cal_confidence)
df_pred_prob.drop(labels=["date", "time", "unix_time", "iso8601"], inplace=True) # 删除本身
