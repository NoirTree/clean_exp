'''
Train embedding.
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
n_features = len(features) # 108维

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
} # 类别_对应的函数名
len(type_foo_dict) # 34; 有一些是相同type的不同format，例如bic11和bic8

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
} # 类别_真实标签名
n_labels = len(set(type_label_dict.values())) # 22

# 产生fake data
def generate_fake_data(type, type_foo_dict, n_row=300, n_col=20):
    _training_df = pd.DataFrame()
    foo = getattr(fake, type_foo_dict[type])
    for i in range(n_col):
        _training_df[type+"_"+str(i)] = [foo() for i in range(n_row)]
    return _training_df

# generate_fake_data("iban", type_foo_dict) # 测试

# 结果扩充为矩阵
def turn_infered_results_to_df(infer_type_dicts_srs, features):
    '''
    infer_type_dicts_srs: a series, each word is the result of infer_types_BF_srs
    features: supported features in clean_module
    '''
    _df_mm = pd.DataFrame(index=infer_type_dicts_srs.index, columns=features)
    for type, dicts in infer_type_dicts_srs.items():
        for pretype, prob in dicts.items():
                _df_mm.loc[type, pretype] = prob
    return _df_mm.fillna(0)
    # return np.nan_to_num(_df_mm.to_numpy(dtype=float), nan=0)


# 生成训练数据
def generate_training_data(type_foo_dict, type_label_dict,
                           features, n=20, needed_original_data=False):
    '''
    n: how many instances of each type are generated
    needed_original_data: whether return the original fake dataset or not.
        When set as "False", the function only return the inferred result of `infer_types_BF_srs`
        Note: the dataset could be quite big.

    :returns:
        X: df. each row corresponds to an embedding of a column
        y: df. one-hot label
    '''
    X = pd.DataFrame(columns=features)
    y = pd.Series(dtype=object)
    if needed_original_data:
        fake_df =pd.DataFrame()

    for type in type_foo_dict:
        _fake_df = generate_fake_data(type, type_foo_dict, n_row=300, n_col=n)
        _infer_type_dicts_srs = _fake_df.apply(infer_types_BF_srs)
        _training_df = turn_infered_results_to_df(_infer_type_dicts_srs, features)
        X = pd.concat((X, _training_df))
        true_label = type_label_dict[type]
        y = pd.concat((y, pd.Series([true_label]*n))) # 真实的标签
        if needed_original_data:
            fake_df = pd.concat((fake_df, _fake_df), axis=1)

    if needed_original_data:
        return X.reset_index(drop=True), y.reset_index(drop=True), fake_df
    else:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    # X, y = generate_training_data(type_foo_dict, type_label_dict,
#                            features, n=3)
# y_onehot = pd.get_dummies(y)

# 构建网络
class NaiveNet(nn.Module):
    def __init__(self, n_features, n_labels, n_hiddens):
        super(NaiveNet, self).__init__()
        self.num_inputs, self.num_outputs, self.num_hiddens = n_features, n_labels, n_hiddens
        self.net = nn.Sequential(
                nn.Linear(self.num_inputs, self.num_hiddens),
                nn.BatchNorm1d(self.num_hiddens),
                nn.ReLU(),
                nn.Linear(self.num_hiddens, self.num_hiddens),
                nn.BatchNorm1d(self.num_hiddens),
                nn.ReLU(),
                nn.Linear(self.num_hiddens, self.num_outputs),
        )

    def forward(self, x):
        assert x.shape[1]==self.num_inputs
        return self.net(x)

# 测试
# naivenet = NaiveNet(n_features, n_labels, n_features) # n_hiddens = n_features
# X_mm = X.to_numpy(dtype=float)
# naivenet(torch.from_numpy(X_mm).float()) # 必须有float


# # training data
# X_train, y_train, fake_df_train = generate_training_data(type_foo_dict, type_label_dict,
#                            features, n=30, needed_original_data=True) # n=每种type实例数
#
# X_train.to_csv("X_30.csv", index = False)
# y_train.to_csv("y_30.csv", index = False)
# fake_df_train.to_csv("fake_df_30.csv", index = False)
## 直接读取这个就行。n=30
X_train = pd.read_csv("X_30.csv")
y_train = pd.read_csv("y_30.csv")

train_values = torch.from_numpy(X_train.to_numpy(dtype=float)).float()
le = LabelEncoder()
train_labels = le.fit_transform(y_train)
train_labels = torch.LongTensor(train_labels).view(-1, 1)


# # test data
X_test, y_test, fake_df_test = generate_training_data(type_foo_dict, type_label_dict,
                           features, n=10, needed_original_data=True) # 每种type有10列
#
# X_test.to_csv("X_10.csv", index = False)
# y_test.to_csv("y_10.csv", index = False)
# fake_df_test.to_csv("fake_df_10.csv", index = False)

# ## 直接读取这个就行。n=10
X_test = pd.read_csv("X_10.csv")
y_test = pd.read_csv("y_10.csv")

test_values = torch.from_numpy(X_test.to_numpy(dtype=float)).float()
le = LabelEncoder()
test_labels = le.fit_transform(y_test)
test_labels = torch.LongTensor(test_labels).view(-1, 1)


## NetWork
naivenet = NaiveNet(n_features, n_labels, n_features) # n_hiddens = n_features
# loss & optimize
loss = torch.nn.CrossEntropyLoss() # softmax + crossentropy
optimizer = torch.optim.Adam(naivenet.parameters(), lr=1e-4, weight_decay=1e-4)

# evaluate
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1).view(-1, 1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# training
num_epochs, weight_decay, batch_size = 100, 1e-4, 128
train_dataset = torch.utils.data.TensorDataset(train_values, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_values, test_labels)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

train_l_sum, train_acc_sum, n, batch_count = 0, 0, 0, 0
train_acc_lst, test_acc_lst = [], []
train_loss_lst, test_loss_lst = [], []


for epoch in range(num_epochs):
    naivenet.train() # for dropout and batch normalization
    train_l_sum, train_acc_sum, n, batch_count = 0, 0, 0, 0
    for X, y in train_iter: # X.shape = [batch_size, feature]
        y_hat = naivenet(X.float())
        l = loss(y_hat, y.view(-1, 1).squeeze()) # 这里必须用squeeze是大坑
        # grad-back-step
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item() # l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1).view(-1, 1) == y).sum().item() # .sum().cpu().item()
        n += y.shape[0] # 按instance数
        batch_count += 1 # 按batch数

    naivenet.eval() # for evaluation
    test_acc = evaluate_accuracy(test_iter, naivenet)
    print('epoch %d, loss %.4f, train acc %.3f'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n))
    print('test acc test_acc %.3f}' % test_acc)
    train_acc_lst.append(train_acc_sum / n)
    test_acc_lst.append(test_acc)
    train_loss_lst.append(loss(naivenet(train_values), train_labels.view(-1, 1).squeeze()).item())
    test_loss_lst.append(loss(naivenet(test_values), test_labels.view(-1, 1).squeeze()).item())

# 作图
# acc
plt.plot(range(1, num_epochs+1), test_acc_lst, linestyle=':', label="test")
plt.plot(range(1, num_epochs+1), train_acc_lst, linestyle='-', label="train")
plt.legend()
plt.show()

# loss
plt.plot(range(1, num_epochs+1), test_loss_lst, linestyle=':', label="test")
plt.plot(range(1, num_epochs+1), train_loss_lst, linestyle='-', label="train")
plt.legend()
plt.show()


