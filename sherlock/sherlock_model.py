import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from faker import Faker

from sherlock import helpers
from sherlock.features.preprocessing import extract_features, convert_string_lists_to_lists, prepare_feature_extraction

fake = Faker()
Faker.seed(0)

# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

## generate data for Sherlock
# 随机生成 3710*n_col
# def generate_sherlock_df(type_foo_dict, n_col=30, mean_n_row=3710):
#     """
#     generate data conformed to restrictions in Sherlock.
#     :param type_foo_dict: dictionary, key:value = type name:function name
#     :param mean_n_row: mean of the number of row
#     :param n_col: number of column
#     :return: df_value, df_label
#     """
#     _values = pd.DataFrame(columns=["values"], index=range(n_col), dtype=object)
#     _labels = pd.DataFrame(columns=["type"])
#     for i in range(n_col):
#         n_row = round(np.random.normal(3710))
#         foo_name = np.random.choice(list(type_foo_dict.keys()))
#         foo = getattr(fake, type_foo_dict[foo_name])
#         _values.iat[i, 0] = str([str(foo()) for i in range(n_row)])
#         _labels.loc[i] = foo_name
#
#     return _values, _labels
# # # preprocess
# num_train = 10
# train_df, train_labels = generate_sherlock_df(type_foo_dict, num_train) # 样本数
# train_df
# train_labels
# train_samples_converted, y_train = convert_string_lists_to_lists(train_df, train_labels, "values", "type")


# 把正常的dataset转成sherlock需要的输入
def transform_into_sherlock_df(fake_df):
    sherlock_df = pd.DataFrame(columns=["values"], index=range(fake_df.shape[1]), dtype=object)
    sherlock_labels = pd.DataFrame(columns=["type"])
    for i in range(fake_df.shape[1]):
        col = fake_df.columns[i]
        type_name = col[:col.rfind("_")]  # type名字
        sherlock_df.iat[i, 0] = list(map(str, fake_df.iloc[:, i].to_list()))
        sherlock_labels.loc[i] = type_name
    return sherlock_df, sherlock_labels

# extract feature
import os
os.getcwd()
os.chdir(os.path.join(os.getcwd(), "sherlock"))

def extract_feature(fake_df_path, out_path): # in_path: fake_df位置, out_path: feature位置
    _fake_df = pd.read_csv(fake_df_path)
    _df, _labels = transform_into_sherlock_df(_fake_df)
    _df_converted, _y = _df["values"], _labels["type"].to_list()
    _X = extract_features(_df_converted)
    _X.to_csv(out_path, index=False)
    return _X, _labels

def get_dataset(fake_df_path, feature_path, device="cpu", needed_extract=False, needed_X=False):
    if needed_extract:
        _X, _labels = extract_feature(fake_df_path, feature_path)
    else:
        _fake_df = pd.read_csv(fake_df_path)
        _, _labels = transform_into_sherlock_df(_fake_df)
        _X = pd.read_csv(feature_path)

    # values
    _X = _X.apply(lambda x: pd.Series.astype(x, dtype=float))
    _columns_means = pd.DataFrame(_X.mean()).transpose()  # 用于impute NaN
    _X.fillna(_columns_means.iloc[0], inplace=True)
    _X_ts = torch.tensor(_X.values, device=device)
    # labels
    le = LabelEncoder()
    _X_labels = le.fit_transform(_labels.iloc[:, 0].to_list())
    _X_labels = torch.tensor(_X_labels, device=device).view(-1, 1)
    _dataset = torch.utils.data.TensorDataset(_X_ts, _X_labels)

    if needed_X:
        return _dataset, _X
    else:
        return _dataset

## training data
dataset, X_train = get_dataset(fake_df_path="../fake_df_30_new.csv",
                            feature_path="../sherlock_features_train_new.csv",
                            needed_X=True)

test_set = get_dataset(fake_df_path="../fake_df_10_new.csv",
                       feature_path="../sherlock_features_test_new.csv")


# 拆分出validate
train_set_size = int(len(X_train)*0.8)
valid_set_size = len(X_train)-train_set_size
train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])


## model construction
class FlattenLayer(nn.Module): # from book
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x.shape = (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class SubNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(SubNet, self).__init__()
        self.num_inputs, self.num_outputs, self.num_hiddens = \
            num_inputs, num_outputs, num_hiddens
        self.net = nn.Sequential(OrderedDict([
            ('Flatten', FlattenLayer()),
            ('Linear1', nn.Linear(self.num_inputs, self.num_hiddens)),
            ('BatchNorm', nn.BatchNorm1d(self.num_hiddens)),
            ('ReLU1', nn.ReLU()),
            ('Dropout', nn.Dropout(0.3)),
            ('Linear2', nn.Linear(self.num_hiddens, self.num_hiddens)),
            ('ReLU2', nn.ReLU()),
            ('Linear3', nn.Linear(self.num_hiddens, self.num_outputs)),
        ]))

    def forward(self, x):
        return self.net(x)

class Sherlock(nn.Module):
    '''
    接收num_outputs(分类数目)作为输入
    '''
    def __init__(self, num_outputs):
        super(Sherlock, self).__init__()
        self.num_inputs, self.num_outputs, self.num_hiddens = num_outputs*3+27, num_outputs, 500
            # num_inputs是3个子网输出+特殊的27个特征
        self.CharNet = SubNet(960, self.num_outputs, 300) # num_inputs, num_outputs, num_hiddens
        self.WordNet = SubNet(201, self.num_outputs, 200)
        self.ParaNet = SubNet(400, self.num_outputs, 400)
        self.MajorNet = nn.Sequential(OrderedDict([
            # ('Flatten', FlattenLayer()),
            ('Linear1', nn.Linear(self.num_inputs, self.num_hiddens)),
            ('BatchNorm', nn.BatchNorm1d(self.num_hiddens)),
            ('ReLU1', nn.ReLU()),
            ('Dropout', nn.Dropout(0.3)),
            ('Linear2', nn.Linear(self.num_hiddens, self.num_hiddens)),
            ('ReLU2', nn.ReLU()),
            ('Linear3', nn.Linear(self.num_hiddens, self.num_outputs)),
        ]))

    def forward(self, x):
        # x是已经preprocessed过的tensor, x.shape[1] = 1588 = 960+201+400+27
        # print(x)
        # print(x.shape)
        assert x.shape[1] == 1588
        x_char = self.CharNet(x[:, :960])
        x_word = self.WordNet(x[:, 960:1161]) # +201
        x_para = self.ParaNet(x[:, -400:]) # 是从后往前存的
        x_major = torch.cat((x_char, x_word, x_para, x[:, 1161:-400].float()), dim=1)
        # print(x_major.shape[1])
        assert x_major.shape[1] == self.num_inputs

        return self.MajorNet(x_major)

def validation(net, valid_iter, loss, device="cpu"):
    '''一次完整遍历。注意该函数不会把net搬到device上，只接受本来就在device上的net'''
    net.eval()
    l_sum = 0
    acc_sum = 0
    n = 0

    with torch.no_grad(): # 不要训练模型！！
        for X, y in valid_iter:
            X = X.to(device)
            y = y.to(device)

            y_hat = net(X.float())
            l_sum += loss(y_hat, y.view(-1, 1).squeeze()).cpu().item()
            acc_sum += (y_hat.argmax(dim=1).view(-1, 1) == y).sum().cpu().item()
            n += y.shape[0]

    return l_sum/len(valid_iter), acc_sum/n

def train(net, train_iter, valid_iter, epochs, loss, optimizer,
          patience=5, # early-stopping
          device="cpu",
          plot=False,
          needed_acc_lst=False):

    trigger_times = 0 # how many times the loss is decreasing?
    last_loss = 1000

    if plot:
        train_acc_lst, valid_acc_lst = [], []

    for epoch in range(1, epochs+1):
        net.train()
        acc_sum = 0
        n = 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)

            # print(X.shape) # 测试
            # print(X) # 测试
            y_hat = net(X.float())
            l = loss(y_hat, y.view(-1, 1).squeeze())
            # grad-back-step
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            acc_sum += (y_hat.argmax(dim=1).view(-1, 1) == y).sum().cpu().item()
            n += y.shape[0]

        # print("one epoch ends.")
        current_loss, valid_acc = validation(net, valid_iter, loss, device)
        if plot:
            train_acc_lst.append(acc_sum/n)
            valid_acc_lst.append(valid_acc)
        print(f"current validation loss: {current_loss}, "
              f"validation acc {valid_acc}, train acc{acc_sum/n}")
        if current_loss>last_loss:
            trigger_times+=1
            print(f"trigger_times: {trigger_times}")
            if trigger_times>patience:
                print("Early stopping!!")
                return net
        else:
            trigger_times=0
            print("trigger_times: 0")

        last_loss=current_loss

    if plot:
        plt.plot(range(1, epochs + 1), valid_acc_lst, linestyle=':', label="valid_acc")
        plt.plot(range(1, epochs + 1), train_acc_lst, linestyle='-', label="train_acc")
        plt.legend()
        plt.show()
    if needed_acc_lst:
        return net, train_acc_lst, valid_acc_lst
    else:
        return net

def test(net, test_iter, device="cpu"):
    net.eval()
    acc_sum = 0
    n = 0
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)

            y_hat = net(X.float())
            l = loss(y_hat, y.view(-1, 1).squeeze())

            n += y.shape[0]
            acc_sum += (y_hat.argmax(dim=1).view(-1, 1)==y).sum().cpu().item()

    print(f"Test Accuracy: {acc_sum/n}")
    return acc_sum/n

### main
print('Device state:', device)

num_epochs, weight_decay, batch_size, lr = 100, 1e-4, 128, 1e-4
n_labels = 22 # 现有分类数
loss = torch.nn.CrossEntropyLoss() # softmax + crossentropy
net = Sherlock(n_labels).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

# initialize
for param in net.parameters(): # 参见Sherlock的配置json。其实pytorch会自动初始化的
    nn.init.normal_(param, mean=0, std=0.01)
    nn.init.constant_(param, val=0)

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False) # test不用
valid_iter = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)

# Train
sherlock_model, sherlock_train_acc_lst, sherlock_valid_acc_lst = train(net, train_iter,
                                            valid_iter, num_epochs,
                                            loss, optimizer,
                                            patience=5, device=device,
                                            plot=True, needed_acc_lst=True)
sherlock_test_acc = test(net, test_iter, device=device)








##################以前写的training，也可以用#################
# training
num_epochs, weight_decay, batch_size = 100, 1e-4, 128
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
train_l_sum, train_acc_sum, n, batch_count = 0, 0, 0, 0

for epoch in range(num_epochs):
    net.train() # for dropout and batch normalization
    for X, y in train_iter: # X.shape = [batch_size, feature]
        y_hat = net(X.float()) # 注意Adam一定要X.float()
        l = loss(y_hat, y.view(-1, 1).squeeze()) # 这里必须用squeeze是大坑
        # grad-back-step
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1).view(-1, 1) == y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1

    net.eval() # for evaluation
    # test_acc = evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f, train acc %.3f'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n))
    # print('test acc test_acc %.3f}' % test_acc)


################## 以前写的数据预处理 ##################
X_train = pd.read_csv("../sherlock_features_train_new.csv")
X_train.head()
# values
X_train= X_train.apply(lambda x: pd.Series.astype(x, dtype=float))
train_columns_means = pd.DataFrame(X_train.mean()).transpose() # 用于impute NaN
X_train.fillna(train_columns_means.iloc[0], inplace=True)
X_train_ts = torch.tensor(X_train.values, device=device)
# labels
le = LabelEncoder()
X_labels = le.fit_transform(train_labels.iloc[:, 0].to_list())
X_train_labels = torch.tensor(X_labels, device=device).view(-1, 1)
dataset = torch.utils.data.TensorDataset(X_train_ts, X_train_labels)

## test data
test_fake_df = pd.read_csv("../fake_df_10_new.csv")
test_df, test_labels = transform_into_sherlock_df(test_fake_df)
test_samples_converted, y_test = test_df["values"], test_labels["type"].to_list()
# X_test = extract_features(test_samples_converted)
# X_test.to_csv("../sherlock_features_test_new.csv", index=False)
X_test = pd.read_csv("../sherlock_features_test_new.csv")
X_test.head()
# values
X_test= X_test.apply(lambda x: pd.Series.astype(x, dtype=float))
X_test.fillna(train_columns_means.iloc[0], inplace=True) # impute NaN
X_test_ts = torch.tensor(X_test.values, device=device)
# labels
le = LabelEncoder()
X_labels = le.fit_transform(test_labels.iloc[:, 0].to_list())
X_test_labels = torch.tensor(X_labels, device=device).view(-1, 1)
test_set = torch.utils.data.TensorDataset(X_test_ts, X_test_labels)

###############以下为debug。若训练模型只要运行上述代码即可##################
## 拆开debug
CharNet = SubNet(960, 34, 300)
x_char = CharNet(X[:, :960].float())
x_char

WordNet = SubNet(201, 34, 200)
x_word = WordNet(X[:, 960:1161].float())
x_word

ParaNet = SubNet(400, 34, 400)

num_inputs, num_outputs, num_hiddens = 129, 34, 500
MajorNet = nn.Sequential(OrderedDict([
            ('Flatten', FlattenLayer()),
            ('Linear1', nn.Linear(num_inputs, num_hiddens)),
            ('BatchNorm', nn.BatchNorm1d(num_hiddens)),
            ('ReLU1', nn.ReLU()),
            ('Dropout', nn.Dropout(0.3)),
            ('Linear2', nn.Linear(num_hiddens, num_hiddens)),
            ('ReLU2', nn.ReLU()),
            ('Linear3', nn.Linear(num_hiddens, num_outputs)),
        ]))
###############(archive)构建子网##################
# 1.1 character subnet
net = SubNet(960, 34, 300) #
# net(X.float())

num_inputs, num_outputs, num_hiddens = 960, 34, 300

# deal with values
X_train_char = X_train.iloc[:, :960].apply(lambda x: pd.Series.astype(x, dtype=float))
X_train_char_ts = torch.tensor(X_train_char.values)
# deal with labels
le = LabelEncoder()
X_labels_char = le.fit_transform(train_labels.iloc[:, 0].to_list())
X_train_char_labels = torch.LongTensor(X_labels_char,).view(-1, 1)

subnet_char = nn.Sequential(OrderedDict([ # forward被自动调用
            ('flatten', FlattenLayer()),
            ('linear1', nn.Linear(num_inputs, num_hiddens)),
            ('BatchNorm', nn.BatchNorm1d(num_hiddens)),
            ('ReLU1', nn.ReLU()),
            ('Dropout', nn.Dropout(0.3)),
            ('linear2', nn.Linear(num_hiddens, num_hiddens)),
            ('ReLU2', nn.ReLU()),
            ('linear3', nn.Linear(num_hiddens, num_outputs)),
        ]))

# 1.1.2 loss function and optimize
loss = torch.nn.CrossEntropyLoss() # softmax + crossentropy
optimizer = torch.optim.Adam(subnet_char.parameters(), lr=1e-4, weight_decay=1e-4)

# 1.1.3 initialize
for param in subnet_char.parameters(): # 参见Sherlock的配置json。其实pytorch会自动初始化的
    nn.init.normal_(param, mean=0, std=0.01)
    nn.init.constant_(param, val=0)


# 1.1.4 train
num_epochs, weight_decay, batch_size = 100, 1e-4, 128

dataset = torch.utils.data.TensorDataset(X_train_char_ts, X_train_char_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

train_l_sum, train_acc_sum, n, batch_count = 0, 0, 0, 0

for epoch in range(num_epochs):
    for X, y in train_iter: # X.shape = [batch_size, feature]
        y_hat = subnet_char(X.float()) # 注意Adam一定要X.float()
        l = loss(y_hat, y.view(-1, 1).squeeze()) # 这里必须用squeeze是大坑
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1
    # test_acc = evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f, train acc %.3f'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n))
    # print('test acc test_acc %.3f}' % test_acc)


# def train(net, train_features, train_labels, test_features, test_labels,
          # num_epochs, learning_rate, weight_decay, batch_size):


############在sherlock原始数据上测试模型############
class Sherlock_ori(nn.Module): # 输出维度与我们不同
    def __init__(self, **kwargs):
        super(Sherlock_ori, self).__init__(**kwargs)
        self.num_inputs, self.num_outputs, self.num_hiddens = 261, 78, 500
        self.CharNet = SubNet(960, 78, 300) # num_inputs, num_outputs, num_hiddens
        self.WordNet = SubNet(201, 78, 200)
        self.ParaNet = SubNet(400, 78, 400)
        self.MajorNet = nn.Sequential(OrderedDict([
            # ('Flatten', FlattenLayer()),
            ('Linear1', nn.Linear(self.num_inputs, self.num_hiddens)),
            ('BatchNorm', nn.BatchNorm1d(self.num_hiddens)),
            ('ReLU1', nn.ReLU()),
            ('Dropout', nn.Dropout(0.3)),
            ('Linear2', nn.Linear(self.num_hiddens, self.num_hiddens)),
            ('ReLU2', nn.ReLU()),
            ('Linear3', nn.Linear(self.num_hiddens, self.num_outputs)),
        ]))

    def forward(self, x):
        # x是已经preprocessed过的tensor, x.shape[1] = 1588 = 960+201+400+27
        # print(x.shape[1])
        assert x.shape[1] == 1588
        x_char = self.CharNet(x[:, :960])
        x_word = self.WordNet(x[:, 960:1161]) # +201
        x_para = self.ParaNet(x[:, -400:]) # 是从后往前存的
        x_major = torch.cat((x_char, x_word, x_para, x[:, 1161:-400].float()), dim=1)
        # print(x_major.shape[1])
        assert x_major.shape[1] == self.num_inputs

        return self.MajorNet(x_major)

net_ori = Sherlock_ori()
net_ori.cuda()
X_train_preprocessed = pd.read_parquet(r"E:\sherlock-project\data\data\processed\X_train.parquet")
y_train_preprocessed = pd.read_parquet(r"E:\sherlock-project\data\data\processed\y_train.parquet").reset_index(drop=True)

