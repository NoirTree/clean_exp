'''
generate benchmark for standardization.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataprep.clean import *
import phonenumbers
import os

## 第一层label
def get_phone_label(srs, country_code="US"):
     '''接受一个series，返回 [phone, label]组成的DataFrame'''
     df = pd.DataFrame(columns=["phone", "label"])
     cnt=0
     for x in srs:
          try:
               if phonenumbers.is_valid_number(phonenumbers.parse(x, country_code)):
                    df.loc[cnt] = [x, True]
                    cnt+=1
               else:
                    df.loc[cnt] = [x, False]
                    cnt+=1
          # except phonenumbers.phonenumberutil.NumberParseException: # 文本中的phone
          #      for match in phonenumbers.PhoneNumberMatcher(x, country_code):
          #           df.loc[cnt] = (
          #                [phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164),
          #                 True]
          #           )
          #           cnt+=1

          # 文本中的电话强制转化为e164格式的话，会影响整体格式占比
          except:# phonenumbers.phonenumberutil.NumberParseException:
               continue
     return df


def layer1_label(path):
     '''产生validate的标签'''
     df = clean_headers(pd.read_csv(path), report=False)
     phone_cols = [x for x in df.columns.values.tolist() if x.find("phone") != -1]
     phone_label_df = pd.DataFrame(columns=["phone", "label"])
     for phone_col in phone_cols:
          phone_srs = df[phone_col]
          phone_srs.dropna(inplace=True)
          # phone_label1_srs = phone_srs.apply(lambda x: is_phone_value(x, "US"))
          phone_label_df = get_phone_label(phone_srs)
     return phone_label_df

# 注: 该文件夹未上传至github.
layer1_label("./phone_benchmark/DCAS_Managed_Public_Buildings.csv")

file_lst = [x for x in os.listdir("./phone_benchmark") if x.endswith(".csv")]
res_df = pd.DataFrame(columns=["phone", "label"])
for x in file_lst:
     res_df = pd.concat([res_df, layer1_label(os.path.join("./phone_benchmark", x))])

res_df = res_df.drop_duplicates().reset_index(drop=True)
res_df


## 第二层：对同一种value生成不同label(format)
res_valid_df = res_df[res_df["label"]] # 仅选择valid
res_valid_df.drop("label", axis=1, inplace=True) # 丢掉label(都是True)

res_valid_df["__phonenumber_parse__"] = res_valid_df["phone"].map(
     lambda x: phonenumbers.parse(x, "US")
)
res_valid_df["international"] = res_valid_df["__phonenumber_parse__"].map(
     lambda x: phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.INTERNATIONAL))
res_valid_df["national"] = res_valid_df["__phonenumber_parse__"].map(
     lambda x: phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.NATIONAL))
res_valid_df["rfc3966"] = res_valid_df["__phonenumber_parse__"].map(
     lambda x: phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.RFC3966))
res_valid_df["e164"] = res_valid_df["__phonenumber_parse__"].map(
     lambda x: phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.E164))
res_valid_df.drop("__phonenumber_parse__", axis=1, inplace=True)

res_valid_df["national"]

## 统计原始数据中，不同格式的数据占比
formats=["international", "national", "rfc3966", "e164"]
n_row=len(res_valid_df)
temp=n_row
for format in formats:
     num = (res_valid_df["phone"] == res_valid_df[format]).sum()
     print(f'{format}: {num/n_row}')
     temp -= num
print(f'poorly formated: {temp/n_row}')
# 所以data standardization还是很重要的！


###################################
## 产生数据
# 把formateed过的也作为一个instance
extended_valid_df = res_valid_df

formats=["international", "national", "rfc3966", "e164"]
for format in formats:
     single_format_df = res_valid_df.drop('phone', axis=1)
     single_format_df['phone'] = single_format_df[format]
     extended_valid_df = pd.concat([extended_valid_df, single_format_df])


extended_valid_df = extended_valid_df.drop_duplicates().reset_index(drop=True)
extended_valid_df

# invalid
invalid_srs = res_df[res_df["label"]==False]["phone"]
# invalid_srs

def generate_phone_dataset(
        valid_df,
        invalid_srs,
        n_row=500,
        targeted_format="e164", # "e164"
        invalid_ratio=0.5,
        # mixed_format_ratio=0.5, # use original data as input
):
     '''生成按要求比例配比的数据集。返回一个DataFrame，columns=["value", "label"]'''
     ## 有放回随机抽样
     if targeted_format not in {'international', 'national', 'rfc3966', 'e164'}:
          raise ValueError(f'Targeted_format is invalid.')

     n_valid = round(n_row*(1-invalid_ratio))
     res_df = valid_df.sample(n_valid, replace=True)[["phone", targeted_format]]
     res_df.rename({"phone":"value", targeted_format:"label"}, axis=1, inplace=True)
     invalid_df = pd.DataFrame({
          "value": invalid_srs.sample(n_row-n_valid, replace=True),
          "label": np.NaN
     })
     return pd.concat([res_df, invalid_df]).sample(frac=1).reset_index(drop=True)


# test_data = generate_phone_dataset(extended_valid_df, invalid_srs)
# label_clean_data = clean_phone(test_data, "value", output_format="e164", report=False)[["label", "value_clean"]]
# label_clean_data

## 计算结果
def cmp(x, y):
     return (x==y) or (x is np.NaN) and (y is np.NaN)

foo=np.frompyfunc(cmp, 2, 1)

def cal_phone_result(n_turn=30, n_row=500, targeted_format="e164"):
     acc, pre, recall = 0, 0, 0
     for i in range(n_turn):
          P, FP, TP, T = 0, 0, 0, 0
          test_data = generate_phone_dataset(extended_valid_df, invalid_srs, n_row, targeted_format)
          label_clean_data = clean_phone(test_data, "value", output_format=targeted_format, report=False)[
               ["label", "value_clean"]]
          cmp_res = foo(label_clean_data["label"], label_clean_data["value_clean"])
          acc += cmp_res.sum()/len(cmp_res)

          positive_df = label_clean_data.dropna(subset=["value_clean"])
          P = len(positive_df)
          FP = positive_df["label"].isna().sum()
          pre = pre + (P - FP) / P

          TP = P - FP
          T = (test_data["label"].isna() == False).sum()
          recall = recall+ TP / T  # 76%
     return acc/n_turn, pre/n_turn, recall/n_turn


cal_phone_result() # (0.8570666666666664, 0.9555358071666442, 0.7613333333333332)
cal_phone_result(targeted_format="national") # (0.8586666666666667, 0.9536613245273525, 0.7584000000000001)

################################
# demo
generate_phone_dataset(
     extended_valid_df, invalid_srs,
     n_row=300, targeted_format="e164", invalid_ratio=0.7)

generate_phone_dataset(
     extended_valid_df, invalid_srs,
     n_row=300, targeted_format="international", invalid_ratio=0)


# result (acc, precision, recall)
cal_phone_result() # (0.8570666666666664, 0.9555358071666442, 0.7613333333333332)
cal_phone_result(targeted_format="national") # (0.8586666666666667, 0.9536613245273525, 0.7584000000000001)

################################
## manual labeling
df = pd.read_csv(r"E:\南大\01学习资料\大四上学期\SFU_DataPrep\fake_data\phone_benchmark\manuel_labelling\national.csv")
n = len(df)
n

res_libphone = 0
for x in df["phone"]:
     try:
          phonenumbers.is_valid_number(phonenumbers.parse(str(x), "US"))
          res_libphone += 1
     except:
          print(x)
res_libphone/n

validate_phone(df["phone"]).sum()/n
df[validate_phone(df["phone"])==False]


# 单个值测试
country_code="US"
x="(760) 699-6919"
phonenumbers.is_valid_number(phonenumbers.parse(x, country_code)) # 检查是否合法
validate_phone(x)

phonenumbers.format_number(phonenumbers.parse(x, country_code), phonenumbers.PhoneNumberFormat.NATIONAL)
phonenumbers.format_number(phonenumbers.parse(x, country_code), phonenumbers.PhoneNumberFormat.INTERNATIONAL)
phonenumbers.format_number(phonenumbers.parse(x, country_code), phonenumbers.PhoneNumberFormat.RFC3966)
phonenumbers.format_number(phonenumbers.parse(x, country_code), phonenumbers.PhoneNumberFormat.E164)
# # 用clean转换格式
df=pd.DataFrame(columns=["phone"])
df.loc[0]="1-800-994-6494"
clean_phone(df, "phone",output_format="national") # 没有country_code的national

# 在文本中
lst=[]
for match in phonenumbers.PhoneNumberMatcher("Main: 212-566-1974 Chambers entr:212-566-5677 Loading Dock: 212-566-1976", country_code):
     lst.append(match.number)
# dir(lst[0])
lst
lst[0].national_number

phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.NATIONAL)
phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.RFC3966)
phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.E164)



################################
## 零散测试
cmp_res = foo(label_clean_data["label"], label_clean_data["value_clean"])
label_clean_data[cmp_res==False]
test_data[cmp_res==False] # 格式不支持
acc = cmp_res.sum()/len(cmp_res) # accuracy
acc

# 测试pre和recall
positive_df = label_clean_data.dropna(subset=["value_clean"])
P = len(positive_df)
FP = positive_df["label"].isna().sum() # validate_phone只有FN而没有FP
pre = (P-FP)/P
pre # 100%

TP = P-FP
T = (test_data["label"].isna()==False).sum()
recall = TP/T # 76%



### libphonenumber测试
df = pd.read_csv("./phone_benchmark/Directory_Of_DHS_Contacts.csv")
phone_col_name = "Phone Number"
phone = df[phone_col_name]
phone_clean = clean_phone(df, phone_col_name, inplace=True, report=False)[phone_col_name+"_clean"]
na_index_lst = phone_clean[phone_clean.isna()].index.values.tolist()
phone[na_index_lst] # 本身就是nan

## 文本中的type
df = clean_headers(pd.read_csv("./phone_benchmark/DCAS_Managed_Public_Buildings.csv"))
df.columns
phone_col_name = "custodial_borough_supervisor_phone"
phone = df[phone_col_name]
phone_clean = clean_phone(df, phone_col_name, output_format="national", inplace=True, report=False)[phone_col_name+"_clean"]
phone[na_index_lst]
phone_clean

validate_clean_lib = phone.apply(lambda x: phonenumbers.is_valid_number(phonenumbers.parse(x, "US")))
# 直接报错
# na_index_lst = phone_clean[phone_clean.isna()].index.values.tolist()
# na_index_lst_lib = validate_clean_lib[validate_clean_lib.isna()].index.values.tolist()
# phone[na_index_lst_lib]

# 从文本抓取
for match in phonenumbers.PhoneNumberMatcher("917-337-6309,  347-386-2992", "US"):
     print(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.NATIONAL))

## 不支持的type
df = clean_headers(pd.read_csv("./phone_benchmark/OWEB_Small_Grant_Teams.csv"))
df.columns
phone_col_name = "phone"
phone = df[phone_col_name]
phone_clean = clean_phone(df, phone_col_name, inplace=True, report=False)[phone_col_name+"_clean"]
validate_clean_lib = phone.apply(lambda x: phonenumbers.is_valid_number(phonenumbers.parse(x, "US")))
na_index_lst = phone_clean[phone_clean.isna()].index.values.tolist()
na_index_lst_lib = validate_clean_lib[validate_clean_lib.isna()].index.values.tolist()
phone[na_index_lst]
phone
phone_clean
phone[na_index_lst_lib]

validate_phone("(503) 935-5360") # ????
validate_phone("541-247-2755x4#")

# format 转换
help(clean_phone) # nanp, e164, national
z = phonenumbers.parse("541-247-2755x4#", "US")
z
phonenumbers.is_possible_number(z)
phonenumbers.is_valid_number(z)
