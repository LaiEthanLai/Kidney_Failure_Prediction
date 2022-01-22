import pandas as pd
import numpy as np
import math
from operator import add

def time_substraction(examine_date: str, pivot: str) -> int:
    e_year, e_month, e_date = examine_date.split('-')
    p_year, p_month, p_date = pivot.split('-')
    return np.round( ( ( 365*int(p_year) + 30*int(p_month) + int(p_date))  - \
    ( 365*int(e_year) + 30*int(e_month) + int(e_date) ) ) / 365.0 )

specific_features = ['PERSONID2' , 'LABDATE', 'B_CRE', 'B_K', 'B_NA', 'B_UN', 'Hemoglobin',
                    'MCHC', 'PLT', 'WBC', 'Albumin', 'B_P', 'B_UA',
                    'Calcium', 'Triglyceride', 'LDL', 'UPCR']

ckd = pd.read_csv('raw_data/new_ckd.csv', usecols=['PERSONID2','SEX', 'END_DATE1', 'OUTC1'])
lab = pd.read_csv('raw_data/new_lab.csv', usecols=specific_features)
ckd_set = set(ckd.values[:, 0])
ckd_dict = {}
for i in ckd.values:
    ckd_dict[i[0]] = [i[1], i[2], i[3]] #[sex, end_date, outc] 



######### start deleting #########
print("start deleting")


lab_list = lab.values.tolist()
intervals = []
data_list = []
i = -1
for item in lab_list:
    i += 1
    if i > 1 and i < len(lab_list)-2:
        if item[0]!=lab_list[i-1][0] and item[0]!=lab_list[i+1][0]: #把只有一個的id刪掉
            continue
    if (item[0] not in ckd_set) or (ckd_dict[item[0]][2] == 0) or isinstance(item[1], float):
        continue
    else:
        sex, checkdate = ckd_dict[item[0]][0:2]
        interval = time_substraction(item[1], checkdate) #return end_date - checkdate
        interval = int(interval)
        if interval < 0:
            lab_list.remove(item)
            continue
        elif interval > 5 and interval <= 10:
            interval = 6
        elif interval > 10:
            interval = 7
        intervals.append(interval)
        item.append(sex)
        p = [item[0]] + item[2:]
        data_list.append([item[0]] + item[2:])

personal_wise_mean = {} # key:name -> value:mean of 16 variables
pre_name = data_list[0][0]
personal = []
count_number = [0] * 16
submatix_head = 0
submatrix_tail = 0



######### start computing mean #########
print("start computing mean")

for i in range(len(data_list)):
    cur_name = data_list[i][0]
    if cur_name == pre_name and i != len(data_list)-1:
        submatrix_tail += 1
    else:
        if i == len(data_list)-1:
            subarray = np.array(data_list)[submatix_head : submatrix_tail+1, 1:]
        else:
            subarray = np.array(data_list)[submatix_head : submatrix_tail, 1:]
        subarray = np.array(subarray,dtype = 'float')
        subarray_mean = np.nanmean(subarray, axis=0)
        subarray = subarray.values
        if i == len(data_list)-1:
            personal_wise_mean[cur_name] = subarray_mean
        else :
            personal_wise_mean[pre_name] = subarray_mean
        pre_name = cur_name
        submatix_head = i
        submatrix_tail = i+1




######### start filling mean #########
print("start filling mean")

interpolate = {1:[1.0, 0.9], 2:[3.95, 3.95], 3:[141, 141], 4:[13.5, 13.5], 5:[15.5, 13.5], \
               6:[34.3, 34.4], 7:[267, 267], 8:[6.3, 6.3], 9:[4.6, 4.6], 10:[3.75, 3.75],  \
               11:[6, 4.45], 12:[9.45, 9.45], 13:[150, 150], 14:[100, 100],15:[150, 150]}
               
for i in range(len(data_list)):
    cur_name = data_list[i][0]
    sex = data_list[i][-1]
    for j in range(len(data_list[i])-1):
        if math.isnan(float(data_list[i][j+1])):
            if math.isnan(personal_wise_mean[cur_name][j]):
                data_list[i][j+1] = interpolate[j+1][sex]
            else:
                data_list[i][j+1] = personal_wise_mean[cur_name][j]
data_list = np.array(data_list, dtype = 'float')[:, 1:-1]
data_mean = np.nanmean(data_list, axis=0)
data_var = np.nanmean(np.power((data_list - np.nanmean(data_list, axis=0)), 2), axis=0)
data_list = ((data_list - data_mean)/(data_var+0.0008)).tolist()


df_label = pd.DataFrame(intervals)
df_data = pd.DataFrame(data_list)
pd.DataFrame.to_csv(df_data, 'preprocessed_data/ftcapp_data.csv', index = False,header = False)
pd.DataFrame.to_csv(df_label, 'preprocessed_data/ftcapp_label.csv', index_label=False, index = False,header = False)