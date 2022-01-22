import pandas as pd
import numpy as np
import math

def time_substraction(examine_date: str, pivot: str) -> int:
    e_year, e_month, e_date = examine_date.split('-')
    p_year, p_month, p_date = pivot.split('-')
    return np.round( ( ( 365*int(p_year) + 30*int(p_month) + int(p_date))  - \
    ( 365*int(e_year) + 30*int(e_month) + int(e_date) ) ) / 365.0 )

specific_features = ['PERSONID2' , 'LABDATE', 'B_CRE', 'B_K', 'B_NA', 'B_UN', 'Hemoglobin',
                    'MCHC', 'PLT', 'WBC', 'Albumin', 'B_P', 'B_UA',
                    'Calcium', 'Triglyceride', 'LDL', 'UPCR']


ckd = pd.read_csv('new_ckd.csv', usecols=['PERSONID2','END_DATE1', 'OUTC1', 'SEX'])
lab = pd.read_csv('new_lab.csv', usecols=specific_features)
ckd_set = set(ckd.values[:, 0])


ckd_dict = {}
for i in ckd.values:
    ckd_dict[i[0]] = [i[1], i[2], i[3]]

lab_list = lab.values.tolist()
intervals = []
data_list = []
for item in lab_list:
    print(f'value = {ckd_dict[item[0]][1]}')
    if item[0] not in ckd_set or ckd_dict[item[0]][2] == 0 or isinstance(ckd_dict[item[0]][1], float):
        continue
    else:
        sex, checkdate = ckd_dict[item[0]][0:1]
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
        data_list.append(item[2:])

interpolate = {1:[1.0, 0.9], 2:[3.95, 3.95], 3:[141, 141], 4:[13.5, 13.5], 5:[15.5, 13.5], \
               6:[34.3, 34.4], 7:[267, 267], 8:[6.3, 6.3], 9:[4.6, 4.6], 10:[3.75, 3.75],  \
               11:[6, 4.45], 12:[9.45, 9.45], 13:[200, 200], 14:[180.5, 158.9], 15:[150, 150], \
               16:[96.59442943, 96.59442943], 17:[5, 5], 18:[6.5, 6.5], 19:[60, 60], 20:[100, 100], \
               21:[150, 150]}


data_np = np.array(data_list)
data_mean = np.nanmean(data_np, axis=0)
data_var = np.nanmean(np.power((data_np - np.nanmean(data_np, axis=0)), 2), axis=0)

for i in range(data_np.shape[0]):
    sex = data_list[i][-1]
    for j in range(data_np.shape[1]-1):
        if math.isnan(data_np[i][j]) is True:
            data_np[i][j] = interpolate[j+1][sex] + np.random.normal(data_mean[j], data_var[j])

data_list = ((data_np - data_mean)/(data_var+0.0008)).tolist()

df_label = pd.DataFrame(intervals)
df_data = pd.DataFrame(data_list)
pd.DataFrame.to_csv(df_data, 'data.csv', index = False,header = False)
pd.DataFrame.to_csv(df_label, 'label.csv', index_label=False, index = False,header = False)
