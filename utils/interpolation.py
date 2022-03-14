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
    ckd_dict[i[0]] = [i[1], i[2], i[3]]
lab_list = lab.values.tolist()



######### start deleting & labeling #########
print("start deleting")


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
            lab_list.remove(item) #delete the data which end_data < checkdate
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




######### start interpolating #########
print("start interpolating")

interpolated = np.array([]) # the final interpolarting result stores here
submatix_head = 0
submatrix_tail = 0

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
        subarray = pd.DataFrame(subarray)
        subarray.interpolate(method="linear", inplace = True)
        subarray.fillna(method = "bfill", inplace=True)
        subarray = subarray.values
        for k in range(len(subarray[0])-1):
            if len(subarray) > 4:
                for j in range(len(subarray)):
                    
                    if j > 1 and j < len(subarray)-2:
                        subarray[j][k] = (subarray[j-2][k] + subarray[j-1][k] + subarray[j][k] + subarray[j+1][k] + subarray[j+2][k])/5 # add the moving average filter for the nearest five data (optional)
        if interpolated.size == 0:
            interpolated = subarray
        else:
            interpolated = np.concatenate((interpolated, subarray), axis = 0)
        pre_name = cur_name
        submatix_head = i
        submatrix_tail = i+1

######### start filling #########
print("start filling")

interpolate = { 1:[1.0, 0.9], 2:[3.95, 3.95], 3:[141, 141], 4:[13.5, 13.5], 5:[15.5, 13.5], \
               6:[34.3, 34.4], 7:[267, 267], 8:[6.3, 6.3], 9:[4.6, 4.6], 10:[3.75, 3.75],  \
               11:[6, 4.45], 12:[9.45, 9.45], 13:[150, 150], 14:[100, 100], 15:[150, 150] }
#dict = {column of data : [mean value filling for male, mean value filling for female]}

interpolated = interpolated.tolist()
for i in range(len(interpolated)):
    sex = int(interpolated[i][-1])
    for j in range(len(interpolated[i])):
        if math.isnan(float(interpolated[i][j])):
            interpolated[i][j] = interpolate[j+1][sex]
interpolated = np.array(interpolated)
interpolated = interpolated[:, 0:-1]
interpolated = np.array(interpolated, dtype = 'float')
data_mean = np.nanmean(interpolated, axis=0)
data_var = np.nanmean(np.power((interpolated - np.nanmean(interpolated, axis=0)), 2), axis=0)
interpolated = ((interpolated - data_mean)/(data_var+0.0008)).tolist()

df_label = pd.DataFrame(intervals)
df_data = pd.DataFrame(interpolated)
pd.DataFrame.to_csv(df_data, 'preprocessed_data/tsaoapp_with_filter_data.csv', index = False,header = False)
pd.DataFrame.to_csv(df_label, 'preprocessed_data/tsaoapp_with_filter_label.csv', index_label=False, index = False,header = False)
