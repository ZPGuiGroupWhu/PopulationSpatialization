import pandas as pd
import numpy as np
<<<<<<< HEAD
import json
from conv import *
from utils import *
from sklearn.ensemble import RandomForestRegressor


# 100m
RESOLUTION = 100
N_ROW = 1540
N_COLUMN = 1250

## 200m
# RESOLUTION = 200
# N_ROW = 770
# N_COLUMN = 625

## 500m
# RESOLUTION = 500
# N_ROW = 308
# N_COLUMN = 250

relative_src = 'resource/feature_vector_of_pixel_grade_proportions_' + str(RESOLUTION) +'.txt'
county_src = r"resource/wuhanCountyPop.txt"
sub_src =  r"resource/wuhanSubDistrictPop.txt"
sub_pop =  r"resource/sub_pop.txt"
sub_pd =  r"resource/sub_population_density.txt"

subPop = pd.read_table(sub_pop,sep=',',index_col='gid')
subPop = subPop.sort_index()

sub_pd = pd.read_table(sub_pd,sep=',',index_col='gid')
sub_pd = sub_pd.sort_index()
=======
from config import *

relative_src = 'result_relative.txt'
absolute_src = 'result_absolute.txt'

tmp = pd.read_table("community_county.txt",sep=',',index_col='gid')
tmp = tmp.sort_index()
for row in tmp.iterrows():
    if row[1][0] in zhuchengqu:
        zhuchengqu_community.append(row[0])
    if row[1][0] in yuanchengqu:
        yuanchengqu_community.append(row[0])
subPop = pd.read_table(sub_pop,sep=',',index_col='gid')
subPop = subPop.sort_index()

sub_midu = pd.read_table(sub_midu,sep=',',index_col='gid')
sub_midu = sub_midu.sort_index()
community_midu = pd.read_table(community_midu,sep=',',index_col='gid')
community_midu = community_midu.sort_index()
>>>>>>> 4a4ae98319efd5f1c49261908170bf1f118cc22b


subPara_relative = pd.read_table(relative_src,sep=',',index_col='subid')
subPara_relative = subPara_relative.sort_index()



<<<<<<< HEAD

grid_src = r"..\Data\grid" + str(RESOLUTION) +".csv"
=======
import matplotlib.pyplot as plt



from conv import *
from utils import *
import random


>>>>>>> 4a4ae98319efd5f1c49261908170bf1f118cc22b
gridInfo = pd.read_table(grid_src,sep=',',index_col='id')
gridInfo = gridInfo.sort_index()

ini_matrix = np.zeros((N_ROW,N_COLUMN))

countyInfo = pd.read_table(county_src,sep=',',index_col='countyId')
county_dict = countyInfo.to_dict()['countyPopNum']
subInfo = pd.read_table(sub_src,sep=',',index_col='subId')

sub_dict = subInfo.to_dict()['subPopNum']

index_matrix = np.array(gridInfo.index).reshape((N_ROW,N_COLUMN))[::-1]#索引的二维矩阵
county_matrix = np.array(gridInfo['county_id']).reshape((N_ROW,N_COLUMN))[::-1]#countyid的二维矩阵
sub_matrix = np.array(gridInfo['sub_id']).reshape((N_ROW,N_COLUMN))[::-1]#subid的二维矩阵
<<<<<<< HEAD
building_area_matrix = np.array(gridInfo['building_area']).reshape((N_ROW,N_COLUMN))[::-1]#是否又建筑物的二维矩阵


f = open("resource/class_"+ str(RESOLUTION) +".txt",'r')
=======
has_juminhouse_index = np.array(gridInfo['has_juminhouse']).reshape((N_ROW,N_COLUMN))[::-1]#是否又建筑物的二维矩阵
building_area_matrix = np.array(gridInfo['building_area']).reshape((N_ROW,N_COLUMN))[::-1]#是否又建筑物的二维矩阵

subOfCounty = {}
for index,row in subInfo.iterrows():
    subOfCounty[row['countyId']] = {}
for index,row in subInfo.iterrows():
    subOfCounty[row['countyId']][index] = row['subPopNum']


import json
f = open("class.txt",'r')
>>>>>>> 4a4ae98319efd5f1c49261908170bf1f118cc22b
classes = json.loads(f.read())
f.close()
result_matrixs =[]


for key,value in classes.items():
    if "poi" in key:
        tmp_matrix = np.array(gridInfo['count_'+key.replace('_','')]).reshape((N_ROW,N_COLUMN))[::-1]
    elif "night" in key:
        tmp_matrix = np.array(gridInfo['mobile_night']).reshape((N_ROW,N_COLUMN))[::-1]

    for i in range(len(value)):
        if i == 0:
            ini_matrix[tmp_matrix == 0] = 1
        elif i ==  len(value)-1:
            ini_matrix[(tmp_matrix >= int(value[i-1])) & (tmp_matrix <= int(value[i]))] = 1
        else:
            ini_matrix[(tmp_matrix >= int(value[i-1])) & (tmp_matrix < int(value[i]))] = 1
        result_matrixs.append(ini_matrix)
        ini_matrix = np.zeros((N_ROW, N_COLUMN))

for mat in result_matrixs:
    mat.resize(N_ROW*N_COLUMN,1)



<<<<<<< HEAD
=======
import jenkspy
>>>>>>> 4a4ae98319efd5f1c49261908170bf1f118cc22b
poi1_matrix = np.array(gridInfo['count_poi1']).reshape((N_ROW,N_COLUMN))[::-1]
poi2_matrix = np.array(gridInfo['count_poi2']).reshape((N_ROW,N_COLUMN))[::-1]

poi7_matrix = np.array(gridInfo['count_poi7']).reshape((N_ROW,N_COLUMN))[::-1]
poi8_matrix = np.array(gridInfo['count_poi8']).reshape((N_ROW,N_COLUMN))[::-1]
poi11_matrix = np.array(gridInfo['count_poi11']).reshape((N_ROW,N_COLUMN))[::-1]
poi14_matrix = np.array(gridInfo['count_poi14']).reshape((N_ROW,N_COLUMN))[::-1]
poi21_matrix = np.array(gridInfo['count_poi21']).reshape((N_ROW,N_COLUMN))[::-1]
poi22_matrix = np.array(gridInfo['count_poi22']).reshape((N_ROW,N_COLUMN))[::-1]
poi23_matrix = np.array(gridInfo['count_poi23']).reshape((N_ROW,N_COLUMN))[::-1]
poi24_matrix = np.array(gridInfo['count_poi24']).reshape((N_ROW,N_COLUMN))[::-1]
poi3_matrix = np.array(gridInfo['count_poi3']).reshape((N_ROW,N_COLUMN))[::-1]
poi18_matrix = np.array(gridInfo['count_poi18']).reshape((N_ROW,N_COLUMN))[::-1]
poi_matrix = poi1_matrix+poi2_matrix+poi3_matrix+poi7_matrix+poi8_matrix+poi11_matrix+poi14_matrix+poi18_matrix+poi21_matrix+poi22_matrix+poi23_matrix+poi24_matrix
night_point_matrix = np.array(gridInfo['mobile_night']).reshape((N_ROW,N_COLUMN))[::-1]
night_point_matrix[np.isnan(night_point_matrix)] = 0


<<<<<<< HEAD
noPoiNoBuilding = ((np.isnan(sub_matrix)==False)&(building_area_matrix == 0) & (poi_matrix ==0))
noPoiHasBuilding = ((np.isnan(sub_matrix)==False)&(building_area_matrix > 0) & (poi_matrix ==0))
=======
noPoiNoBuilding = ((np.isnan(sub_matrix)==False)&(has_juminhouse_index == 0) & (poi_matrix ==0))
noPoiHasBuilding = ((np.isnan(sub_matrix)==False)&(has_juminhouse_index == 1) & (poi_matrix ==0) &(night_point_matrix==0))
>>>>>>> 4a4ae98319efd5f1c49261908170bf1f118cc22b



wei_matrix = np.sqrt(building_area_matrix)
max1 = max(wei_matrix[noPoiHasBuilding])
tmp1_matrix = np.zeros((N_ROW, N_COLUMN))
tmp1_matrix += 1






<<<<<<< HEAD

county = {11: 0, 18: 0, 15: 0, 14: 0, 12: 0, 6: 0, 10: 0, 3: 0, 1: 0, 16: 0, 4: 0, 5: 0, 7: 0, 9: 0, 8: 0, 17: 0, 2: 0}
for i in range(1):
    rfc = RandomForestRegressor(max_features='sqrt',n_estimators=250,min_samples_leaf=1,
                                oob_score=True,criterion="mse",bootstrap=True,random_state=i)

    rfc.fit(subPara_relative, sub_pd)
    result = rfc.predict(np.hstack(result_matrixs)).reshape((N_ROW,N_COLUMN))

    # building patch data
    _ll = list(result[poi_matrix == 0])
    _ll.sort()
    tmp1_matrix[noPoiHasBuilding] = wei_matrix[noPoiHasBuilding] / max1
    tmp1_matrix[noPoiHasBuilding] *= _ll[0]*3
    result[noPoiHasBuilding] = tmp1_matrix[noPoiHasBuilding]
    result[noPoiNoBuilding] = 0
    # #

    # mobile positioning data
=======
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

fc = 0
sum = 0
max  = 100000
sum_r2 = 0
sum_MAE=0
sum_RMSE = 0
max_MAE =1000000
max_RMSE = 1000000000
county = {11: 0, 18: 0, 15: 0, 14: 0, 12: 0, 6: 0, 10: 0, 3: 0, 1: 0, 16: 0, 4: 0, 5: 0, 7: 0, 9: 0, 8: 0, 17: 0, 2: 0}
var=0
for i in range(1):
    rfc = RandomForestRegressor(max_features='sqrt',n_estimators=250,min_samples_leaf=1,
                                oob_score=True,criterion="mse",bootstrap=True,random_state=2)
    #################################百分比 密度
    rfc.fit(subPara_relative, sub_midu)

    result = rfc.predict(np.hstack(result_matrixs)).reshape((N_ROW,N_COLUMN))

    # building patch data
    tmp1_matrix[noPoiHasBuilding] = wei_matrix[noPoiHasBuilding] / max1
    #
    tmp1_matrix[noPoiHasBuilding] *= _ll[0]*3

    result[noPoiHasBuilding] = tmp1_matrix[noPoiHasBuilding]
    # #

    # # mobile positioning data
>>>>>>> 4a4ae98319efd5f1c49261908170bf1f118cc22b
    x = list(result[poi_matrix > 0])
    x.sort()
    y = list(night_point_matrix[poi_matrix > 0])
    y.sort()
    [rows, cols] = result.shape
    for i in range(rows):
        for j in range(cols):
            if (poi_matrix[i, j] > 0 ):
                index_x = 0
                index_y = 0
                for index, val in enumerate(x):
                    if result[i,j] < val:

                        index_x = (index+0.0000001)/len(x)
                        break
                for index, val in enumerate(y):
                    if night_point_matrix[i,j] < val:
                        index_y = (index+0.0000001)/len(y)
                        break
                result[i,j] = x[int(len(x)*(index_y*0.5+index_x*0.5))]


<<<<<<< HEAD
    # spatial filtering
    result = conv(result, gau(3), county_matrix, 0)


    weight_matrix = normalize(result,county_matrix,county_dict)
    pop_matrix = calCountyPop(weight_matrix,county_matrix,county_dict)
=======
    result = conv(result, gau(3), county_matrix, 0)

    weight_matrix = normalize(result,county_matrix,county_dict)
    # # print(np.min(weight_matrix))


    pop_matrix = calCountyPop(weight_matrix,county_matrix,county_dict)

>>>>>>> 4a4ae98319efd5f1c49261908170bf1f118cc22b
    pop_dict = {}
    [rows, cols] = pop_matrix.shape
    for i in range(rows):
        for j in range(cols):
            if (np.isnan(county_matrix[i, j])):
                pass
            else:
                if (county_matrix[i, j] in pop_dict):

                    pop_dict[county_matrix[i, j]].append(pop_matrix[i,j])
                else:
                    pop_dict[county_matrix[i, j]] = []

<<<<<<< HEAD
    MAE, RMSE ,r2= calSubError(pop_matrix,sub_matrix,sub_dict)
    print("MAE :" + str(MAE))
    print("RMSE :" + str(RMSE))
    print("r2 :" + str(r2))
=======



    var += math.sqrt(np.var(pop_matrix[np.isnan(county_matrix) == False] / 0.01))
    MAE, RMSE ,r2= calSubError(pop_matrix,sub_matrix,sub_dict)
    print(MAE)

    # outGridPop(pop_matrix,index_matrix,county_matrix)
    sum_MAE += MAE
    if max_MAE > MAE:
        max_MAE = MAE
    sum_RMSE += RMSE
    if max_RMSE > RMSE:
        max_RMSE = RMSE

    sum_r2 += r2
print("MAE mean：" + str(sum_MAE / 10))

print("MAE min：" + str(max_MAE))
print("RMSE mean：" + str(sum_RMSE / 10))
print("var mean：" + str(var / 10))

print("RMSE min：" + str(max_RMSE))
print("r2 mean：" + str(sum_r2/10))
print("fc mean：" + str(fc/10))
>>>>>>> 4a4ae98319efd5f1c49261908170bf1f118cc22b

