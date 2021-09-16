import pandas as pd
import numpy as np


# 100 200 500
RESOLUTION = 500

N_ROW = 0
N_COLUMN = 0

if RESOLUTION == 100:
    N_ROW = 1540
    N_COLUMN = 1250

elif RESOLUTION == 200:
    N_ROW = 770
    N_COLUMN = 625
elif RESOLUTION == 500:
    N_ROW = 308
    N_COLUMN = 250

grid_src = r"..\Data\grid" + str(RESOLUTION) + ".csv"



#读取街道人口的相关信息
sub_midu = pd.read_table("../Code/resource/sub_midu.txt",sep=',',index_col='gid')
sub_midu = sub_midu.sort_index()

from utils import *
import random

#
# #读取格网的相关信息
gridInfo = pd.read_table(grid_src,sep=',',index_col='id')
gridInfo = gridInfo.sort_index()

# #county的id和人口
countyInfo = pd.read_table( "../Code/resource/wuhanCountyPop.txt",sep=',',index_col='countyId')
county_dict = countyInfo.to_dict()['countyPopNum']
# #sub的id和人口
subInfo = pd.read_table(r"../Code/resource/wuhanSubDistrictPop.txt",sep=',',index_col='subId')
sub_dict = subInfo.to_dict()['subPopNum']
#
index_matrix = np.array(gridInfo.index).reshape((N_ROW,N_COLUMN))[::-1]#索引的二维矩阵
county_matrix = np.array(gridInfo['county_id']).reshape((N_ROW,N_COLUMN))[::-1]#countyid的二维矩阵
sub_matrix = np.array(gridInfo['sub_id']).reshape((N_ROW,N_COLUMN))[::-1]#subid的二维矩阵



###########################

sub_value = pd.read_table("sub_value.txt",  sep=',' , dtype=float,
                            usecols=['light', 'dem', 'ndvi','road_dis', 'poi_dis', 'poi'])

grid_value = pd.read_table("grid_value.txt",  sep=',' , dtype=float,
                            usecols=['light', 'dem', 'ndvi','road_dis', 'poi_dis', 'poi'])
##########################


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
#24
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
for i in range(10):
    rfc = RandomForestRegressor(max_features='sqrt',n_estimators=250,min_samples_leaf=1,
                                oob_score=True,criterion="mae",bootstrap=True,random_state=i)
    rfc.fit(sub_value, sub_midu)

    result = rfc.predict(grid_value).reshape((N_ROW, N_COLUMN))[::-1]

    weight_matrix = normalize(result,county_matrix,county_dict)


    pop_matrix = calCountyPop(weight_matrix,county_matrix,county_dict)

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

    var += math.sqrt(np.var(pop_matrix[np.isnan(county_matrix) == False] / 0.01))
    # print(var)
    MAE, RMSE ,r2= calSubError(pop_matrix,sub_matrix,sub_dict)
    print(MAE)

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
