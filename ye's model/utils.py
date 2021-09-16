import numpy as np
import math
# from sklearn.metrics import r2_score
# from sklearn.metrics import r2
def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

def corrcoef(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=math.sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den

def single_county_normalize(weigtht_matrix,county_matrix,countyid):
    '''
    对某个区县内的人口权重归一化
    '''
    isCurCounty = (county_matrix == countyid)
    _sum = np.sum(weigtht_matrix[isCurCounty])
    weigtht_matrix[isCurCounty] = weigtht_matrix[isCurCounty]/_sum
    
    return weigtht_matrix

def normalize(weigtht_matrix,county_matrix,county_dict):
    '''
    对计算的权重按区县进行归一化
    weight_matrix:
    county_matrix:
    county_dict:包含countyid的list
    '''
    for key,value in county_dict.items():
        weigtht_matrix = single_county_normalize(weigtht_matrix,county_matrix,key)
        
    return weigtht_matrix

def calCountyPop(weigtht_matrix,county_matrix,county_dict):
    '''
    
    county_dict: index为county的id  value为county的人口数  的 字典
    '''
    for key,value in county_dict.items():
        weigtht_matrix = calSingleCountyPop(weigtht_matrix,county_matrix,key,value)
    
    return weigtht_matrix

def calSingleCountyPop(weigtht_matrix,county_matrix,countyid,countypop):
    '''
    计算某一个区县的格网人口
    '''
    isCurCounty = (county_matrix == countyid)
    weigtht_matrix[isCurCounty] = weigtht_matrix[isCurCounty]*countypop
    
    return weigtht_matrix

##############################################
##############################################
####下面是误差计算阶段########################


def calSingleSubError(pop_matrix,sub_matrix,subid,subpop):
    '''
    计算某一个街道的误差
    '''
    isCurSub = (sub_matrix == subid)
    _sum = np.sum(pop_matrix[isCurSub])
    error = _sum - subpop
    
    return error

def calSubError(pop_matrix,sub_matrix,sub_dict):
    '''
    计算街道级别MAE
    sub_dict: key 为 subid   value为subpop 的字典
    '''
    error_sum = 0
    error_mean = 0
    rmse_sum = 0
    y_true = []
    y_pred = []
    f = open('wucha_sub.txt','w')
    for key,value in sub_dict.items():
        y_true.append(value)

        error = calSingleSubError(pop_matrix,sub_matrix,key,value)
#         print(error)
        y_pred.append(value+error)
        f.write(str(key)+','+str(int(error))+'\n')
        error_sum += math.fabs(error)
        rmse_sum += error*error

    error_mean = error_sum / len(sub_dict)
    f.close()
    r2 = corrcoef(y_true,y_pred)**2

    return (error_mean, int(math.sqrt(rmse_sum / len(sub_dict))),r2)

def calCountyMAE(pop_matrix,sub_matrix,subOfCounty):
    
    countyMAE_dict = {}
    for key,value in subOfCounty.items():
        countyMAE_dict[int(key)] = int(calSingleCountyMAE(pop_matrix,sub_matrix,value))
        
    return countyMAE_dict

def calSingleCountyMAE(pop_matrix,sub_matrix,subOfCounty_dict):
    '''
    计算某一个街道的误差
    '''
    error_sum = 0
    error_mean = 0
    for key,value in subOfCounty_dict.items():
        error_sum += math.fabs(calSingleSubError(pop_matrix,sub_matrix,key,value))
    error_mean = error_sum / len(subOfCounty_dict)
    return error_mean

##############################################3
##############################################
##激活函数

def sigmoid(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            try:
                matrix[i,j] = 1.0 / (1 + 1/ math.exp(matrix[i,j]))
            except:
                print(matrix[i,j])
    return matrix

def outGridPop(pop_matrix,index_matrix,grid_scale,MAE,county_matrix):
    f = open('pop_grid'+str(grid_scale)+"&MAE="+str(MAE)+'.txt','w')
    for row in range(index_matrix.shape[0]) :
        for coloumn in range(index_matrix.shape[1]) :
            if county_matrix[row,coloumn] >=1 and county_matrix[row,coloumn] < 200:
                f.write(str(index_matrix[row][coloumn])+","+str(pop_matrix[row][coloumn])+"\n")
    f.close()


e = np.e
pi = np.pi
d =1.2
def g(x,y):
    return e**(-(x**2+y**2)/(2*d**2))

def gau(n):
    moban = np.zeros((n,n),dtype=int)
    h=int(n/2)
    for i in range(n):
        for j in range(n):
            b=g(i-h,j-h)/g(h,h)
            moban[i,j]=b
    return moban


# def calR2(pop_matrix,sub_matrix,sub_dict):
#     '''
#     计算街道级别MAE
#     sub_dict: key 为 subid   value为subpop 的字典
#     '''
#     y_true = []
#     y_pred = []
#     for key,value in sub_dict.items():
#         y_true.append(value)
#
#         error = calSingleSubError(pop_matrix,sub_matrix,key,value)
# #         print(error)
#         y_pred.append(value+error)
#
#     error_mean = error_sum / len(sub_dict)
#     r2 = corrcoef(y_true,y_pred)**2
#
#     return (error_mean, int(math.sqrt(rmse_sum / len(sub_dict))),r2)