import numpy as np

def conv(grid,filter,county_matrix,type):
    '''
    '''
    #赋值空矩阵作为计算结果矩阵
    # result = np.zeros(grid.shape)
    # print(type)
    result = grid
    #计算卷积算子的大小
    filter_size = filter.shape[0]
#     print(filter_size)

    #填充padding
    pad = int((filter_size-1)/2) 
    grid = zero_pad(grid,pad)
    
    for h in range(result.shape[0]):
        for w in range(result.shape[1]):
            if type == 2: ## 远城区
                if county_matrix[h,w] in [2,3,5,6,10,12,14,18]:
                    # print(county_matrix[h,w])
                    vert_start = h
                    vert_end = vert_start + filter_size
                    horiz_start = w
                    horiz_end = horiz_start + filter_size

                    slice = grid[vert_start:vert_end, horiz_start:horiz_end]
                    result[h, w] = conv_single_step(slice, filter)
            elif type == 1: ## 主城区
                if county_matrix[h,w] in [1,4,7,8,9,11,15,16,17]:
                    vert_start = h
                    vert_end = vert_start + filter_size
                    horiz_start = w
                    horiz_end = horiz_start + filter_size

                    slice = grid[vert_start:vert_end, horiz_start:horiz_end]
                    result[h,w] = conv_single_step(slice, filter)
            else:
                # print((w,h))
                vert_start = h
                vert_end = vert_start + filter_size
                horiz_start = w
                horiz_end = horiz_start + filter_size

                slice = grid[vert_start:vert_end, horiz_start:horiz_end]
                result[h, w] = conv_single_step(slice, filter)
    return result
def conv_single_step(slice,filter):   
    '''
    '''
    s = slice * filter
    # Sum over all entries of the volume s.
    result = np.sum(s)
    
    return result

def zero_pad(X, pad):
    '''
    '''
    X_pad = np.pad(X, ((pad,pad), (pad,pad)), 'constant') 
    return X_pad

    
