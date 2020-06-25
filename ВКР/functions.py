import numpy as np
import pandas as pd
import numpy as np, scipy.stats as st
import math
from scipy.stats import normaltest
from scipy.stats import shapiro

def count_residuals(X,Y):
    """
    Function counts standardized residuals of 3 types:
    - standardized residuals (Pearson)
    - studentized redisuals (Internally studentized residuals)
    - studentized deleted residuals (Externally studentized residuals / Jacknife residuals / R-student residuals)
    
    The output is a python dictionary with structure as follows:
    {"residuals":[], "standardized":[],"studentized":[], "deleted_studentized":[]}
    -----
    Parameters
    
    X - independent variable
    Y - dependent variable
    
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    n = len(X)
    diff_mean_sqr = np.dot((X - mean_X), (X - mean_X)) # квадрат разности между X и средним
    # np.dot - умножение 
    beta1 = np.dot((X - mean_X), (Y - mean_Y)) / diff_mean_sqr
    beta0 = mean_Y - beta1 * mean_X
    y_hat = beta0 + beta1 * X # расчет модельного значения Y
    residuals = Y - y_hat # расчет простых остатков
    h_ii = (X - mean_X) ** 2 / diff_mean_sqr + (1 / n) # диагональный элемент проекционной матрицы
    Var_e = math.sqrt(sum((Y - y_hat) ** 2)/(n-2)) # вариация остатков
    SE_regression = Var_e*((1-h_ii) ** 0.5)
    standardized_residuals = residuals/Var_e
    studentized_residuals = residuals/SE_regression # 
    deleted_studentized_residual = [r_i*math.sqrt((n-2-1)/(n-2-r_i**2)) for r_i in studentized_residuals]
    # является ли наблюдение выбросом по значению каждого из типов остатков
    out_stand_res = list(np.array(standardized_residuals)>3) and list(np.array(standardized_residuals)<-3)
    out_stud_res = list(np.array(studentized_residuals)>3) and list(np.array(studentized_residuals)< -3)
    out_del_stud_res = list(np.array(studentized_residuals)>3) and list(np.array(studentized_residuals)< -3)
    return {"residuals":residuals, "standardized":standardized_residuals,"studentized":studentized_residuals,
            "deleted_studentized":deleted_studentized_residual}


def stat_intervals(a, mode = "prop"):
    """
    a - list or array to count confidence intervals on
    mode - takes value "prop" (proportion) or "percent" 
    """
    
    res_init = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
    if mode == "prop":
        result_out = (round(res_init[0], 3), round(res_init[1], 3))
    else:
        result_out = (round(res_init[0]*100, 3), round(res_init[1]*100, 3))
    return result_out

def stat_intervals2(a, mode = "prop"):
    """
    a - list or array to count confidence intervals on
    mode - takes value "prop" (proportion) or "percent" 
    """
    
    res_init = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
    
    return res_init


def are_medium_outliers(array_):
    """
    The fucntion takes an input array and returns the number and proportion of outliers in array
    """
    IRQ = np.quantile(array_, 0.75) - np.quantile(array_, 0.25)
    num = 0
    for value in array_:
        if value >= np.quantile(array_, 0.75)+1.5*IRQ:
            # lower than 
            num+=1
            out = True
        if value <= np.quantile(array_, 0.25)-1.5*IRQ:
            num+=1
            out = True
    return num, num/10000


def out_N_obs(array_, p):
    """
    The function changes a set proportion of observations in the array into outliers
    --------
    Parameters:
    array_ - data array with values to change
    p - proportion of outliers to get
    """
    IRQ = np.quantile(array_, 0.75) - np.quantile(array_, 0.25)
    mean_ = array_.mean()
    max_ = array_.max()
    min_ = array_.min()
    array = list(array_)
    N = len(array)
    N_max = N*p*1.1
    last_n = 0
    indices = []
    while last_n < N_max: # отбираются индексы наблюдений, которые станут выбросами
        ind = int(np.random.randint(0, N))
        if ind not in indices:
            last_n +=1
            indices.append(ind)
    for index in indices:
        r = np.random.choice([-1, 1], p = [0.4, 0.6]) #предполагается, что большие значения будут преобладать
        if r == 1: #трудоголик
            h = np.array(range(130, 230, 2)) # генерируем возможноe время работы с интервалом 2/10 часа
            # генерируемые числа для удобства пока представляют собой число часов*10, в итоге часы возвращаются к нормальным значениям
            workday_prob = []
            # создаем список с вероятностями для каждого времени работы за компьютером
            for leng in h:
                if leng < 180:  # рассчитывается вероятность встречаемости значений из промежутка
                    workday_prob.append(0.5/25)
                elif leng < 200:
                    workday_prob.append(0.2/10)
                elif leng < 210:
                    workday_prob.append(0.2/5)
                else:
                    workday_prob.append(0.1/10)
#             print("Trud", workday_prob)
            # случайным убразом выбираются значения из списка h, причем указывается, насколько вероятно появление каждого из них (p)
            array[index] = np.random.choice(h/10, p = workday_prob) 
        else:
            h = np.array(range(0, 50, 2))
            workday_prob1 = []
            for leng in h:
                if leng < 16: # рассчитывается вероятность встречаемости значений из промежутка
                    workday_prob1.append(0.3/8)
                elif leng < 20:
                    workday_prob1.append(0.2/2)
                elif leng < 28:
                    workday_prob1.append(0.2/4)
                else:
                    workday_prob1.append(0.3/11)
#             print("Ne trud:", workday_prob1)
            
            array[index] = np.random.choice(h/10, p = workday_prob1)
    return array, indices