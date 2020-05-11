from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

def load_data(path):
    high_price = []
    low_price = []
    close_price=[]
    all_price=[]
    with open(path, 'r',newline="",encoding='utf-8') as csvfile:
        rows=csv.reader(csvfile)
        for i,row in enumerate(rows):
            # 表頭跳過
            if i==0: 
                continue
            # 轉成Float
            high_price.append(float(row[2]))
            low_price.append(float(row[3]))
            close_price.append(float(row[4]))
    all_price.append(high_price)
    all_price.append(low_price)
    all_price.append(close_price)
    return all_price
     
def get_stock_price(path):
    all_price = []
    path_list = os.listdir(path)
    path_list.sort()  # 對讀取的路徑進行排序
    for filename in path_list:
        all_price.append(load_data(os.path.join(path, filename)))
    return all_price

def split_data(all_price):
    high_price = all_price[0]
    low_price = all_price[1]
    close_price = all_price[2]
    X=[]
    Y=[]
    roudnm = int(len(all_price[0])/13)
    real_x=[]
    for run in range(roudnm):
        feature = []
        target = []
        for i in range(run*13,run*13+3):
            feature.append(high_price[i])
            feature.append(low_price[i])
            feature.append(close_price[i])
        for i in range(run*13+3, run*13+13):
            target.append(close_price[i])
        X.append(np.array(feature))
        Y.append(np.array(target))
    for k in range(len(close_price)-13, len(close_price)-10):
        real_x.append(high_price[i])
        real_x.append(low_price[i])
        real_x.append(close_price[i])
    
    real_y = close_price[-10:]
   
    return X, Y, real_x, real_y
def main():
    day=[1,2,3,4,5,6,7,8,9,10]
    path = 'stock_data'
    all_price = get_stock_price(path)
    # 永豐餘 #
    YFY_all_price = all_price[0]
    YFY_X, YFY_Y, YFY_real_x, YFY_real_y = split_data(YFY_all_price)
    YFY_regr = LinearRegression()
    YFY_regr.fit(YFY_X, YFY_Y)
    YFY_predict_y = YFY_regr.predict(np.array([YFY_real_x]))
    plt.plot(day, YFY_real_y, 'b',label="YFY-Real")
    plt.plot(day, list(YFY_predict_y[0]), 'g', label="YFY-Predict")
    plt.legend()
    # plt.show()
    # 永豐餘 #
    # 台積電 #
    TMSC_all_price = all_price[1]
    TMSC_X, TMSC_Y, TMSC_real_x, TMSC_real_y = split_data(TMSC_all_price)
    TMSC_regr = LinearRegression()
    TMSC_regr.fit(TMSC_X, TMSC_Y)
    TMSC_predict_y = TMSC_regr.predict(np.array([TMSC_real_x]))
    plt.plot(day, TMSC_real_y, 'r', label="TSMC-Real")
    plt.plot(day, list(TMSC_predict_y[0]), 'c', label="TSMC-Predict")
    # plt.show()
    # 台積電 #
    # 聯發科 #
    MTK_all_price = all_price[2]
    MTK_X, MTK_Y, MTK_real_x, MTK_real_y = split_data(MTK_all_price)
    MTK_regr = LinearRegression()
    MTK_regr.fit(MTK_X, MTK_Y)
    MTK_predict_y = MTK_regr.predict(np.array([MTK_real_x]))
    plt.plot(day, MTK_real_y, 'm', label="MTK-Real")
    plt.plot(day, list(MTK_predict_y[0]), 'y', label="MTK-Predict")
    plt.legend()
    plt.show()
    # 聯發科 #


if __name__ == "__main__":
    main()

