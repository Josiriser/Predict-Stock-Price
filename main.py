from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

def load_data(path):
    price=[]
    with open(path, 'r',newline="",encoding='utf-8') as csvfile:
        rows=csv.reader(csvfile)
        for i,row in enumerate(rows):
            # 表頭跳過
            if i==0: 
                continue
            # 把收盤價轉成Float
            price.append(float(row[4]))
    return price
     
def get_stock_price(path):
    all_price = []
    path_list = os.listdir(path)
    path_list.sort()  # 對讀取的路徑進行排序
    for filename in path_list:
        all_price.append(load_data(os.path.join(path, filename)))
    return all_price

def main():
    path = 'stock_data'
    all_price = get_stock_price(path)
   
    x = all_price[1]
    x_array=np.array(x)
    X = x_array.reshape(len(x), 1)
    y = all_price[1]
    y_array=np.array(y)
    #將線性回歸的函式庫載入，準備要執行線性回歸

    #首先開一台線性回歸機
    regr = LinearRegression()
    #目前x array是1x50的矩陣
    #但是要做線性回歸計算的話，需改成 50x1 array
    #將X設定為50x1的矩陣
    # X = x.reshape(50, 1)


    #透過LinearRegression.fit()去進行機器學習
    #參數餵給他修正過後的X以及正確答案y
    regr.fit(X, y)

    #取出機器學習的結果LinearRegression.predict
    #注意這裡傳入的參數是修正過的X
    Y = regr.predict(X)

    #最後就可以來檢查是否機器學習出來的曲線是否跟原本的曲線接近
    # plt.scatter(x, y)
    plt.plot(x,y, 'r')
    plt.plot(x, Y, 'g')
    plt.show()
    print(123)


if __name__ == "__main__":
    main()

