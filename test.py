from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#開發互動功能要引用的函式庫
from ipywidgets import interact
from ipywidgets import interact_manual

#熊貓是python的excel
import pandas as pd

#先重複同樣6.4課程的動作，模擬出假數據
x = np.linspace(0, 5, 50)
y = 1.2 * x + 0.8 + 0.6 * np.random.randn(50)

#然後畫圖確認一下假數據沒錯！
plt.scatter(x, y)
plt.plot(x, 1.2 * x + 0.8, 'r')

#將線性回歸的函式庫載入，準備要執行線性回歸

#首先開一台線性回歸機
regr = LinearRegression()

#目前x array是1x50的矩陣
#但是要做線性回歸計算的話，需改成 50x1 array
x

#將X設定為50x1的矩陣
X = x.reshape(50, 1)

#經確認的確是50x1 array
X

#透過LinearRegression.fit()去進行機器學習
#參數餵給他修正過後的X以及正確答案y
regr.fit(X, y)

#取出機器學習的結果LinearRegression.predict
#注意這裡傳入的參數是修正過的X
Y = regr.predict(X)

#最後就可以來檢查是否機器學習出來的曲線是否跟原本的曲線接近
plt.scatter(x, y)
plt.plot(x, 1.2 * x + 0.8, 'r')
plt.plot(x, Y, 'g')
plt.show()
#說實在綠色線跟紅色線還蠻接近的！成功！
