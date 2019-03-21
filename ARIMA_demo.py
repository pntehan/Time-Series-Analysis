from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

series = pd.read_csv('./Data/yadz.csv', index_col='Time')
series_diff = series.diff(1).dropna()

# # 自相关图、偏序相关图
# plot_pacf(series, title='偏自相关图').show()
# plot_acf(series, title='自相关图').show()

# 分割数据
X = series.values
train, test = X[:-365], X[-365:]
history = [x for x in train]
prediction = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0, 1, 0)).fit()
	yhat = model.forecast()[0]
	prediction.append(yhat)
	obs = test[t]
	history.append(obs)

error = mean_squared_error(test, prediction)
plt.plot(test, label='期望值')
plt.plot(prediction, label='预测值')
plt.legend()
plt.title('Test MSE:%.3f'%error)
plt.show()
plt.savefig('ARIMA.png')




