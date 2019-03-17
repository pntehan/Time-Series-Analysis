# 对于python实现时间序列分析的更深入理解

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./Data/gzsh.csv').set_index('Time').head(365)
data_diff = data.diff(1).dropna()
# print(data_diff)
# print(data.head())
# plt.plot(data)
# plt.show()

# 稳定性检测，绘制滚动统计，ADF检测

def rolling_statistics(timeseries, name):
	# 绘制滚动统计
	rolmean = timeseries.rolling(window=12).mean()
	rolstd = timeseries.rolling(window=12).std()
	# print(rolmean.head())
	# print(rolstd.head())
	orig = plt.plot(timeseries, color='blue', label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	std = plt.plot(rolstd, color='black', label='Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation({})'.format(name))
	plt.show(block=False)

from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries, name):
	# ADF值检验
	rolling_statistics(timeseries, name)# 绘图
	print('ADF检验的结果:')
	dftest = adfuller(timeseries, autolag='AIC')
	dfoutput = pd.Series(dftest[:4],
						index=['Test-Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
	for key, value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)

# adf_test(data.Data[:365], name='Origin')
# # 结果是不平稳时间序列，则使用差分的方法使其趋于平稳
# adf_test(data_diff.Data[:365], name='Diff1')


# # 使用自相关图与偏自相关图再次检验时间序列的平稳性
# import statsmodels.api as sm
#
# def acf_pacf_plot(ts_log_diff):
# 	sm.graphics.tsa.plot_acf(ts_log_diff, lags=40)
# 	sm.graphics.tsa.plot_pacf(ts_log_diff, lags=40)
#
# acf_pacf_plot(data_diff)

# 参数与模型选择
import sys
from statsmodels.tsa.arima_model import ARMA, ARIMA
import numpy as np

def _proper_model(ts_log_diff, maxlag):
	best_p = 0
	best_q = 0
	best_bic = sys.maxsize
	best_model = None
	for p in np.arange(maxlag):
		for q in np.arange(maxlag):
			model = ARMA(ts_log_diff, order=(p, q))
			try:
				results_ARMA = model.fit(disp=-1)
			except:
				continue
			bic = results_ARMA.bic
			# print(bic, best_bic)
			if bic<best_bic:
				best_p = p
				best_q = q
				best_bic = bic
				best_model = results_ARMA
		print('第{}次完成...'.format(p+1))
	return best_p, best_q, best_model
# print(_proper_model(data_diff, maxlag=9))



# AR模型,q=0
model = ARIMA(data, order=(1, 1, 0))
results_AR = model.fit(disp=-1)
plt.plot(data_diff, color='blue', label='Diff1')
plt.plot(results_AR.fittedvalues, color='red', label='forecast')
plt.title('RSS:%.4f'%(sum((results_AR.fittedvalues.values-data_diff.values)**2)))
plt.show()
# second = pd.read_csv('./Data/gzsh.csv')[365:396].set_index('Time')
# forecast_n = 31
# forecast_AR = results_AR.forecast(forecast_n)[0]
# AR_data = pd.DataFrame({'Data':forecast_AR}, index=second.index)
# plt.plot(second, color='blue')
# plt.plot(AR_data, color='red')
# plt.title('AR_Model')
# plt.show()

# # MR模型,p=0
# model = ARIMA(data, order=(0, 1, 1))
# results_AR = model.fit(disp=-1)
# # plt.plot(data_diff, color='blue', label='Diff1')
# # plt.plot(results_AR.fittedvalues, color='red', label='forecast')
# # # plt.title('RSS:%.4f'%(sum((results_AR.fittedvalues-data_diff)**2)))
# # plt.show()
# second = pd.read_csv('./Data/gzsh.csv')[365:396].set_index('Time')
# forecast_n = 31
# forecast_AR = results_AR.forecast(forecast_n)[0]
# AR_data = pd.DataFrame({'Data':forecast_AR}, index=second.index)
# plt.plot(second, color='blue')
# plt.plot(AR_data, color='red')
# plt.title('MR_Model')
# plt.show()
#
# # ARIMA模型
# model = ARIMA(data, order=(1, 1, 1))
# results_AR = model.fit(disp=-1)
# # plt.plot(data_diff, color='blue', label='Diff1')
# # plt.plot(results_AR.fittedvalues, color='red', label='forecast')
# # # plt.title('RSS:%.4f'%(sum((results_AR.fittedvalues-data_diff)**2)))
# # plt.show()
# second = pd.read_csv('./Data/gzsh.csv')[365:396].set_index('Time')
# forecast_n = 31
# forecast_AR = results_AR.forecast(forecast_n)[0]
# AR_data = pd.DataFrame({'Data':forecast_AR}, index=second.index)
# plt.plot(second, color='blue')
# plt.plot(AR_data, color='red')
# plt.title('ARIMA_Model')
# plt.show()











