'''对时间序列的预处理，判断属于平稳序列还是非平稳序列，判断是否为白噪声序列'''

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# 单位根检验
def process(info):
	result = {}
	result['ADF'] = info[0]
	result['P-value'] = info[1]
	result['critical'] = info[4]
	return result

def test_stochastic(ts, log):
	return acorr_ljungbox(ts, lags=log)[1]

data = pd.read_csv('./Data/yadz.csv')
# print(data)
x = data.Time.values[:365]
y1 = data.Data.values[:365]
# plt.plot(x, y1)
# plt.show()
y2 = data.Data.diff(1).fillna(method='bfill').values[:365]
# plt.plot(x, y2)
# plt.show()
y3 = data.Data.diff(2).fillna(method='bfill').values[:365]
# plt.plot(x, y3)
# plt.show()

ts_diff = pd.DataFrame({'Data': y2})

print('原始数据:\n{}'.format(process(sm.tsa.stattools.adfuller(y1))))
# print(test_stochastic(y1, 1))
print('一阶差分数据:\n{}'.format(process(sm.tsa.stattools.adfuller(y2))))
# print(test_stochastic(y2, 1))
print('二阶差分数据:\n{}'.format(process(sm.tsa.stattools.adfuller(y3))))
# print(test_stochastic(y3, 1))
'''如果ADF的值小于critical中三个值且P-value趋向于0，则该时间序列是平稳序列'''

'''确定ARMA的阶数'''

# 利用AIC, BIC统计量自动确定
import statsmodels.tsa.stattools as st

order = st.arma_order_select_ic(y1, max_ar=5, max_ma=5, ic=['aic', 'bic', 'hqic'])
print(order.bic_min_order)











