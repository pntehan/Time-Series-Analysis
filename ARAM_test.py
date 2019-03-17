from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

trainSeting = pd.read_csv('./Data/yadz.csv').set_index('Time').head(365)
ts_diff = trainSeting.diff(1).dropna()
# print(ts_diff)
model = ARIMA(trainSeting, order=(1, 1, 1))
result_AR = model.fit(disp=-1)

# plt.plot(trainSeting)
# plt.plot(result_AR.fittedvalues, color='red')
# # plt.show()
# # plt.title('RSS: %.4f'%sum((result_AR.fittedvalues.values-trainSeting['Data'].values)**2))
# plt.show()

forecast_n = 365
forecast_AR = result_AR.forecast(forecast_n)[0]


two = pd.read_csv('./Data/yadz.csv').set_index('Time')[365:730]
AR_restored = forecast_AR
print(AR_restored)
new = pd.DataFrame({'Time': two.index, 'Data': AR_restored}).set_index('Time')
plt.plot(two, color='blue')
plt.plot(new, color='red')
plt.show()
# plt.savefig('first_model.png')



