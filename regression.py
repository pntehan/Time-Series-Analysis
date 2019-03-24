'''python实现线性回归算法'''

import numpy as np
from numpy.linalg import det
# 计算数组的行列式
from numpy.linalg import inv
# 计算数组矩阵的逆矩阵
from numpy import mat
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
	'''
	实现线性回归算法类
	'''
	def __init__(self):
		pass

	def train(self, x_trian, y_trian):
		x_mat = mat(x_trian).T
		y_mat = mat(y_trian).T
		[m, n] = x_mat.shape
		x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
		self.weight = mat(random.rand(n+1, 1))
		if det(x_mat * x_mat) == 0:
			print('x乘以x的行列式为0')
			return
		else:
			self.weight = inv(x_mat.T * x_mat) * x_mat.T * y_mat
		return self.weight

	def locally_weighted_linear_regression(self, test_point, x_train, y_train, k=1.0):
		x_mat = mat(x_train).T
		[m, n] = x_mat.shape
		x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
		y_mat = mat(y_train).T
		test_point_mat = mat(test_point)
		test_point_mat = np.hstack((test_point_mat, mat([[1]])))
		self.weight = mat(np.zeros((n+1, 1)))
		weights = mat(np.eye((m)))
		test_data = np.tile(test_point_mat, [m, 1])
		distances = (test_data-x_mat)*(test_data-x_mat).T/(n+1)
		distances = np.exp(distances/(-2*k**2))
		weights = np.diag(np.diag(distances))
		xTx = x_mat.T*(weights*x_mat)
		if det(xTx) == 0.0:
			print('xTx行列式值为0')
			return
		self.weight = xTx.I * x_mat.T * weights * y_mat
		return test_point_mat*self.weight

	def ridge_regression(self, x_train, y_trian, lam=0.2):
		x_mat = mat(x_train).T
		[m, n] = np.shape(x_mat)
		x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
		y_mat = mat(y_trian).T
		self.weight = mat(random.rand(n+1, 1))
		xTx = x_mat.T * x_mat + lam * mat(np.eye(n))
		if det(xTx) == 0.0:
			print('xTx的行列式值为0')
			return
		self.weight = xTx.I * x_mat.T * y_mat
		return self.weight

	def lasso_regression(self, x_train, y_train, eps=0.01, itr_num=100):
		x_mat = mat(x_train).T
		[m, n] = np.shape(x_mat)
		x_mat = (x_mat-x_mat.mean(axis=0)) / x_mat.std(axis=0)
		x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
		y_mat = mat(y_train).T
		y_mat = (y_mat-y_mat.mean(axis=0)) / y_mat.std(axis=0)
		self.weight = mat(random.rand(n+1, 1))
		best_weight = self.weight.copy()
		for i in range(itr_num):
			print(self.weight.T)
			lowest_error = np.inf
			for j in range(n+1):
				for sign in [-1, 1]:
					weight_copy = self.weight.copy()
					weight_copy[j] += eps*sign
					y_predict = x_mat*weight_copy
					error = np.power(y_mat-y_predict, 2).sum()
					if error < lowest_error:
						lowest_error = error
						best_weight = weight_copy
			self.weight = best_weight
		return self.weight

	def lwlr_predict(self, x_test, x_train, y_train, k=1.0):
		m = len(x_test)
		y_predict = mat(np.zeros((m, 1)))
		for i in range(m):
			y_predict[i] = self.locally_weighted_linear_regression(x_test[i], x_train, y_train, k)
		return y_predict

	def lr_predict(self, x_test):
		m = len(x_test)
		x_mat = np.hstack((mat(x_test).T, np.ones((m, 1))))
		return x_mat * self.weight

	def plot_lr(self, x_train, y_train):
		x_min = x_train.min()
		x_max = x_train.max()
		y_min = self.weight[0] * x_min + self.weight[1]
		y_max = self.weight[0] * x_max + self.weight[1]
		plt.scatter(x_train, y_train)
		plt.plot([x_min, x_max], [y_min[0, 0], y_max[0, 0]], '-g')
		plt.show()

	def plot_lwlr(self, x_train, y_train, k=1.0):
		x_min = x_train.min()
		x_max = x_train.max()
		x = np.linspace(x_min, x_max, 1000)
		y = self.lwlr_predict(x, x_train, y_train, k)
		plt.scatter(x_train, y_train)
		plt.plot(x, y.getA()[:, 0], '-g')
		plt.show()

	def plot_weight_with_lambda(self, x_train, y_train, lambdas):
		weights = np.zeros((len(lambdas), ))
		for i in range(len(lambdas)):
			self.ridge_regression(x_train, y_train, lambdas[i])
			weights[i] = self.weight[0]
		plt.plot(np.log(lambdas), weights)
		plt.show()

def main():
	# 读取数据
	data = pd.read_csv('./Data/regression.csv')
	data = data / 30
	# 分割数据
	x_train = data['x'].values
	y_train = data['y'].values
	# 实例化类
	regression = LinearRegression()
	# 计算最优权重
	# regression.lasso_regression(x_train, y_train, itr_num=1000)
	# # 线性回归画图
	# regression.plot_lr(x_train, y_train)
	# regression.plot_lwlr(x_train, y_train, 1)
	# regression.train(x_train, y_train)
	y_predict = regression.lwlr_predict([[15], [20]], x_train, y_train, k=0.01)
	print(y_predict)
	regression.ridge_regression(x_train, y_predict, lam=3)
	regression.plot_lr(x_train, y_train)

if __name__ == '__main__':
	main()












