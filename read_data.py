'''
读取txt数据将其保存为csv文件格式，方便后续工作的进行与开展
'''

from glob import *
import pandas as pd
import threading
import os

dirs = glob('./水氡数据/*.txt')
root = './Data'
name = ['gzsh', 'xcdz', 'yadz']

def get_data(path):
    # 从文件中获取数据返回
    print('从{}获取数据中...'.format(path))
    with open(path, 'r') as fp:
        time = []
        data = []
        for x in fp.readlines():
            y = x.strip('\n')
            time.append(int(y[:8]))
            data.append(float(y[8:]))
    return time, data

def to_csv(time, data, filename):
    # 将数据转换为数据集写入新的路径
    print('写入数据中...')
    frame = {'Time': time, 'Data': data}
    output = pd.DataFrame(frame, columns=['Time', 'Data'])
    output.to_csv(root+'/'+filename+'.csv')
    print('数据已写入{}...'.format(root+'/'+filename+'.csv'))

def main(path, filename):
    time, data = get_data(path)
    to_csv(time, data, filename)

if __name__ == '__main__':
    if not os.path.exists(root):
        os.makedirs(root)
    for i in range(len(dirs)):
        thread = threading.Thread(target=main, args=(dirs[i], name[i]))
        thread.start()
        thread.join()
