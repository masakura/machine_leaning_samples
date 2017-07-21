#----------------------------------------
# purpose: kerasによる弁別器テストスクリプト　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に層名（クラス名）が入っていること。
# created: 2017-07-08
#----------------------------------------
import pandas
import pickle
import numpy as np
from sklearn import preprocessing # 次元毎の正規化に使う
from keras.models import model_from_json


# データの読み込み
data = pandas.read_csv("iris_test.csv")
#print(data)
x = (data.iloc[:, 0:4]).values # transform to ndarray
x = preprocessing.scale(x)     # 次元毎に正規化する
y = (data.iloc[:, 4:5]).values
y = [flatten for inner in y for flatten in inner] # transform 2次元 to 1次元 ぽいこと

# 機械学習器を復元
model = model_from_json(open('model', 'r').read())
model.load_weights('param.hdf5')

# テスト用のデータを保存
with open("test_result.csv", "w") as fw:
	test = model.predict(x)
	print(test)
	fw.write(str(test))
