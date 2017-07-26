#----------------------------------------
# purpose: ランダムフォレストによる弁別器テストスクリプト　学習編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に正解ラベルが入っていること。
# created: 2017-07-05
#----------------------------------------
import pandas
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# データの読み込み
data = pandas.read_csv("HR_comma_sep_learn.csv")
#print(data)
trainFeature = (data.iloc[:, :-1]).values # transform to ndarray
trainLabel = (data.iloc[:, -1:]).values
trainLabel = np.ravel(trainLabel) # transform 2次元 to 1次元 ぽいこと

# 学習
clf = RandomForestClassifier()               # 学習器
clf.fit(trainFeature, trainLabel)
result = clf.score(trainFeature, trainLabel) # 学習データに対する、適合率

# 学習結果を保存
with open('entry.pickle', 'wb') as f:
	pickle.dump(clf, f)

# 1個だけテスト
test = clf.predict([trainFeature[0]])
print(test)

# 額数データに対する適合率
print(result)
print(clf.feature_importances_)	# 各特徴量に対する寄与度を求める