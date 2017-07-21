# purpose: kerasによるiris識別プログラム　学習編
# 教師データの正解ラベルは文字列でクラスが指定されている事を想定している。
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に層名（クラス名）が入っていること。
# created: 2017-07-08
import numpy as np
import pandas
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from sklearn import preprocessing # 次元毎の正規化に使う
from sklearn.preprocessing import OneHotEncoder # 判別問題における数値の正解ラベルをベクトル化するライブラリ
from sklearn.feature_extraction import DictVectorizer # 判別問題における文字列による正解ラベルをベクトル化する



# データの読み込み
df = pandas.read_csv("iris_learning.csv")
df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True) # ランダムに並べ替える（効果高い）
x = (df.iloc[:, 0:4]).values # transform to ndarray
x = preprocessing.scale(x) # 次元毎に正規化する
y = (df.iloc[:, 4:5]).values
print("x", x)
print("y", y)

# 正解ラベルを01のリストを作成
vec = DictVectorizer()
y = vec.fit_transform([{"class":mem[0]} for mem in y]).toarray() # 判別問題における文字列による正解ラベルをベクトル化する
print("y", y)

# 学習器の準備
model = Sequential()
model.add(Dense(2, input_shape=(4, ))) # 入力層は全結合層で入力が4次元のベクトルで、出力先のユニット数が16。活性化関数はなし（または次で定義）。
model.add(Activation('relu')) # 全結合層で、活性化関数がReLU。この書き方だと、上のdenseがReLUであると指定している
model.add(Dense(3))
model.add(Activation('softmax')) # 出力が合計すると1になる
model.compile(optimizer='adam',
      loss='categorical_crossentropy', # binary_crossentropy
      metrics=['accuracy'])

# 学習
model.fit(x, y, epochs=500, batch_size=50, verbose=1) # nb_epochは古い引数の書き方なので、epochsを使う@2017-07

# 学習のチェック
result = model.predict_classes(x, batch_size=5, verbose=0) # クラス推定
print("result1: ", result)
result = model.predict(x, batch_size=5, verbose=0) # 各ユニットの出力を見る
print("result2: ", result)
result = model.predict_proba(x, batch_size=5, verbose=0) # 確率を出す（softmaxを使っているので、predictと同じ出力になる）
print("result3: ", result)


# 学習結果を保存
print(model.summary()) # レイヤー情報を表示(上で表示させると流れるので)
open("model", "w").write(model.to_json())
model.save_weights('param.hdf5')

