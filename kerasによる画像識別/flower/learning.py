# purpose: kerasによる花の画像を利用したCNNのテスト　学習編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 
# created: 2018-02-17
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import os


data_format = "channels_last"


def openfile(dir_name, data_format="channels_last"):
    """ 画像をリストとして返す
    """
    image_list = []
    files = os.listdir(dir_name)    # ディレクトリ内部のファイル一覧を取得
    print(files)

    for file in files:
        root, ext = os.path.splitext(file)  # 拡張子を取得
        if ext != ".jpg":
            break

        path = os.path.join(dir_name, file) # ディレクトリ名とファイル名を結合して、パスを作成
        # 画像を32x32pixelに変換し、1要素が[R,G,B]3要素を含む２次元配列として読み込む。
        # [R,G,B]はそれぞれが0-255の配列。
        image = np.array(Image.open(path).resize((32, 32)))
        if data_format == "channels_first":
            image = image.transpose(2, 0, 1)   # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
        image = image / 255.                   # 値を0-1に正規化
        image_list.append(image / 255.)        # 出来上がった配列をimage_listに追加  
    
    return image_list

# 画像を読み込む
img1 = openfile('1_train')
img2 = openfile('2_train')
x = np.array(img1 + img2)  # リストを結合
y = np.array([0] * len(img1) + [1] * len(img2))  # 正解ラベルを作成
y = np_utils.to_categorical(y)                   # ベルをone-hot-encoding形式に変換
print(x.shape)
print(y)
#exit()


# モデルの作成
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", data_format=data_format, input_shape=x.shape[1:]))  # カーネル数32, カーネルサイズ(3,3), input_shapeは1層目なので必要。https://keras.io/ja/layers/convolutional/#conv2d
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))                # 出力層のユニット数は2
model.add(Activation('sigmoid'))
model.add(Activation('softmax'))
opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 最適化器のセット。lrは学習係数
model.compile(optimizer=opt,       # コンパイル
      loss='categorical_crossentropy',
      metrics=['accuracy'])
print(model.summary())


# 学習
epochs = 100 # 1つのデータ当たりの学習回数
batch_size = 8              # 学習係数を更新するために使う教師データ数
history = model.fit(x, y, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=1, 
    validation_split=0.1,
    #validation_data=(x_test, y_test), # validation_dataをセットするとvalidation_splitは無視される
    shuffle=True,           # 学習毎にデータをシャッフルする
    ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）


def plot_history(history):
    """ 損失の履歴を図示する
    from http://www.procrasist.com/entry/2017/01/07/154441
    """
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"^-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.grid()
    plt.yscale("log") # ケースバイケースでコメントアウト
    plt.show()


# 学習結果を保存
print(model.summary())                    # レイヤー情報を表示(上で表示させると流れるので)
open("model", "w").write(model.to_json()) # モデル情報の保存
model.save_weights('param.hdf5')          # 獲得した結合係数を保存
plot_history(history)                     # lossの変化をグラフで表示


