#ニューラルネットワーク(分類)

import numpy as np
import matplotlib.pyplot as plt

#x,y座標
X = np.arange(-1.0,1.0,0.2)
Y = np.arange(-1.0,1.0,0.2)

#出力を格納する10*10のグリッド
Z = np.zeros((10,10))

#重み
w_im = np.array([[4.0,4.0],[4.0,4.0]])
w_mo = np.array([[1.0],[-1.0]])

#バイアス
b_im = np.array([3.0,-3.0])
b_mo = np.array([0.1])

#中間層
def middle_layer(x,w,b):
    u = np.dot(x,w) + b
    #シグモイド関数
    return 1/(1+np.exp(-u))

#出力層
def output_layer(x,w,b):
    u = np.dot(x,w) + b
    #恒等関数
    return u

#グリッドの各マスでニューラルネットワークの演算
for i in range(10):
    for j in range(10):

        #順伝播
        #入力層
        inp = np.array([X[i],Y[j]])
        #中間層
        mid = middle_layer(inp, w_im, b_im)
        #出力層
        out = output_layer(mid, w_mo, b_mo)

        #グリッドのNNの出力を格納
        Z[j][i] = out[0]

plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
plt.colorbar()
plt.show()
