#coding: utf-8
import tensorflow as tf
import random

##定義変数
#ネットワークのユニット数
INPUT_SIZE = 3
OUTPUT_SIZE = 1
#学習回数
epoch = 40000

###########
##-関数群-##
###########

#-ネットワーク関連-#

#@brief:順伝播計算を定義する
#@param:input_placeholder   入力データのプレースホルダー
#@return:y  順伝播計算の結果
def FunctionModel(input_placeholder):
    #モデルの定義を行う
    W1 = tf.Variable(tf.zeros([INPUT_SIZE, OUTPUT_SIZE]))#結合重み
    b1 = tf.Variable(tf.zeros([OUTPUT_SIZE]))#バイアス項
    y = tf.matmul(input_placeholder, W1) + b1#行列計算
    #順伝播の計算結果を返す
    return y


#@brief:誤差の計算を行う
#@param:label_placeholder   正解ラベルのプレースホルダー
#       output  順伝播の計算結果
#@return:loss   誤差の値
def loss(label_placeholder,output):
    #誤差関数の設定
    loss = tf.reduce_sum((tf.square(label_placeholder - output)))
    #誤差を返す
    return loss


#@brief:学習器の設定を行う
#@param:loss    誤差の値
#@return:train_step 学習器の設定
def training(loss):
    return tf.train.AdamOptimizer().minimize(loss)


#-データ関連-#

#@brief:訓練データを用意する関数
#@return:x_train    訓練データのinput
#        y_train    訓練データの正解
def prepare_TrainData():
    ####################
    # 訓練データ作成
    # 入力：３つの整数a,b,c
    # 出力：a+2b+3c+4
    ####################
    x_train = []
    y_train = []
    tmp = []
    # 訓練データの作成
    for i in range(1000):
        for j in range(3):
            tmp.append(random.randint(1, 1000))
        x_train.append(tmp)
        y_train.append([tmp[0] + 2 * tmp[1] + 3 * tmp[2] + 4])
        tmp = []
    #訓練データを返す
    return x_train,y_train


#############
##-メイン文-##
#############
if __name__ == "__main__":
    ##訓練データの準備
    #訓練データをロード
    x_train,y_train = prepare_TrainData()

    ##DNN関係
    #入力とラベルのプレースホルダーを設定する
    _x = tf.placeholder(tf.float32,[None,INPUT_SIZE])#任意のユニット数の入力層から3層への結合
    _y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
    #DNNの訓練
    output = FunctionModel(_x)#順伝播の計算
    loss = loss(_y,output)#誤差の計算
    train_step = training(loss)#誤差の逆伝播
    #定義した全ての変数を初期化
    init = tf.initialize_all_variables()

    #-セッションの開始-#
    with tf.Session() as sess:
        sess.run(init)#全ての変数を初期化
        #初期状態をprint
        print ("初期状態")
        print ('誤差' + str(sess.run(loss, feed_dict={_x: x_train, _y: y_train})))
        #
        for step in range(epoch):
            #100回周期で誤差を表示
            if step % 100 == 0:
                sess.run(train_step, feed_dict={_x: x_train, _y: y_train})
                print ('\nStep: %s' % (step))
                print ('誤差' + str(sess.run(loss, feed_dict={_x:x_train, _y:y_train})))
            else:
                sess.run(train_step, feed_dict={_x: x_train, _y: y_train})

        print ("予測結果")
        print (sess.run(output,feed_dict={_x:[[3,4,5]]}))#答えは30