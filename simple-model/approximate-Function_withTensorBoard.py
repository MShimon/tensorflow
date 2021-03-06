#coding: utf-8
import tensorflow as tf
import random


#-メイン文-#
if __name__ == "__main__":
    ##モデルの定義
    #入力層
    #任意のユニット数の入力層から3層への結合
    _x = tf.placeholder(tf.float32,[None,3])
    #ネットワークの重み
    W1 = tf.Variable(tf.zeros([3, 1]))
    #バイアス項
    b1 = tf.Variable(tf.zeros([1]))
    #行列の計算
    y = tf.matmul(_x, W1) + b1

    # TensorBoardへ表示するための変数を用意する（ヒストグラム用）
    w_hist = tf.summary.histogram("weights", W1)
    b_hist = tf.summary.histogram("biases", b1)
    y_hist = tf.summary.histogram("y", y)

    ##誤差関数及び学習器の設定
    #正解の値を格納するplaceholderを作成
    _y = tf.placeholder(tf.float32,[None,1])
    #誤差関数の設定
    loss = tf.reduce_sum((tf.square(_y - y)))

    # TensorBoardへloss（学習のダメ具合の指標として設定したスカラー値）を表示するための変数を用意する（イベント用）
    loss_summary = tf.summary.scalar("loss", loss)

    #学習器及び学習率の設定
    train_step = tf.train.AdamOptimizer().minimize(loss)
    #定義した全ての変数を初期化
    init = tf.initialize_all_variables()


    ####################
    # 訓練データ作成
    # 入力：３つの整数a,b,c
    # 出力：a+2b+3c+4
    ####################
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    tmp = []
    # 訓練データの作成
    for i in range(1000):
        for j in range(3):
            tmp.append(random.randint(1, 1000))
        x_train.append(tmp)
        y_train.append([tmp[0] + 2 * tmp[1] + 3 * tmp[2] + 4])
        tmp = []

    #-セッションの開始-#
    with tf.Session() as sess:
        # 上記で用意した合計４つのTensorBoard描画用の変数を、TensorBoardが利用するSummaryデータとしてmerge（合体）する
        # また、そのSummaryデータを書き込むSummaryWriterを用意し、書き込み先を'/tmp/tf_logs'ディレクトリに指定する
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/tensorflow_log", sess.graph_def)
        sess.run(init)
        print ("初期状態")
        print ('誤差' + str(sess.run(loss, feed_dict={_x: x_train, _y: y_train})))

        for step in range(40000):
            #100回周期で誤差を表示
            if step % 100 == 0:
                sess.run(train_step, feed_dict={_x: x_train, _y: y_train})
                print ('\nStep: %s' % (step))
                print ('誤差' + str(sess.run(loss, feed_dict={_x:x_train, _y:y_train})))
                result = sess.run([merged, loss],feed_dict={_x:x_train, _y:y_train})
                summary_str = result[0]
                acc = result[1]
                writer.add_summary(summary_str, step)
            else:
                sess.run(train_step, feed_dict={_x: x_train, _y: y_train})

        print ("予測結果")
        print (sess.run(y,feed_dict={_x:[[3,4,5]]}))#答えは30