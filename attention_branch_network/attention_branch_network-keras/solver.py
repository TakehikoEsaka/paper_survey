import pickle
import numpy as np
from os.path import join

# CategoricalCrossEntropyやBinaryCrossEntropyをとれる．
# model.compile()時に使う
from tensorflow.keras import losses

# SGD() / RMSprop / Adagrad / Adadelta / Adam() / Adamax / Nadam / TFOptimizerが選択可能
# 読み物：https://qiita.com/ZoneTsuyoshi/items/8ef6fa1e154d176e25b8
# model.compile()時に使う
from tensorflow.keras import optimizers

# 評価関数を指定する事が出来る．
# 自作の評価関数を指定することも可能．
from tensorflow.keras import metrics

"""
【コールバック関数】

訓練中の様子のモデル内部状態・統計量を可視化する為に使う．
アーリーストッピング：es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
テンソルボード用：tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)

チェックポイント：cp_cb = keras.callbacks.ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
※  チェックポイントファイルがかぶるといけないので，以下のようにファイルパスを変更可能
※ (続き) fpath = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'

ラーニングレートを変更する：lr_cb = keras.callbacks.LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))
※ パラメータ設定例：start = 0.03, stop = 0.001, nb_epoch = 1000, learning_rates = np.linspace(start, stop, nb_epoch)
呼び出したコールバックは，model.fit()時に，callbackの引数として[es_cb, tb_cb]のように渡す．

# 評価値の改善が止まった時に学習率を小さくする：keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
"""

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# 計算を早くする為にfloat16で計算する.
# kerasのmixed_precicion.policyで設定可能．
from tensorflow.keras.mixed_precision import experimental as mixed_precicion

from utils.modules import get_model


class Solver(object):
    def __init__(self, args, data_loader, valid_loader=None):
        self.data_loader = data_loader
        self.valid_loader = valid_loader
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.mixed_training = args.mixed_training
        self.n_epochs = args.n_epochs
        self.save_dir = args.save_dir

        if args.mixed_training:
            # 計算を早くする為にfloat16で計算する.
            # kerasのmixed_precicion.policyで設定可能．
            policy = mixed_precicion.Policy('mixed_float16')
            mixed_precicion.set_policy(policy)

        self.n_classes = len(np.unique(data_loader.y_train))

        """
        cifar10からMNISTに変更する為，以下の用にmodelのgetの仕方を変更した．
        """
        #self.model = get_model((None, None, 3), self.n_classes)
        w = data_loader.x_test.shape[1]
        h = data_loader.x_test.shape[2]
        self.model = get_model((w, h, 1), 10)
        print("model input : " + str(self.model.input))
        print("model output : " + str(self.model.output))
        
        self.model.compile(
            loss=[losses.SparseCategoricalCrossentropy(),
                  losses.SparseCategoricalCrossentropy()],
            optimizer=optimizers.Adam(lr=args.lr),
            metrics = ['acc']
        )

    def train(self):
        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                filepath=join(self.save_dir, "w.hdf5"),
                monitor="val_perception_branch_output_acc",
                save_best_only=True,
                mode="max",
                verbose=1
            )
        )
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_perception_branch_output_acc",
                factor=.5,
                patience=2,
                min_lr=1e-6,
                mode='max',
                verbose=1
            )
        )

        history = self.model.fit_generator(
            generator=self.data_loader,
            validation_data=self.valid_loader,
            epochs=self.n_epochs,
            steps_per_epoch=len(self.data_loader),
            validation_steps=len(self.valid_loader),
            verbose=1,
            workers=10,
            max_queue_size=30,
            callbacks=callbacks
        )

        with open(join(self.save_dir, "train_log.pkl"), "wb") as f:
            pickle.dump(history.history, f)

