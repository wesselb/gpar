import numpy as np
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.utils import np_utils


def build_model(params):
    n_hidden_layers = int(np.round(params["n_hidden_layers"]))
    n_neurons = int(np.round(params["n_neurons"]))
    log_l1_weight_reg = params["log_l1_weight_reg"]
    log_l2_weight_reg = params["log_l2_weight_reg"]
    prob_drop_out = float(params["prob_drop_out"])
    log_l_rate = params["log_learning_rate"]

    model = Sequential()
    model.add(
        Dense(
            n_neurons,
            input_shape=(784,),
            W_regularizer=l1_l2(
                l1=np.exp(log_l1_weight_reg), l2=np.exp(log_l2_weight_reg)
            ),
        )
    )
    model.add(Activation("relu"))
    model.add(Dropout(prob_drop_out))
    for i in range(n_hidden_layers - 1):
        model.add(
            Dense(
                n_neurons,
                W_regularizer=l1_l2(
                    l1=np.exp(log_l1_weight_reg), l2=np.exp(log_l2_weight_reg)
                ),
            )
        )
        model.add(Activation("relu"))
        model.add(Dropout(prob_drop_out))
    n_classes = 10
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    adam = Adam(lr=np.exp(log_l_rate), beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy", optimizer=adam)

    return model


def fit_model(x_train, y_train, x_test, y_test, x_val, y_val, params):
    nb_epoch = 150
    batch_size = 4000
    model = build_model(params)
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=2,
        validation_data=(x_val, y_val),
    )

    return history


def get_results(params):
    nb_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    state = np.random.get_state()
    np.random.seed(0)
    perm = np.random.permutation(60000)
    i_train = perm[0:50000]
    i_val = perm[50000:60000]
    np.random.set_state(state)

    x_val = x_train[i_val, :]
    y_val = y_train[i_val]
    x_train = x_train[i_train, :]
    y_train = y_train[i_train]

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    y_val = np_utils.to_categorical(y_val, nb_classes)

    history = fit_model(x_train, y_train, x_test, y_test, x_val, y_val, params)

    results = {
        "params": params,
        "val_loss": history.history["val_loss"],
        "train_loss": history.history["loss"],
        "epochs": history.epoch,
    }

    return results


if __name__ == "__main__":
    params = {
        "n_hidden_layers": 2,
        "n_neurons": 100,
        "log_l1_weight_reg": -10,
        "log_l2_weight_reg": -10,
        "prob_drop_out": 0.2,
        "log_learning_rate": -10,
    }
    get_results(params=params)
