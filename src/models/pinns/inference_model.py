# import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import tensorflow as tf

from src.utils import DenseBlock


class MyoPINN(tf.keras.Model):
    def __init__(
        self,
        N,
        layers,
        layer_width,
        lr,
        loss_weights=(1, 1, 0, 0, 0, 0),
        bn=False,
        log_domain=False,
        trainable_pars="all",
        shape_in=(1,),
        std_t=1,
    ):
        """
        Creates the standard ODE-supervised DNN
        :param N: ndarray, mask of the CE-MRI (in form ndarray:(size_x, size_y))
        :param layers: int, number of dense layer repetitions in the NN
        :param layer_width: int, number of neurons per dense layer in the NN
        :param lr: float, the initial learning rate for the optimization process
        :param loss_weights: tuple of weights for (loss_u, loss_r, loss_e, loss_b, loss_reg, loss_var)
        :param bn: bool, the use of batch normalization after a dense layer
        :param log_domain: bool, the optimization of the ODE parameters in log domain
        :param trainable_pars: string denoting the trainable parameters, options are: "all", "ODE" and "NN"
        :param shape_in: shape of input to the NN (useful if wanting to pass both spatial and time coordinates)
        :param std_t: standard deviation of the normalized 'time' input
        """
        # initialize
        super(MyoPINN, self).__init__()
        self.lw_u, self.lw_r, self.lw_e, self.lw_b, self.lw_reg, self.lw_var = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        self.optimizer = None
        self.var_list = None
        self.N = N
        self.mr_shape = N.shape
        self.indices = np.count_nonzero(N)

        if len(shape_in) == 1:
            # one-to-many input
            self.neurons_out = [
                1,
                self.mr_shape[0] * self.mr_shape[1] * shape_in[0][0],
                self.mr_shape[0] * self.mr_shape[1] * shape_in[0][0],
            ]
        else:
            # mesh input
            self.neurons_out = [1, 1, 1]

        self.std_t = std_t

        # initialize kinetic parameters
        low = tf.math.log(0.05) if log_domain else 0.05
        high = tf.math.log(0.15) if log_domain else 0.15
        self.f = tf.Variable(
            tf.random.uniform(
                [self.mr_shape[0] * self.mr_shape[1], 1], minval=low, maxval=high
            ),
            name="v_f",
        )
        self.vp = tf.Variable(
            tf.random.uniform(
                [self.mr_shape[0] * self.mr_shape[1], 1], minval=low, maxval=high
            ),
            name="v_vp",
        )
        self.ve = tf.Variable(
            tf.random.uniform(
                [self.mr_shape[0] * self.mr_shape[1], 1], minval=low, maxval=high
            ),
            name="v_ve",
        )
        self.ps = tf.Variable(
            tf.random.uniform(
                [self.mr_shape[0] * self.mr_shape[1], 1], minval=low, maxval=high
            ),
            name="v_ps",
        )

        self.epochs = 0
        self.log_domain = log_domain
        self.NN = DenseBlock(
            layers,
            layer_width,
            shape_in=shape_in,
            num_out=3,
            neurons_out=self.neurons_out,
            bn=bn,
        )
        self.set_lr(lr)
        self.set_loss_weights(loss_weights)
        self.set_trainable_pars(trainable_pars)

        # tensorboard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.string = "{}_N{}_lr{}_layers{}_neurons{}_bn{}_log{}_weights{}".format(
            current_time,
            self.indices,
            lr,
            layers,
            layer_width,
            bn,
            log_domain,
            loss_weights,
        )
        self.train_summary_writer = tf.summary.create_file_writer(
            "logs/" + self.string
        )

    @tf.function
    def call(self, t):
        # get ODE parameters
        pars = self.get_ode_pars()

        # NN
        [caif, ce, cp] = self.NN(t)
        cmyo = ce * pars[..., 2] + cp * pars[..., 1]

        # ODE
        cmyo_t = (1 / self.std_t) * self.__fwd_gradients(cmyo, t)
        f = cmyo_t - pars[..., 0] * (caif - cp)

        return caif, cmyo, ce, cp, cmyo_t, f

    @tf.function
    def call_NN(self, t):
        # get ODE parameters
        pars = self.get_ode_pars()

        # NN
        [caif, ce, cp] = self.NN(t)
        cmyo = ce * pars[..., 2] + cp * pars[..., 1]

        return caif, cmyo, ce, cp

    def fit(
        self,
        data_u,
        data_r,
        data_e,
        data_b,
        true_pars,
        batch_size,
        epochs,
        verbose=False,
    ):
        t0 = time.time()
        for ep in range(self.epochs + 1, self.epochs + epochs + 1):
            iter_u = iter(data_u.batch(batch_size))
            iter_r = iter(data_r.shuffle(len(list(data_r))).batch(batch_size))
            iter_e = iter(data_e.shuffle(len(list(data_e))).batch(batch_size))
            for _ in range(int(np.ceil(len(list(data_u)) / batch_size))):
                batch_u = next(iter_u)
                batch_r = next(iter_r)
                batch_e = next(iter_e)
                batch_b = next(iter(data_b.batch(1)))
                self.optimize(batch_u, batch_r, batch_e, batch_b)

            # track
            if ep % 100 == 0 or ep == 1:
                u = next(iter(data_u.batch(len(list(data_u)))))
                r = next(iter(data_r.batch(len(list(data_u)))))
                e = next(iter(data_e.batch(len(list(data_u)))))
                b = next(iter(data_b.batch(1)))
                loss_u = self.__loss_u(u[1:], self(u[0])).numpy().item()
                loss_r = self.__loss_r(self(r)).numpy().item()
                loss_e = self.__loss_e(self(e)).numpy().item()
                loss_b = self.__loss_b(self.call_NN(b)).numpy().item()
                loss_reg = self.__loss_reg(self.call_NN(r)).numpy().item()
                if self.lw_var:
                    loss_reg += self.__loss_var(b[1], self.get_ode_pars())
                pred = self.predict()

                self.track_statement(
                    loss_u,
                    loss_r,
                    loss_e,
                    loss_b,
                    loss_reg,
                    pred,
                    true_pars,
                    ep,
                    verbose,
                )

                if False in tf.math.is_nan(self.f):
                    self.save()
                else:
                    print("NaN values detected, training aborted")
                    break

        t1 = time.time()
        if verbose:
            print("\ttraining took {0:.2f} minutes".format((t1 - t0) / 60))
        self.epochs += epochs

    def get_error(self, true):
        pred = self.predict()
        true = tf.convert_to_tensor(true)

        mse = tf.reduce_mean(tf.square(pred - true), axis=(0, 1, 2))
        return tf.reduce_mean(mse / tf.reduce_mean(true, axis=(0, 1, 2)))

    def get_loss(self, u, r, e, b):
        loss = tf.constant(0, dtype=tf.float32)
        if self.lw_u:
            loss += self.lw_u * self.__loss_u(u[1:], self(u[0]))
        if self.lw_r:
            loss += self.lw_r * self.__loss_r(self(r))
        if self.lw_e:
            loss += self.lw_e * self.__loss_e(self(e))
        if self.lw_b:
            loss += self.lw_b * self.__loss_b(self.call_NN(b))
        if self.lw_reg:
            loss += self.lw_reg * self.__loss_reg(self.call_NN(r))
        if self.lw_var:
            loss += self.lw_var * self.__loss_var(b[1], self.get_ode_pars())
        return loss

    def get_ode_pars(self):
        var_tensor = tf.concat([self.f, self.vp, self.ve, self.ps], axis=-1,)
        var_tensor = tf.exp(var_tensor) if self.log_domain else var_tensor
        return tf.multiply(var_tensor, tf.constant([10.0, 1.0, 1.0, 10.0]))

    def optimize(self, u, r, e, b):
        def loss():
            return self.get_loss(u, r, e, b)

        self.optimizer.minimize(loss=loss, var_list=self.var_list)

    def predict(self):
        var_tensor = tf.concat([self.f, self.vp, self.ve, self.ps], axis=-1,)
        var_tensor = tf.reshape(var_tensor, self.mr_shape + (1, 4,))
        var_tensor = tf.exp(var_tensor) if self.log_domain else var_tensor
        return tf.multiply(var_tensor, tf.constant([10.0, 1.0, 1.0, 10.0]))

    def save(self):
        # weights to .tf
        self.save_weights("model_weights/{}.tf".format(self.string))

    def load(self, string):
        self.load_weights("model_weights/{}".format(string))

    def set_lr(self, lr):
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

    def set_loss_weights(self, loss_weights):
        (
            self.lw_u,
            self.lw_r,
            self.lw_e,
            self.lw_b,
            self.lw_reg,
            self.lw_var,
        ) = loss_weights

    def set_trainable_pars(self, trainable_pars):
        if trainable_pars == "all":
            self.var_list = self.trainable_variables
        elif trainable_pars == "ODE":
            self.var_list = [
                var for var in self.trainable_variables if "v_" in var.name
            ]
        elif trainable_pars == "NN":
            self.var_list = [
                var for var in self.trainable_variables if "v_" not in var.name
            ]
        elif trainable_pars == "transfer":
            self.var_list = [
                var
                for var in self.trainable_variables
                if "out" in var.name or "v_" in var.name
            ]
        else:
            raise Exception(
                "string 'trainable pars' must be one of ('all', 'ODE', 'NN', 'transfer')"
            )

    def track_statement(
        self, loss_u, loss_r, loss_e, loss_b, loss_reg, pred, true_pars, ep, verbose
    ):
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                "training/loss", loss_u + loss_r + loss_e + loss_b + loss_reg, step=ep
            )
            tf.summary.scalar("training/loss_u", loss_u, step=ep)
            tf.summary.scalar("training/loss_r", loss_r, step=ep)
            tf.summary.scalar("training/loss_e", loss_e, step=ep)
            tf.summary.scalar("training/loss_b", loss_b, step=ep)
            tf.summary.scalar("training/loss_reg", loss_reg, step=ep)

            for i, param in enumerate(("f", "vp", "ve", "ps")):
                tf.summary.scalar(
                    "vars/{}".format(param),
                    (
                        tf.reduce_sum(pred[..., 0, i] * self.N)
                        / tf.math.count_nonzero(self.N, dtype=tf.float32)
                    )
                    .numpy()
                    .item(),
                    step=ep,
                )
        if true_pars is not None:
            error = self.get_error(true_pars)
            with self.train_summary_writer.as_default():
                tf.summary.scalar("training/error", error.numpy().item(), step=ep)
                for i, param in enumerate(("f", "vp", "ve", "ps")):
                    tf.summary.scalar(
                        "vars_mse/{}".format(param),
                        tf.reduce_mean(tf.square(pred[..., i] - true_pars[..., i]))
                        / tf.reduce_mean(true_pars[..., i]).numpy().item(),
                        step=ep,
                    )
        if verbose:
            print(
                "Epoch: {:5d}, Loss: {:0.5f}".format(
                    ep, loss_u + loss_r + loss_e + loss_b + loss_reg
                )
            )
            tf.print(
                "f: ",
                pred[..., 0],
                "vp: ",
                pred[..., 1],
                "ve: ",
                pred[..., 2],
                "ps: ",
                pred[..., 3],
            )

    @staticmethod
    def __fwd_gradients(y, x):
        dummy = tf.ones_like(y)
        g = tf.gradients(y, x, grad_ys=dummy)[0]
        return tf.gradients(g, dummy)[0]

    @staticmethod
    @tf.function
    def __loss_u(y_true, y_pred):
        # solution loss
        loss_aif = tf.reduce_mean(tf.square(y_pred[0] - y_true[0]))
        loss_myo = tf.reduce_mean(tf.square(y_pred[1] - y_true[1]))
        loss_myo_t = tf.reduce_mean(tf.square(y_pred[4] - y_true[2]))
        return loss_aif + loss_myo + loss_myo_t

    @staticmethod
    @tf.function
    def __loss_r(y_pred):
        # residual loss
        loss_r = tf.reduce_mean(tf.square(y_pred[5]))
        return loss_r

    @staticmethod
    @tf.function
    def __loss_b(y_pred):
        # boundary loss
        loss_aif = tf.reduce_mean(tf.square(y_pred[0]))
        loss_myo = tf.reduce_mean(tf.square(y_pred[1]))
        loss_ce = tf.reduce_mean(tf.square(y_pred[2]))
        loss_cp = tf.reduce_mean(tf.square(y_pred[3]))
        return loss_aif + loss_myo + loss_ce + loss_cp

    @staticmethod
    @tf.function
    def __loss_e(y_pred):
        # extrapolation loss
        loss_ce = tf.reduce_mean(tf.square(tf.minimum(y_pred[2], 0.0)))
        loss_cp = tf.reduce_mean(tf.square(tf.minimum(y_pred[3], 0.0)))
        loss_myo_t = tf.reduce_mean(tf.square(tf.maximum(y_pred[4], 0.0)))
        loss_r = tf.reduce_mean(tf.square(y_pred[5]))
        return loss_ce + loss_cp + loss_myo_t + loss_r

    @staticmethod
    @tf.function
    def __loss_reg(y_pred):
        # regularization loss
        loss_ce = tf.reduce_mean(tf.square(tf.minimum(y_pred[2], 0.0)))
        loss_cp = tf.reduce_mean(tf.square(tf.minimum(y_pred[3], 0.0)))
        return loss_ce + loss_cp

    @staticmethod
    @tf.function
    def __loss_var(y_true, y_pred):
        # AUC loss
        loss_var = tf.reduce_mean(
            tf.square((y_pred[..., 1] + y_pred[..., 2]) - y_true[0])
        )
        return loss_var


class FullPINN(MyoPINN):
    def __init__(self, *args, **kwargs):
        # initialize
        super(FullPINN, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, t):
        # get ODE parameters
        pars = self.get_ode_pars()

        # NN
        [caif, ce, cp] = self.NN(t)
        cmyo = ce * pars[..., 2] + cp * pars[..., 1]

        # ODE's
        ce_t = (1 / self.std_t) * self._MyoPINN__fwd_gradients(ce, t)
        cp_t = (1 / self.std_t) * self._MyoPINN__fwd_gradients(cp, t)
        cmyo_t = ce_t * pars[..., 2] + cp_t * pars[..., 1]

        f_ce = pars[..., 2] * ce_t - pars[..., 3] * (cp - ce)
        f_cp = (
            pars[..., 1] * cp_t - pars[..., 3] * (ce - cp) - pars[..., 0] * (caif - cp)
        )

        return caif, cmyo, ce, cp, cmyo_t, f_ce, f_cp

    @staticmethod
    @tf.function
    def _MyoPINN__loss_r(y_pred):
        # residual loss
        loss_re = tf.reduce_mean(tf.square(y_pred[5]))
        loss_rp = tf.reduce_mean(tf.square(y_pred[6]))
        return loss_re + loss_rp

    @staticmethod
    @tf.function
    def _MyoPINN__loss_e(y_pred):
        # extrapolation loss
        loss_ce = tf.reduce_mean(tf.square(tf.minimum(y_pred[2], 0.0)))
        loss_cp = tf.reduce_mean(tf.square(tf.minimum(y_pred[3], 0.0)))
        loss_myo_t = tf.reduce_mean(tf.square(tf.maximum(y_pred[4], 0.0)))
        loss_re = tf.reduce_mean(tf.square(y_pred[5]))
        loss_rp = tf.reduce_mean(tf.square(y_pred[6]))
        return loss_ce + loss_cp + loss_myo_t + loss_re + loss_rp


class CombinedPINN(MyoPINN):
    def __init__(self, *args, **kwargs):
        # initialize
        super(CombinedPINN, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, t):
        # get ODE parameters
        pars = self.get_ode_pars()

        # NN
        [caif, ce, cp] = self.NN(t)
        cmyo = ce * pars[..., 2] + cp * pars[..., 1]

        # ODE's
        ce_t = (1 / self.std_t) * self._MyoPINN__fwd_gradients(ce, t)
        cp_t = (1 / self.std_t) * self._MyoPINN__fwd_gradients(cp, t)
        cmyo_t = ce_t * pars[..., 2] + cp_t * pars[..., 1]

        f_cmyo = cmyo_t - pars[..., 0] * (caif - cp)
        f_ce = pars[..., 2] * ce_t - pars[..., 3] * (cp - ce)
        f_cp = (
            pars[..., 1] * cp_t - pars[..., 3] * (ce - cp) - pars[..., 0] * (caif - cp)
        )

        return caif, cmyo, ce, cp, cmyo_t, f_cmyo, f_ce, f_cp

    @staticmethod
    @tf.function
    def _MyoPINN__loss_r(y_pred):
        # residual loss
        loss_rmyo = tf.reduce_mean(tf.square(y_pred[5]))
        loss_re = tf.reduce_mean(tf.square(y_pred[6]))
        loss_rp = tf.reduce_mean(tf.square(y_pred[7]))
        return loss_rmyo + loss_re + loss_rp

    @staticmethod
    @tf.function
    def _MyoPINN__loss_e(y_pred):
        # extrapolation loss
        loss_ce = tf.reduce_mean(tf.square(tf.minimum(y_pred[2], 0.0)))
        loss_cp = tf.reduce_mean(tf.square(tf.minimum(y_pred[3], 0.0)))
        loss_myo_t = tf.reduce_mean(tf.square(tf.maximum(y_pred[4], 0.0)))
        loss_rmyo = tf.reduce_mean(tf.square(y_pred[5]))
        loss_re = tf.reduce_mean(tf.square(y_pred[6]))
        loss_rp = tf.reduce_mean(tf.square(y_pred[7]))
        return loss_ce + loss_cp + loss_myo_t + loss_rmyo + loss_re + loss_rp


class MeshPINN(MyoPINN):
    def __init__(self, *args, **kwargs):
        # initialize
        super(MeshPINN, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, xy, t):
        # get ODE parameters
        pars = self.get_ode_pars()

        # NN
        [caif, ce, cp] = self.NN([xy, t])
        cmyo = ce * pars[..., 2] + cp * pars[..., 1]

        # ODE
        cmyo_t = (1 / self.std_t) * self._MyoPINN__fwd_gradients(cmyo, t)
        f = cmyo_t - pars[..., 0] * (caif - cp)

        return caif, cmyo, ce, cp, cmyo_t, f

    @tf.function
    def call_NN(self, xy, t):
        # get ODE parameters
        pars = self.get_ode_pars()

        # NN
        [caif, ce, cp] = self.NN([xy, t])
        cmyo = ce * pars[..., 2] + cp * pars[..., 1]

        return caif, cmyo, ce, cp

    def fit(
        self,
        data_u,
        data_r,
        data_e,
        data_b,
        true_pars,
        batch_size,
        epochs,
        verbose=False,
    ):
        data_size_u = len(list(data_u))
        data_size_r = len(list(data_r))
        data_size_e = len(list(data_e))
        data_size_b = len(list(data_b))

        t0 = time.time()
        for ep in range(self.epochs + 1, self.epochs + epochs + 1):
            iter_u = iter(data_u.batch(batch_size))
            iter_r = iter(data_r.shuffle(data_size_r).batch(batch_size))
            iter_e = iter(data_e.shuffle(data_size_e).batch(batch_size))
            for i in range(int(np.ceil(data_size_u / batch_size))):
                batch_u = next(iter_u)
                batch_r = next(iter_r)
                batch_e = next(iter_e)
                batch_b = next(iter(data_b.batch(data_size_b)))
                self.optimize(batch_u, batch_r, batch_e, batch_b)

            # track
            if ep % 10 == 0 or ep == 1:
                u = next(iter(data_u.batch(batch_size)))
                r = next(iter(data_r.batch(batch_size)))
                e = next(iter(data_r.batch(batch_size)))
                b = next(iter(data_b.batch(data_size_b)))
                loss_u = self._MyoPINN__loss_u(u[2:], self(u[0], u[1])).numpy().item()
                loss_r = self._MyoPINN__loss_r(self(r[0], r[1])).numpy().item()
                loss_e = self._MyoPINN__loss_e(self(e[0], e[1])).numpy().item()
                loss_b = (
                    self.lw_b
                    * self._MyoPINN__loss_b(self.call_NN(b[0], b[1])).numpy().item()
                )
                loss_reg = (
                    self._MyoPINN__loss_reg(self.call_NN(r[0], r[1])).numpy().item()
                )
                if self.lw_var:
                    loss_reg += self.__loss_var(b[2], self.get_ode_pars())
                pred = self.get_ode_pars()

                self.track_statement(
                    loss_u,
                    loss_r,
                    loss_e,
                    loss_b,
                    loss_reg,
                    pred,
                    true_pars,
                    ep,
                    verbose,
                )

        t1 = time.time()
        if verbose:
            print("\ttraining took {0:.2f} minutes".format((t1 - t0) / 60))
        self.epochs += epochs
        self.save()

    def get_loss(self, u, r, e, b):
        loss = tf.constant(0, dtype=tf.float32)
        if self.lw_u:
            loss += self.lw_u * self._MyoPINN__loss_u(u[2:], self(u[0], u[1]))
        if self.lw_r:
            loss += self.lw_r * self._MyoPINN__loss_r(self(r[0], r[1]))
        if self.lw_e:
            loss += self.lw_e * self._MyoPINN__loss_e(self(r[0], r[1]))
        if self.lw_b:
            loss += self.lw_b * self._MyoPINN__loss_b(self.call_NN(b[0], b[1]))
        if self.lw_reg:
            loss += self.lw_reg * self._MyoPINN__loss_reg(self.call_NN(r[0], r[1]))
        if self.lw_var:
            loss += self.lw_var * self.__loss_var(b[2], self.get_ode_pars())
        return loss

    def get_error(self, true):
        pred = self.get_ode_pars()
        true = tf.convert_to_tensor(true)

        mse = tf.reduce_mean(tf.square(pred - true), axis=(0, 1, 2))
        return tf.reduce_mean(mse / tf.reduce_mean(true, axis=(0, 1, 2)))

    def get_ode_pars(self):
        var_tensor = tf.concat([self.f, self.vp, self.ve, self.ps], axis=-1,)
        var_tensor = tf.reshape(var_tensor, self.mr_shape + (1, 4,))
        var_tensor = tf.exp(var_tensor) if self.log_domain else var_tensor
        return tf.multiply(var_tensor, tf.constant([10.0, 1.0, 1.0, 10.0]))


class FullMeshPINN(MeshPINN):
    def __init__(self, *args, **kwargs):
        # initialize
        super(FullMeshPINN, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, xy, t):
        # get ODE parameters
        pars = self.get_ode_pars()

        # NN
        [caif, ce, cp] = self.NN([xy, t])
        cmyo = ce * pars[..., 2] + cp * pars[..., 1]

        # ODE's
        ce_t = (1 / self.std_t) * self._MyoPINN__fwd_gradients(ce, t)
        cp_t = (1 / self.std_t) * self._MyoPINN__fwd_gradients(cp, t)
        cmyo_t = ce_t * pars[..., 2] + cp_t * pars[..., 1]

        f_ce = pars[..., 2] * ce_t - pars[..., 3] * (cp - ce)
        f_cp = (
            pars[..., 1] * cp_t - pars[..., 3] * (ce - cp) - pars[..., 0] * (caif - cp)
        )

        return caif, cmyo, ce, cp, cmyo_t, f_ce, f_cp

    @staticmethod
    @tf.function
    def _MyoPINN__loss_r(y_pred):
        # residual loss
        loss_re = tf.reduce_mean(tf.square(y_pred[5]))
        loss_rp = tf.reduce_mean(tf.square(y_pred[6]))
        return loss_re + loss_rp

    @staticmethod
    @tf.function
    def _MyoPINN__loss_e(y_pred):
        # extrapolation loss
        loss_ce = tf.reduce_mean(tf.square(tf.minimum(y_pred[2], 0.0)))
        loss_cp = tf.reduce_mean(tf.square(tf.minimum(y_pred[3], 0.0)))
        loss_myo_t = tf.reduce_mean(tf.square(tf.maximum(y_pred[4], 0.0)))
        loss_re = tf.reduce_mean(tf.square(y_pred[5]))
        loss_rp = tf.reduce_mean(tf.square(y_pred[6]))
        return loss_ce + loss_cp + loss_myo_t + loss_re + loss_rp