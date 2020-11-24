# import necessary modules
import numpy as np
import tensorflow as tf

def gamma_func(K, alpha, beta, time):

    gamma = (K * time ** alpha) * np.exp(-time / beta)

    return gamma

def phys_params_to_tt_params(f, vp, ve, perm_surf):

    tb = vp / f
    tp = vp / (perm_surf + f)
    te = ve / perm_surf

    return tb, tp, te


def phys_params_to_split_model_params(parameters):
    [flow, vol_p, vol_e, PS] = parameters

    alpha = 0.5 * (
        -(flow / vol_p + PS / vol_p + PS / vol_e)
        + tf.sqrt(
            (flow / vol_p + PS / vol_p + PS / vol_e) ** 2
            - 4 * (PS / vol_e) * flow / vol_p
        )
    )

    beta = 0.5 * (
        -(flow / vol_p + PS / vol_p + PS / vol_e)
        - tf.sqrt(
            (flow / vol_p + PS / vol_p + PS / vol_e) ** 2
            - 4 * (PS / vol_e) * flow / vol_p
        )
    )

    Ap = (alpha + PS / vol_e) / (alpha - beta)

    Ae = (PS / vol_p) / (alpha - beta)

    return alpha, beta, Ap, Ae


def tt_params_to_model_params(tb, tp, te):

    k_m = 0.5 * (
        1.0 / tp
        + 1.0 / te
        - tf.sqrt((1.0 / tp + 1.0 / te) ** 2 - 4.0 * (1.0 / tb) * (1.0 / te))
    )
    k_p = 0.5 * (
        1.0 / tp
        + 1.0 / te
        + tf.sqrt((1.0 / tp + 1.0 / te) ** 2 - 4.0 * (1.0 / tb) * (1.0 / te))
    )
    e = (k_p - 1.0 / tb) / (k_p - k_m)
    d = k_p - k_m

    return d, e, k_m


def two_comp_ex_model(parameters, time, AIF):

    [flow, vol_p, vol_e, PS] = parameters

    [Tb, Tp, Te] = phys_params_to_tt_params(flow, vol_p, vol_e, PS)

    [delta, E, km] = tt_params_to_model_params(Tb, Tp, Te)

    return (
        (time[1] - time[0])
        * flow
        * (
            E * np.convolve(np.exp(-km * time), AIF)
            + (1 - E) * np.convolve(np.exp(-((km + delta) * time)), AIF)
        )[: len(AIF)]
    )


def two_comp_ex_model_tf(parameters, time, AIF):

    [flow, vol_p, vol_e, PS] = tf.split(parameters, 4, axis=1)

    [Tb, Tp, Te] = phys_params_to_tt_params(flow, vol_p, vol_e, PS)

    [delta, E, km] = tt_params_to_model_params(Tb, Tp, Te)

    padding_data = tf.constant([[time.shape[0] - 1, 0], [0, 0]])
    kernel = AIF[::-1]
    kernel = tf.reshape(kernel, [AIF.shape[0], 1, 1])

    Alpha = tf.expand_dims(
        tf.transpose(
            tf.pad(tf.exp(tf.transpose(-km, [1, 0]) * time), padding_data, "CONSTANT"),
            [1, 0],
        ),
        -1,
    )
    Beta = tf.expand_dims(
        tf.transpose(
            tf.pad(
                tf.exp(-(tf.transpose((km + delta), [1, 0]) * time)),
                padding_data,
                "CONSTANT",
            ),
            [1, 0],
        ),
        -1,
    )

    Alpha = tf.nn.conv1d(Alpha, kernel, stride=1, padding="VALID")
    Beta = tf.nn.conv1d(Beta, kernel, stride=1, padding="VALID")

    E = tf.expand_dims(E, -1)
    flow = tf.expand_dims(flow, -1)

    a = tf.transpose(
        ((time[1] - time[0]) * flow * (E * Alpha + (1 - E) * Beta))[..., 0], [1, 0]
    )
    return tf.transpose(
        ((time[1] - time[0]) * flow * (E * Alpha + (1 - E) * Beta))[..., 0], [1, 0]
    )


def split_two_comp_ex_model(parameters, time, AIF):
    [alpha, beta, Ap, Ae] = phys_params_to_split_model_params(parameters)

    flow = parameters[0]

    Cp = (
        (time[1] - time[0])
        * flow
        * (
            Ap * np.convolve(np.exp(alpha * time), AIF)
            + (1 - Ap) * np.convolve(np.exp(beta * time), AIF)
        )[: len(AIF)]
    )

    Ce = (
        (time[1] - time[0])
        * flow
        * (
            Ae * np.convolve(np.exp(alpha * time), AIF)
            - Ae * np.convolve(np.exp(beta * time), AIF)
        )[: len(AIF)]
    )

    return Cp[:, np.newaxis], Ce[:, np.newaxis]


def split_two_comp_ex_model_tf(parameters, time, AIF):
    [alpha, beta, Ap, Ae] = phys_params_to_split_model_params(parameters)

    flow = parameters[0]
    padding_data = tf.constant([[len(time) - 1, 0]])
    kernel = AIF[::-1]
    kernel = tf.reshape(kernel, [len(AIF), 1, 1])

    Alpha = tf.reshape(
        tf.pad(tf.exp(alpha * time), padding_data, "CONSTANT"),
        [1, len(time) + len(AIF) - 1, 1],
    )
    Beta = tf.reshape(
        tf.pad(tf.exp(beta * time), padding_data, "CONSTANT"),
        [1, len(time) + len(AIF) - 1, 1],
    )
    Alpha = tf.nn.conv1d(Alpha, kernel, stride=1, padding="VALID")
    Beta = tf.nn.conv1d(Beta, kernel, stride=1, padding="VALID")

    split_list = []
    for i in [Ap, Ae]:
        split_list.append((time[1] - time[0]) * flow * (i * Alpha + (1 - i) * Beta))

    return split_list


def residual(parameters, time, AIF, curve):

    r = curve - two_comp_ex_model(parameters, time, AIF)
    return np.sum(r ** 2) / (len(curve))


def tf_conv_full(inp, kernel):
    # reshape input
    padding_data = tf.constant([[inp.shape[0] - 1, 0], [0, 0]])
    inp = tf.expand_dims(
        tf.transpose(tf.pad(inp, padding_data, "CONSTANT"), [1, 0]), -1
    )

    # reshape kernel
    kernel = tf.reshape(kernel[::-1], [kernel.shape[0], 1, 1])
    # [::-1]
    conv = tf.nn.conv1d(inp, kernel, stride=1, padding="VALID")
    return tf.transpose(conv[..., 0], [1, 0])
