# import necessary modules
import numpy as np
import tensorflow as tf

from src import kinetic_models
from sklearn.metrics import auc


def make_dro(tf_format=0, ps_slice=0, snr=0):
    dro_dims = (40, 120, 3)

    # aif
    aif_time = np.around(np.arange(0, 2.0, 0.02), decimals=2).astype(np.float32)
    aif = kinetic_models.gamma_func(10 ** 5 / 3.0, 2.0, 0.1, aif_time)

    # set parameter volumes
    flows = np.empty(dro_dims)
    for i, flow in enumerate((0.5, 1.0, 1.5, 2.0)):
        flows[i * 10 : (i + 1) * 10] = flow
    vps = np.empty(dro_dims)
    for i, vp in enumerate((0.02, 0.05, 0.1, 0.2)):
        for j in range(3):
            vps[:, 10 * (i + 4 * j) : 10 * (i + 1 + 4 * j)] = vp
    ves = np.empty(dro_dims)
    for i, ve in enumerate((0.1, 0.2, 0.5)):
        ves[:, i * 40 : (i + 1) * 40] = ve
    pss = np.empty(dro_dims)
    for i, ps in enumerate((0.5, 1.5, 2.5)):
        pss[..., i] = ps

    myo_pars = np.stack((flows, vps, ves, pss), axis=-1)[..., np.newaxis, :].astype(
        np.float32
    )
    myo_curves = np.empty((len(aif),) + dro_dims + (1,)).astype(np.float32)
    myo_t_curves = np.empty_like(myo_curves)

    for i in range(dro_dims[0]):
        for j in range(dro_dims[1]):
            for k in range(dro_dims[2]):
                signal = kinetic_models.two_comp_ex_model(
                    myo_pars[i, j, k, 0, :], aif_time, aif
                )

                # v approximates v_e + v_p
                myo_curves[:, i, j, k, 0] = signal
                myo_t_curves[:, i, j, k, 0] = np.gradient(
                    myo_curves[:, i, j, k, 0], aif_time[1] - aif_time[0]
                )

    if snr:
        # add noise
        signal_db_mean = 10 * np.log10(np.mean(myo_curves))
        noise_db_mean = signal_db_mean - snr
        noise_pw_mean = 10 ** (noise_db_mean / 10)
        noise_pw = np.random.normal(0.0, np.sqrt(noise_pw_mean), myo_curves.shape)
        myo_curves += noise_pw

    # remove impossible values
    myo_curves[0, ...] = 0
    # myo_curves = np.maximum(myo_curves, 0)

    # input normalization
    t_std = aif_time.std()
    aif_time = (aif_time - aif_time.mean()) / aif_time.std()

    # output normalization
    max_ = aif.max()
    aif /= max_
    myo_curves /= max_
    myo_t_curves /= max_

    # r and b
    coll_points = np.random.uniform(
        min(aif_time), max(aif_time), len(aif_time) * 5
    ).astype(np.float32)
    bound = np.array([min(aif_time)])
    extrapolate = np.random.uniform(
        max(aif_time), max(aif_time) * 3, len(aif_time) * 5
    ).astype(np.float32)

    # select slice
    myo_pars = myo_pars[..., ps_slice, :, :]
    myo_curves = myo_curves[..., ps_slice, :]
    myo_t_curves = myo_t_curves[..., ps_slice, :]

    if tf_format == -1:
        # return as numpy arrays
        return aif_time, aif, myo_curves, myo_pars

    if tf_format == 0:
        # flatten myo curves
        myo_curves = myo_curves.reshape((myo_curves.shape[:-3] + (-1,)))
        myo_t_curves = myo_t_curves.reshape((myo_t_curves.shape[:-3] + (-1,)))

        return (
            tf.data.Dataset.from_tensor_slices(
                (
                    aif_time[..., np.newaxis],
                    aif[..., np.newaxis],
                    myo_curves,
                    myo_t_curves,
                )
            ),
            tf.data.Dataset.from_tensor_slices((coll_points[..., np.newaxis],)),
            tf.data.Dataset.from_tensor_slices((extrapolate[..., np.newaxis],)),
            tf.data.Dataset.from_tensor_slices((bound[..., np.newaxis],)),
            myo_pars,
            t_std,
        )
    elif tf_format > 0:
        # create meshes
        mesh_x, mesh_y = np.meshgrid(
            np.linspace(0, dro_dims[1] - 1, num=dro_dims[1]),
            np.linspace(0, dro_dims[0] - 1, num=dro_dims[0]),
        )
        mesh_u = np.tile(
            np.stack((mesh_x, mesh_y), axis=-1), (len(aif), 1, 1, 1)
        ).astype(np.float32)
        mesh_r = np.tile(
            np.stack((mesh_x, mesh_y), axis=-1), (len(coll_points), 1, 1, 1)
        ).astype(np.float32)
        mesh_e = np.tile(
            np.stack((mesh_x, mesh_y), axis=-1), (len(extrapolate), 1, 1, 1)
        ).astype(np.float32)
        mesh_b = np.tile(
            np.stack((mesh_x, mesh_y), axis=-1), (len(bound), 1, 1, 1)
        ).astype(np.float32)

        # mesh normalization
        mesh_u = (mesh_u - mesh_u.mean()) / mesh_u.std()
        mesh_r = (mesh_r - mesh_r.mean()) / mesh_r.std()
        mesh_e = (mesh_e - mesh_e.mean()) / mesh_e.std()
        mesh_b = (mesh_b - mesh_b.mean()) / mesh_b.std()

        # tile up
        aif_time = np.tile(
            aif_time[..., np.newaxis, np.newaxis, np.newaxis],
            (1, dro_dims[0], dro_dims[1], 1),
        ).astype(np.float32)
        aif = np.tile(
            aif[..., np.newaxis, np.newaxis, np.newaxis],
            (1, dro_dims[0], dro_dims[1], 1),
        ).astype(np.float32)
        coll_points = np.tile(
            coll_points[..., np.newaxis, np.newaxis, np.newaxis],
            (1, dro_dims[0], dro_dims[1], 1),
        ).astype(np.float32)
        extrapolate = np.tile(
            extrapolate[..., np.newaxis, np.newaxis, np.newaxis],
            (1, dro_dims[0], dro_dims[1], 1),
        ).astype(np.float32)
        bound = np.tile(
            bound[..., np.newaxis, np.newaxis, np.newaxis],
            (1, dro_dims[0], dro_dims[1], 1),
        ).astype(np.float32)

        if tf_format == 1:
            return (
                tf.data.Dataset.from_tensor_slices(
                    (mesh_u, aif_time, aif, myo_curves, myo_t_curves,)
                ),
                tf.data.Dataset.from_tensor_slices((mesh_r, coll_points,)),
                tf.data.Dataset.from_tensor_slices((mesh_e, extrapolate,)),
                tf.data.Dataset.from_tensor_slices((mesh_b, bound,)),
                myo_pars,
                t_std,
            )

        elif tf_format == 2:
            # reshape meshes and time inputs
            mesh_u = mesh_u.reshape((-1, mesh_u.shape[-1]))
            mesh_r = mesh_r.reshape((-1, mesh_r.shape[-1]))
            mesh_e = mesh_e.reshape((-1, mesh_e.shape[-1]))
            mesh_b = mesh_b.reshape((-1, mesh_b.shape[-1]))
            aif_time = aif_time.reshape((-1, aif_time.shape[-1]))
            coll_points = coll_points.reshape((-1, coll_points.shape[-1]))
            extrapolate = extrapolate.reshape((-1, extrapolate.shape[-1]))
            bound = bound.reshape((-1, bound.shape[-1]))

            # outputs
            aif = aif.reshape((-1, aif.shape[-1]))
            myo_curves = myo_curves.reshape((-1, myo_curves.shape[-1]))
            myo_t_curves = myo_t_curves.reshape((-1, myo_t_curves.shape[-1]))

            return (
                tf.data.Dataset.from_tensor_slices(
                    (mesh_u, aif_time, aif, myo_curves, myo_t_curves,)
                ),
                tf.data.Dataset.from_tensor_slices((mesh_r, coll_points,)),
                tf.data.Dataset.from_tensor_slices((mesh_e, extrapolate,)),
                tf.data.Dataset.from_tensor_slices((mesh_b, bound,)),
                myo_pars,
                t_std,
            )

        elif tf_format == 3:
            aif_time = np.concatenate((aif_time, mesh_u), axis=-1)
            return (
                tf.data.Dataset.from_tensor_slices((aif_time, aif, myo_curves)),
                myo_pars,
            )
