# import necessary modules
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class PatientData:
    """
    The class PatientData can be used to easily generate the data of a patient, e.g. loading their contrast agent data,
    simply by giving the ID of the desired patient.
    """

    def __init__(self, patient_ID, slice_ID):
        self.name = ''.join(['pt',str(patient_ID)])
        self.slice = slice_ID

        self.params = None

        self.paths = sorted(
                    glob.glob(
                        "data/processed_fitting_data/{}_*{}.npy".format(self.name, slice_ID)
                    ), key=str.casefold)
        self.aif_path = self.paths[0]
        self.myo_path = self.paths[1]
        self.full_mr = self.paths[2]
        self.rv_path = self.paths[3]
        self.t_path = self.paths[-1]

    def plot_frame(self, t_frame):
        myo_im = np.load(self.myo_path)

        plt.figure()
        plt.imshow(myo_im[..., t_frame])
        plt.title("Patient {}, frame {}".format(self.name, t_frame))
        plt.show()

    def plot_params(self, cmap="CMRmap"):
        # crop the image
        seg = np.amax(np.load(self.myo_path), axis=2)
        seg[seg > 0] = 1
        id = np.argwhere(seg)
        x_min, y_min = id.min(axis=0)
        x_max, y_max = id.max(axis=0)
        seg = seg[x_min : x_max + 1, y_min : y_max + 1]

        plt.figure()
        plt.suptitle("{} slice {} parameters".format(self.name, self.slice))

        for i, param in enumerate((r"$F$", r"$v_{p}$", r"$v_{e}$", r"$PS$")):
            im = self.params[..., 0, i] * seg

            # set vmax
            if i == 0:
                vmax = 3.0
            elif i == 1:
                vmax = 0.08
            elif i == 2:
                vmax = 0.30
            else:
                vmax = 1.0

            plt.subplot(2, 2, i + 1)
            plt.imshow(im, cmap=cmap, vmax=vmax)
            plt.colorbar()
            plt.title(param)
            plt.axis("off")

        plt.tight_layout()

        plt.show()

    def get_segmentation(self):
        # crop the image
        seg = np.amax(np.load(self.myo_path), axis=2)
        seg[seg > 0] = 1
        id = np.argwhere(seg)
        x_min, y_min = id.min(axis=0)
        x_max, y_max = id.max(axis=0)
        seg = seg[x_min : x_max + 1, y_min : y_max + 1]

        return seg

    def get_full_segmentation(self):
        # crop the image
        seg = np.amax(np.load(self.myo_path), axis=2)
        seg[seg > 0] = 1

        return seg

    def get_crop_values(self):
        # crop the image
        seg = np.amax(np.load(self.myo_path), axis=2)
        seg[seg > 0] = 1
        id = np.argwhere(seg)
        x_min, y_min = id.min(axis=0)
        x_max, y_max = id.max(axis=0)
        return x_min, y_min, seg.shape[0] - (x_max + 1), seg.shape[1] - (y_max + 1)

    def tf_load(self, tf_format=0):
        # crop the image
        #seg = np.load(self.seg_path)
        seg = np.amax(np.load(self.myo_path), axis=2)
        id = np.argwhere(seg)
        x_min, y_min = id.min(axis=0)
        x_max, y_max = id.max(axis=0)

        # load patient data
        aif_time = np.load(self.t_path).astype(np.float32)
        aif = np.load(self.aif_path).astype(np.float32)
        myo_curves = np.moveaxis(np.load(self.myo_path).astype(np.float32), -1, 0)[
            ..., np.newaxis
        ]
        myo_curves = myo_curves[:, x_min : x_max + 1, y_min : y_max + 1, :]

        # remove impossible values
        aif_time_0 = np.array([-aif_time[1]])
        aif_0 = np.array([0.0]).astype(np.float32)
        myo_curves_0 = np.zeros_like(myo_curves[0, ...])[np.newaxis, ...]

        aif_time = np.concatenate((aif_time_0, aif_time))
        aif = np.concatenate((aif_0, aif))
        myo_curves = np.concatenate((myo_curves_0, myo_curves))

        # generate gradient
        dims = myo_curves.shape[1:]
        myo_t_curves = np.empty_like(myo_curves)
        for i in range(dims[0]):
            for j in range(dims[1]):
                myo_t_curves[:, i, j, 0] = np.gradient(
                    myo_curves[:, i, j, 0], aif_time[1] - aif_time[0]
                )

        # input normalization
        t_std = aif_time.std()
        aif_time = (aif_time - aif_time.mean()) / aif_time.std()

        # output normalization
        # max normalization
        max_ = aif.max()
        aif /= max_
        myo_curves /= max_
        myo_t_curves /= max_

        # r and b
        coll_points = np.random.uniform(min(aif_time), max(aif_time), 500).astype(
            np.float32
        )
        bound = np.array([min(aif_time)])
        extrapolate = np.random.uniform(
            max(aif_time), max(aif_time) * 3, len(aif_time) * 5
        ).astype(np.float32)

        if tf_format == -1:
            # return as numpy arrays
            return aif_time, aif, myo_curves, None

        if tf_format == 0:
            # flatten myo curves
            myo_curves = myo_curves.reshape((myo_curves.shape[:-3] + (-1,)))
            myo_t_curves = myo_t_curves.reshape((myo_t_curves.shape[:-3] + (-1,)))

            myo_curves = myo_curves[:aif.shape[0]]
            myo_t_curves = myo_t_curves[:aif.shape[0]]

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
                None,
                t_std,
            )
        elif tf_format > 0:
            # create meshes
            mesh_x, mesh_y = np.meshgrid(
                np.linspace(0, dims[1] - 1, num=dims[1]),
                np.linspace(0, dims[0] - 1, num=dims[0]),
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
                (1, dims[0], dims[1], 1),
            ).astype(np.float32)
            aif = np.tile(
                aif[..., np.newaxis, np.newaxis, np.newaxis], (1, dims[0], dims[1], 1),
            ).astype(np.float32)
            coll_points = np.tile(
                coll_points[..., np.newaxis, np.newaxis, np.newaxis],
                (1, dims[0], dims[1], 1),
            ).astype(np.float32)
            extrapolate = np.tile(
                extrapolate[..., np.newaxis, np.newaxis, np.newaxis],
                (1, dims[0], dims[1], 1),
            ).astype(np.float32)
            bound = np.tile(
                bound[..., np.newaxis, np.newaxis, np.newaxis],
                (1, dims[0], dims[1], 1),
            ).astype(np.float32)

            if tf_format == 1:
                return (
                    tf.data.Dataset.from_tensor_slices(
                        (mesh_u, aif_time, aif, myo_curves, myo_t_curves,)
                    ),
                    tf.data.Dataset.from_tensor_slices((mesh_r, coll_points,)),
                    tf.data.Dataset.from_tensor_slices((mesh_e, extrapolate,)),
                    tf.data.Dataset.from_tensor_slices((mesh_b, bound,)),
                    None,
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
                    None,
                    t_std,
                )

            elif tf_format == 3:
                aif_time = np.concatenate((aif_time, mesh_u), axis=-1)
                return (
                    tf.data.Dataset.from_tensor_slices((aif_time, aif, myo_curves)),
                    None,
                )

    def find_parameters(
        self, tf_format, model, batch_size, num_epochs,
    ):
        data = self.tf_load(tf_format)

        model.fit(
            data[0], data[1], data[2], data[3], data[4], batch_size, num_epochs,
        )

        self.params = model.predict()

    def view_mri(self, param_i, alpha=0.15):
        i = self.params[..., 0, param_i] if param_i else None
        view_mri(self.full_mr, i, alpha=alpha)

    def view_myocardium(self, param_i, alpha=0.15):
        i = self.params[..., 0, param_i] if param_i else None
        view_mri(self.myo_path, i, alpha=alpha)
