# import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import os 

from src import kinetic_models, make_dro, MyoPINN, FullPINN, CombinedPINN, FullMeshPINN
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssi


def nlls_dro(ps_slice=1, snr=17.5):
    """
    Infer the parameters using the standard NLLS approach
    :param ps_slice: int of (0, 1, 2) referring to a PS full-slice value of (0.5, 1.5, 2.5) respectively
    :param snr: signal-to-noise ratio of the generated DRO data
    :return: parameter maps for the dro
    """
    data = make_dro(tf_format=-1, ps_slice=ps_slice, snr=snr)
    params = np.empty_like(data[3])

    for i in range(data[-1].shape[0]):
        for j in range(data[-1].shape[1]):
            fit = minimize(
                kinetic_models.residual,
                np.array([1, 0.2, 0.2, 1]),
                args=(data[0], data[1], data[2][:, i, j, 0]),
                tol=1e-8,
                method="L-BFGS-B",
                bounds=((1e-6, None), (1e-6, 1), (1e-6, 1), (1e-6, None)),
            )
            params[i, j, 0, :] = fit.x

    return data[3][..., 0, :], params[..., 0, :]


def train_dro(model, ps_slice=1, snr=17.5, epochs=15000):
    """
    Train a model on the DRO data
    :param model: the model which should fit the data
    :param ps_slice: int of (0, 1, 2) referring to a PS full-slice value of (0.5, 1.5, 2.5) respectively
    :param snr: signal-to-noise ratio of the generated DRO data
    :param epochs: number of dataset iterations during training
    :return: parameter maps for the dro
    """
    n = np.ones(shape=(40, 120))

    # set correct load values
    if model.__name__ in ("FullPINN", "MyoPINN", "CombinedPINN"):
        # no mesh info
        data = make_dro(tf_format=0, ps_slice=ps_slice, snr=snr)
        shape_in = (data[0].element_spec[0].shape,)
    elif model.__name__ == "FullMeshPINN":
        # with mesh info
        data = make_dro(tf_format=1, ps_slice=ps_slice, snr=snr)
        shape_in = (data[0].element_spec[0].shape, data[0].element_spec[1].shape)
    else:
        raise ValueError("Invalid model name")

    # define and train
    pinn = model(
        N=n,
        bn=1,
        log_domain=1,
        lr=1e-3,
        layers=2,
        layer_width=32,
        loss_weights=(5, 1, 0, 1, 1, 0),
        shape_in=shape_in,
        trainable_pars="all",
        std_t=data[-1],
    )

    if os.path.exists(os.path.join("model_weights","dro_" + model.__name__ + ".tf")):
        pinn.load(os.path.join("model_weights","dro_" + model.__name__ + ".tf"))
    else:
        pinn.fit(data[0], data[1], data[2], data[3], data[4], len(list(data[0])), epochs)

    return pinn.predict()[..., 0, :]


def plot_results(ps_slice=1):
    # initialize
    all_results = np.empty([40, 120, 6, 4])

    # get results
    all_results[..., 0, :], all_results[..., -1, :] = nlls_dro(ps_slice)
    all_results[..., 1, :] = train_dro(FullPINN)
    all_results[..., 2, :] = train_dro(FullMeshPINN)
    all_results[..., 3, :] = train_dro(MyoPINN)
    all_results[..., 4, :] = train_dro(CombinedPINN)

    # errors
    mse = np.empty([6, 4])
    ssim = np.empty([6, 4])
    for i in range(6):
        for j in range(4):
            mse[i, j] = np.mean(
                np.square(all_results[..., i, j] - all_results[..., 0, j])
            ) / np.mean(all_results[..., 0, j])
            ssim[i, j] = ssi(
                all_results[..., 0, j],
                all_results[..., i, j],
                data_range=np.max(all_results[..., i, j])
                - np.min(all_results[..., i, j]),
            )
    ssim[0, 3] = 1.0

    # plotting
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(8.27, 11.69))

    param_names = (r"$F$", r"$v_{p}$", r"$v_{e}$", r"$PS$")
    ticks = ([0, 1, 2, 3], [0, 0.08, 0.16, 0.24], [0, 0.2, 0.4, 0.6], [0, 1, 2, 3, 4])
    titles = ("GT", "2CXM", "2CXM + Mesh", "Reduced", "Combined", "NLLS")
    vmax = (3.0, 0.24, 0.60, 4.0)

    for i in range(4):
        for j in range(6):
            label = "NMSE: {:.2f}\nSSIM: {:.2f}".format(mse[j, i], ssim[j, i])

            im = axes[i, j].imshow(
                all_results[..., j, i].transpose(), vmin=0.0, vmax=vmax[i], cmap="CMRmap"
            )
            axes[i, j].set_xlabel(label, fontsize=8)
            axes[i, j].axes.xaxis.set_ticks([])
            axes[i, j].axes.yaxis.set_ticks([])

            if j == 5:
                # Add the colorbar
                cax = inset_axes(
                    axes[i, j],
                    width="20%",
                    height="100%",
                    bbox_to_anchor=(0.75, 0, 1, 1),
                    bbox_transform=axes[i, j].transAxes,
                    loc="right",
                )
                bar = fig.colorbar(
                    im, cax=cax, orientation="vertical", ticks=ticks[i]
                )
                bar.ax.tick_params(labelsize=8)
            if j == 0:
                # Add the param as ylabel
                fs = 18
                pad = 20
                axes[i, j].set_ylabel(
                    param_names[i], fontsize=fs, fontweight="bold", labelpad=pad
                )
            if i == 0:
                # Add the model name as title
                fs = 14
                axes[i, j].set_title(titles[j], fontsize=fs, fontweight="bold")
    plt.show()

    return all_results


if __name__ == "__main__":
    plot_results()