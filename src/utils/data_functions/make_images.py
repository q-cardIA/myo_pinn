# import necessary modules
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import os 

from skimage.metrics import structural_similarity as ssi
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util

from src import make_dro, CombinedPINN, PatientData

def training_visualization(paths):
    string = "logs"
    dict_list = []

    mpl.style.use('seaborn-darkgrid')   

    for path in paths:
        training_dict = {}
        for event in summary_iterator(os.path.join(string, path)):
            if "step" in training_dict:
                if event.step not in training_dict["step"]:
                    training_dict["step"].append(event.step)
            else:
                if event.step != 0:
                    training_dict["step"] = [event.step]

            for value in event.summary.value:
                key = value.tag
                item = tensor_util.MakeNdarray(value.tensor)

                if key in training_dict:
                    training_dict[key].append(item.item())
                else:
                    training_dict[key] = [item.item()]
        dict_list.append(training_dict)

    # plotting
    fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, num=1, figsize=(6.2, 10))

    for dict_t in dict_list:
        axes[0, 0].plot(dict_t["step"], dict_t["training/loss_u"],lw = 2)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title(r"$L_C$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[0, 0].set_ylabel(r"$MSE$", fontsize=15,fontweight='bold',fontname='Calibri')
        # axes[0, 0].set_yticks([0.2, 0.1])
        axes[0, 0].tick_params(axis="y", labelsize=12)

        axes[0, 1].plot(dict_t["step"], dict_t["training/loss_r"],lw = 2)
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title(r"$L_r$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[0, 1].tick_params(axis="y", labelsize=12)

        axes[1, 0].plot(dict_t["step"], dict_t["training/loss_b"],lw = 2)
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_title(r"$L_b$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[1, 0].set_ylabel(r"$MSE$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[1, 0].tick_params(axis="y", labelsize=12)

        axes[1, 1].plot(dict_t["step"], dict_t["training/loss_reg"],lw = 2)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title(r"$L_{reg}$", fontweight="bold", fontsize=15,fontname='Calibri')
        axes[1, 1].tick_params(axis="y", labelsize=12)

    max_f = 0.0
    max_vp = 0.0
    max_ve = 0.0
    max_ps = 0.0
    for dict_t in dict_list:
        max_f = max(dict_t["vars/f"]) if max(dict_t["vars/f"]) > max_f else max_f
        max_vp = max(dict_t["vars/vp"]) if max(dict_t["vars/vp"]) > max_vp else max_vp
        max_ve = max(dict_t["vars/ve"]) if max(dict_t["vars/ve"]) > max_ve else max_ve
        max_ps = max(dict_t["vars/ps"]) if max(dict_t["vars/ps"]) > max_ps else max_ps
        max_f += 0.1 * max_f
        max_vp += 0.1 * max_vp
        max_ve += 0.1 * max_ve
        max_ps += 0.1 * max_ps

    for dict_t in dict_list:
        axes[2, 0].plot(dict_t["step"], dict_t["vars/f"],lw = 2)
        axes[2, 0].set_ylim(ymin=0.0, ymax=max_f)
        axes[2, 0].set_title(r"$F_p$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[2, 0].set_ylabel(r"$\mu_{value}$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[2, 0].tick_params(axis="y", labelsize=12)

        axes[2, 1].plot(dict_t["step"], dict_t["vars/vp"],lw = 2)
        axes[2, 1].set_ylim(ymin=0.0, ymax=max_vp)
        axes[2, 1].set_title(r"$v_{p}$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[2, 1].tick_params(axis="y", labelsize=12)

        axes[3, 1].plot(dict_t["step"], dict_t["vars/ve"],lw = 2)
        axes[3, 1].set_ylim(ymin=0.0, ymax=max_ve)
        axes[3, 1].set_title(r"$v_{e}$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[3, 1].set_xlabel("epoch", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[3, 1].tick_params(axis="y", labelsize=12)

        axes[3, 0].plot(dict_t["step"], dict_t["vars/ps"],lw = 2)
        axes[3, 0].set_ylim(ymin=0.0, ymax=max_ps)
        axes[3, 0].set_xlabel("epoch", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[3, 0].set_title(r"$PS$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[3, 0].set_ylabel(r"$\mu_{value}$", fontsize=15,fontweight='bold',fontname='Calibri')
        axes[3, 0].tick_params(axis="y", labelsize=12)

    fig.subplots_adjust(wspace=0.35, hspace=0.2)
    fig.savefig(
        "training_visualisation.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
    )

    for i in range(2):
        for j in range(2):
            axes[i, j].set_ylim(ymin=0.0)

    return dict_list

def show_maps(ID, weight_paths, param):
    # plotting

    fig, axes = plt.subplots(nrows=1, ncols=3)

    flows = []
    for i in range(3):
        # load patient slice
        patient = PatientData(ID, i + 1)
        kinetic_data = patient.tf_load(0)
        N = patient.get_segmentation()
        current_lr = 1e-3

        pinn = CombinedPINN(
            N=N,
            bn=1,
            log_domain=1,
            lr=current_lr,
            layers=2,
            layer_width=32,
            loss_weights=(5, 1, 0, 1, 1, 0),
            # shape_in=(
            #    kinetic_data[0].element_spec[0].shape,
            #    kinetic_data[0].element_spec[1].shape,
            # ),
            shape_in=(kinetic_data[0].element_spec[0].shape,),
            trainable_pars="all",
            std_t=kinetic_data[-1],
        )
        #pinn.set_trainable_pars("transfer")
        pinn.load(weight_paths[i])

        c = patient.get_crop_values()
        param_map = np.pad(
            pinn.predict()[..., 0, param] * patient.get_segmentation(),
            ((c[0], c[2]), (c[1], c[3])),
        )
        est_map = param_map #np.ma.masked_where(param_map == 0, param_map)
        flows.append(param_map[np.nonzero(param_map)])

        if param == 0:
            vmax = 2.5
            hct = 0.45
        elif param == 1:
            vmax = 0.2
            hct = 0.45
        elif param == 2:
            vmax = 0.25
            hct = 0
        else:
            vmax = 1.0
            hct = 0


        axes[i].axes.yaxis.set_ticks([])
        seg_im = patient.get_full_segmentation()[..., np.newaxis]
        seg_im = seg_im * np.load(patient.full_mr)[:seg_im.shape[0], :seg_im.shape[1]]
        min_ = seg_im.min()
        max_ = seg_im.max()

        im = axes[i].imshow(est_map/(1-hct), cmap="CMRmap", vmin=0.0, vmax=vmax)

    cax = inset_axes(
        axes[2],
        width="5%",
        height="100%",
        bbox_to_anchor=(0.2, 0, 1, 1),
        bbox_transform=axes[2].transAxes,
        loc="center right",
    )

    axes[0].set_title("Basal", fontweight="bold")
    axes[1].set_title("Mid-cavity", fontweight="bold")
    axes[2].set_title("Apical", fontweight="bold")
    axes[0].set_ylabel("PINN inference",)

    if param == 0:
        bar = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, .5, 1, 1.5, 2, 2.5])
    else:
        bar = fig.colorbar(im, cax=cax, orientation="vertical")
    bar.ax.tick_params(labelsize=8)

    plt.suptitle("{} inference".format(patient.name))
    plt.show()

    return flows
