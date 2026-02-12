import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import wrf_functions
import os

# Specify a matplotlib style
plt.style.use("./project.mplstyle")
# Default directory for data
data_dir = "./data/"
vmins = [6, 10, 13, 21]
vmaxs = [7, 11, 14, 22]
vmins = [10]
vmaxs = [11]
for vmin, vmax in zip(vmins, vmaxs):
    plot_dir = f"./plots/stable_{vmin}_{vmax}/"
    os.makedirs(plot_dir, exist_ok=True)
    data = pd.read_pickle(f"{data_dir}/wrf_df.pkl")
    data["time"] = data.index.get_level_values(
        0
    )  # This is just because multiindex is painful
    data["height"] = data.index.get_level_values(
        1
    )  # This is just because multiindex is painful

    ## Convert wind speed/direction into u,v values
    data["u"] = np.sin(np.radians(data.wind_direction)) * data.wind_speed
    data["v"] = np.cos(np.radians(data.wind_direction)) * data.wind_speed

    criteria_col = "is_stable"
    stable = data.groupby(level=0).apply(wrf_functions.is_stable, vmin = vmin, vmax = vmax)
    stable.name = criteria_col
    stable = pd.DataFrame(stable)
    data = data.join(stable)

    duration_col = "stable_duration"
    data[duration_col] = 0
    data = wrf_functions.add_duration(data, criteria_col, duration_col)

    stable_durations = sorted(data[duration_col].unique())


    ## This is a hand selected time to avoid LLJs
    for d in stable_durations[-20:]:
        if d < 10:
            continue
        print(data[data[duration_col] == d].index.unique(level=0))
        for selected_time in data[data[duration_col] == d].index.unique(level=0):
        # selected_time = data[data[duration_col] == d].index.unique(level=0)[0]
            df = data.loc[selected_time - pd.Timedelta(3600, "s"): selected_time + pd.Timedelta(d, "s")]
            df.to_csv(f"{plot_dir}/{df.time.iloc[0].date()}-{d}.csv")
            wfig = wrf_functions.plot_slice(df, 'wind_speed', 'Wind Speed (m/s)', plot_dir, "viridis", f"-{d}")
            tfig = wrf_functions.plot_slice(df, 'temperature', r'$\theta (\mathrm{K})$', plot_dir, "magma", f"-{d}")
            df = data.loc[selected_time: selected_time + pd.Timedelta(d, "s")]
            mean_df = df.groupby(level=1).mean()

            pre_df = data.loc[selected_time - pd.Timedelta(3600, "s"): selected_time]
            pre_mean_df = pre_df.groupby(level=1).mean()

    ## Averaged line plots
            plot_labels = {
                "wind_speed": "Wind speed (m/s)",
                "tke": "TKE",
                "temperature": r"$\theta (\mathrm{K})$",
            }
            with PdfPages(f"{plot_dir}/stable_plots_{mean_df.time.iloc[0].date()}-{d}.pdf") as pdf:
                pdf.savefig(wfig)
                pdf.savefig(tfig)
                for k in plot_labels:
                    plt.figure(k)
                    plt.plot(mean_df[k].values, mean_df.height.values, label=plot_labels[k])
                    # plt.plot(pre_mean_df[k].values, mean_df.height.values, label=plot_labels[k]+'1 hour prior', ls='dashed')
                    plt.ylabel("Height (m AMSL)")
                    plt.xlabel(plot_labels[k])
                    plt.legend()
                    pdf.savefig()
                plt.figure('veer')
                plt.plot(wrf_functions.get_average_veer_profile(df), mean_df.height.values)
                plt.ylabel("Height (m AMSL)")
                plt.xlabel("Veer rel. to 140m ($^\circ$)")
                pdf.savefig()
                plt.figure('WD')
                plt.plot(wrf_functions.get_deltaWD(df), mean_df.height.values)
                plt.ylabel("Height (m AMSL)")
                plt.xlabel(r"$\Delta$ Wind Direction ($^\circ$)")
                pdf.savefig()
            plt.close("all")
