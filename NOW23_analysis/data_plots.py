import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from windrose import WindroseAxes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm

import wrf_functions

# Specify a matplotlib style
plt.style.use("./project.mplstyle")
# Default directory for data
data_dir = "./data/"
plot_dir = "./plots/data_plots"

data = pd.read_pickle(f"{data_dir}/wrf_df.pkl")
data["time"] = data.index.get_level_values(
    0
)  # This is just because multiindex is painful
data["h"] = data.index.get_level_values(
    1
)  # This is just because multiindex is painful

## Convert wind speed/direction into u,v values
data["u"] = np.sin(np.radians(data.wind_direction)) * data.wind_speed
data["v"] = np.cos(np.radians(data.wind_direction)) * data.wind_speed
data["month"] = [x.month for x in data.time]
data["hour"] = [x.hour for x in data.time]
## Helper functions
def plot_interannual_speeds(data, plot_dir):
    ws = data.wind_speed[data.h == 140][data.wind_speed > 0][data.wind_speed < 59]
    hourly = ws.resample('1h', level = 'time').mean()
    daily = ws.resample('1D', level = 'time').mean()
    monthly = ws.resample('1ME', level = 'time').mean()
    plt.figure()
    hourly.plot(ls='solid', linewidth = 0.5, alpha=0.7, color = 'gray', label ='Hourly')
    daily.plot(ls='solid', linewidth = 0.5, alpha=0.9, color = 'k', label = 'Daily')
    monthly.plot(ls='solid', marker = 'o', color = 'red', ms=3, label = 'Monthly' )
    xticks = [monthly.index[0], monthly.index[len(monthly)//2], monthly.index[-1]]
    t_labels = [f'{x.month}/{x.year}' for x in xticks]
    plt.xticks(xticks, t_labels)
    plt.xticks(monthly.index, ['' for _ in monthly.index], minor = True)
    plt.legend(loc=9)
    plt.ylabel('Wind Speed (m/s)')
    plt.savefig(f'{plot_dir}/interannual.pdf')
    plt.close()

def plot_wind_contour(data, plot_dir, season = 'all', time_key = 'hour', time_label='Hour of Day'):
    df = data.groupby(by=[time_key, 'height'],).mean(numeric_only = True)
    N = len(df)
    alts = df.index.levels[1]
    times = df.index.levels[0]
    nt = len(times)
    na = len(alts)
    ws = df.wind_speed.values.reshape((nt, na))
    u = df.u.values.reshape((nt, na))
    v = df.v.values.reshape((nt, na))
    theta = np.arctan2(u,v)
    u,v = np.sin(theta), np.cos(theta)

    x, y = np.meshgrid(alts,times)

    f = plt.figure()
    c = plt.contourf(y, x, ws, cmap="viridis", levels=np.arange(5,14))
    f.colorbar(c, label="Wind speed (m/s)")
    plt.quiver(y,x, u,v)
    plt.xlabel(time_label)
    plt.ylabel("Height (m AMSL)")
    f.savefig(f"{plot_dir}/{time_key}_{season}.pdf")
    plt.close()

def plot_windroses(data, plot_dir, suffix="all"):
    for altitude in [140]:
        plt.figure()
        ax = WindroseAxes.from_ax()
        valid_hub_speeds = data[
            (data.h == altitude)
            & (data.wind_speed > 0)
            & (data.wind_speed < 59)
            & (data.wind_direction > 1)
            & (data.wind_direction < 359)
        ]
        wd = valid_hub_speeds.wind_direction
        ws = valid_hub_speeds.wind_speed
        ax.bar(wd, ws, normed=True, opening=0.8, edgecolor="white", nsector=32)
        #ax.bar(wd, ws, bins =np.linspace(0,20,10), normed=True, opening=0.8, edgecolor="white", nsector=32, cmap=cm.magma)
        ax.set_legend(loc="best")
        plt.savefig(f"{plot_dir}/windrose_{suffix}_{altitude}.pdf")
        f = plt.figure()
        # plt.hist2d(ws, wd, bins=[10,50], cmap='viridis')
        plt.hist2d(ws, wd, bins = [np.linspace(0,20,10), np.linspace(0,360,25)], cmap='viridis')
        # plt.xlim([0,20])
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Wind direction ($^\circ$)')
        plt.savefig(f"{plot_dir}/windpdf_{suffix}_{altitude}.pdf")
        fig = plt.figure()
        # gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                                                    wspace=0.0, hspace=0.0)

        # Create the main 2D histogram plot
        ax_main = fig.add_subplot(gs[1, 0])
        ax_main.hist2d(ws, wd, bins = [np.linspace(0,20,10), np.linspace(0,360,25)], cmap='viridis')
        ax_main.set(xlabel='Wind speed (m/s)', ylabel='Wind direction ($^\circ$)')

        # Create the marginal x histogram
        ax_x = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_x.hist(ws, bins=np.linspace(0,20,10), density = True, color = 'blue', histtype='step')
        ax_x.axis('off')

        # Create the marginal y histogram
        ax_y = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_y.hist(wd, bins=np.linspace(0,360,25), orientation='horizontal', color = 'green', density = True, histtype = 'step')
        ax_y.axis('off')

        plt.savefig(f"{plot_dir}/marg_windpdf_{suffix}_{altitude}.pdf")

## Mean windspeed plots
heights = data.index.levels[1]

plt.figure()
plt.plot(data.wind_speed.groupby(level=1).mean(), heights, linestyle='solid', label = 'all', linewidth=3.0, color='k')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel("Height (m AMSL)")

seasons = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}
# Plot seasonal variability
for season in seasons.keys():
    months = seasons[season]
    plt.plot(data[
            (data.month == months[0])
            | (data.month == months[1])
            | (data.month == months[2])
        ].wind_speed.groupby(level=1).mean(), heights, linestyle='--', label = season)
plt.legend()
# plt.xlim(6,13)
plt.savefig(f"{plot_dir}/mean_speed.pdf")
plt.close()

plot_wind_contour(data, plot_dir)
plot_interannual_speeds(data, plot_dir)
plot_windroses(data, plot_dir)
