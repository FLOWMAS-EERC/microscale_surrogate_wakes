import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

plt.style.use("./project.mplstyle")

# Check if a numpy array is monotonically increasing or decreasing
# Returns:
#   Increasing: 1
#   Decreasing: -1
#   Non-monotonic: 0

def is_monotonic(a, start_idx = 0, ax = 0):
    if np.all(a[start_idx+1:] >= a[start_idx:-1], axis = ax):
        return 1
    elif np.all(a[start_idx+1:] <= a[start_idx:-1], axis = ax):
        return -1
    return 0


# Setting basic criteria for neutral profile detection
# Current criteria 7 m/s < windspeed < 10 m/s
# Monin-Obukhov Length > 1000
def is_neutral(d, vmin=7, vmax=10, MOL_min=1000):
    profile = d.loc[d.name]
    is_valid_velocity = vmax > profile.wind_speed[140.0] > vmin
    is_valid_MOL = abs(profile.invMOL[140.0]) < 1.0 / MOL_min
    increasing_wind_speed = is_monotonic(profile.wind_speed.values)
    decreasing_temperature = -is_monotonic(profile.temperature.values, 5)
    is_neutral = int(is_valid_velocity and is_valid_MOL and increasing_wind_speed == 1)
    #is_neutral = int(is_valid_velocity and is_valid_MOL and increasing_wind_speed == 1 and decreasing_temperature == 1)
    return is_neutral


# Setting basic criteria for neutral profile detection
# Current criteria 7 m/s < windspeed < 10 m/s
# Monin-Obukhov Length > 1000
def is_stable(d, vmin=11, vmax=25, MOL_min=1, MOL_max=500):
    profile = d.loc[d.name]
    is_valid_velocity = vmax > profile.wind_speed[140.0] > vmin
    is_valid_MOL = 1.0/MOL_max < profile.invMOL[140.0] < 1.0 / MOL_min
    increasing_wind_speed = is_monotonic(profile.wind_speed.values)
    decreasing_tke = -is_monotonic(profile.tke.values, 5)
    increasing_temperature = is_monotonic(profile.temperature.values, 5)
    #is_stable = int(is_valid_velocity and is_valid_MOL and increasing_wind_speed and decreasing_tke and increasing_temperature)
    is_stable = int(is_valid_velocity and is_valid_MOL and increasing_wind_speed)
    return is_stable

# Setting basic criteria for neutral profile detection
# Current criteria 7 m/s < windspeed < 10 m/s
# Monin-Obukhov Length > 1000
def is_unstable(d, vmin=7, vmax=10, MOL_min=-500, MOL_max=-1):
    profile = d.loc[d.name]
    is_valid_velocity = vmax > profile.wind_speed[140.0] > vmin
    is_valid_MOL = 1.0/MOL_max < profile.invMOL[140.0] < 1.0 / MOL_min
    is_unstable = int(is_valid_velocity and is_valid_MOL)
    return is_unstable

# Check for contiguous occurrence of criteria and add the duration to an output column
def add_duration(data, criteria_col, output_col):
    neutral_times = data[data[criteria_col] == 1].index.get_level_values("time")
    seen = []
    dt = pd.Timedelta("5m")
    for time in neutral_times:
        if time not in seen:
            current = time
            seen.append(time)
            current_duration = dt
            if (current + dt) in neutral_times:
                while data.loc[current + dt, criteria_col].values[0] != 0:
                    current_duration += dt
                    current += dt
                    seen.append(current)
            else:
                current_duration += dt
                current += dt
            data.loc[time, output_col] = current_duration.total_seconds()
    return data


def is_llj(d, v_min=3.0, dumin=1.5, duminOverUjet=0.1, valid_max=59):
    time = d.time[0]
    profile = d.loc[time]
    heights = profile.index

    # Define which data we need to keep for characterizing the jets
    new_row = {}

    # min velocity and max velocity constraint for data quality
    # This will also filter out NAN values from missing timestamps
    is_valid_speed = (
        all(profile.wind_speed > 0.0)
        and all(profile.wind_speed < valid_max)
        and all(profile.wind_direction > 0.1)
        and all(profile.wind_direction < 359.9)
    )

    if is_valid_speed:
        is_min_velocity = profile.wind_speed[140.0] > v_min
        # peak within range
        umax = profile.wind_speed.max()
        z_umax = profile.wind_speed.idxmax()
        udir = profile.wind_direction[z_umax]
        is_peak_in_range = heights[0] < z_umax < heights[-1]

        # dropoff (find local minima above the peak)
        profileup = profile.wind_speed.loc[z_umax : heights[-1]]
        peaks = find_peaks(-profileup)[0]
        if len(peaks):
            z_locmin = profileup.index[peaks[0]]
        else:
            z_locmin = heights[-1]
        u_locmin = profile.wind_speed[z_locmin]

        du = umax - u_locmin
        is_valid_dropoff = (du > dumin) and (du / umax > duminOverUjet)

        isjet = is_peak_in_range and is_valid_dropoff and is_min_velocity
        # dudz_nose = (umax - profile.wind_speed[heights[0]])/(z_umax - heights[0])
        # is_high_shear = dudz_nose > 0.03
        # isjet = is_jet and is_high_shear

        if isjet:
            new_row["is_jet"] = 1
            new_row["nose_height"] = z_umax
            new_row["nose_top"] = z_locmin
            new_row["nose_speed"] = umax
            new_row["nose_direction"] = udir
        else:
            new_row["is_jet"] = 0
    else:
        new_row["is_jet"] = -1
    new_row["time"] = time

    new_jet = pd.DataFrame(new_row, index=[time])
    # new_jet = pd.DataFrame(new_row)
    return new_jet

def diff_angle(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)

def get_veer_profile(wds):
    veer = np.full(len(wds), np.nan)
    if 1.0 < wds[140.0] < 359.0:
        for i in range(len(wds)):
            if 1 < wds.values[i] < 359:
                veer[i] = wds.values[i] - wds[140]
    return veer


def get_average_veer_profile(df):
    times = df.time.unique()
    veers = [get_veer_profile(df.loc[t].wind_direction) for t in times]
    return np.nanmean(veers, axis=0)


def get_deltaWD(df):
    alts = df.index.levels[1]
    na = len(df.index.levels[1])
    deltaWD = np.empty(na)
    for i, alt in enumerate(alts):
        amax = df.xs(alt, level=1).wind_direction.max()
        amin = df.xs(alt, level=1).wind_direction.min()
        if amin < 1.0 or amax > 359.0:
            deltaWD[i] = np.nan
        else:
            deltaWD[i] = diff_angle(amax, amin)
    return deltaWD

##### Plotting functions #####
def plot_slice(df, key, label,  plot_dir, cmap = "viridis", postfix=""):
    N = len(df)
    na = len(df.index.levels[1])
    nt = N // na
    x = ((df.time.values - df.time.values[0]) / 60e9).reshape((nt, na))
    y = df.height.values.reshape((nt, na))
    z = df[key].values.reshape((nt, na))
    f = plt.figure()
    c = plt.contourf(x, y, z, cmap = cmap)
    f.colorbar(c, label=label)
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Height (m AMSL)")
    #f.savefig(f"{plot_dir}/{key}_contour_{df.time.iloc[0].date()}{postfix}.pdf")
    return f

def plot_wind_contour(data, plot_dir, season = 'all', time_key = 'hour', time_label='Hour of Day (EST)'):
    valid_data = data[
          (data.wind_speed > 0)
        & (data.wind_speed < 59)
        & (data.wind_direction > 1)
        & (data.wind_direction < 359)
    ]
    df = valid_data.groupby(by=[time_key, 'alt'],).mean(numeric_only = True)
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

    # f = plt.figure()
    # # plt.plot(valid_data.wind_speed.mean(level=1), alts)
    # plt.plot(valid_data.wind_speed.groupby(level=1).mean(), alts)
    # plt.xlabel('Wind speed (m/s)')
    # plt.ylabel("Height (m AMSL)")
    # plt.xlim(5,14)
    # f.savefig(f"{plot_dir}/mean_speed_{season}.pdf")
    # plt.close()

    if season == 'all':
        valid_data = data[
              (data.wind_speed > 0)
            & (data.wind_speed < 59)
            & (data.wind_direction > 1)
            & (data.wind_direction < 359)
            & (data.is_jet == 1)
        ]
        df = valid_data.groupby(by=[time_key, 'alt'],).mean(numeric_only = True)
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
        f.savefig(f"{plot_dir}/llj_{time_key}_{season}.pdf")
        plt.close()
