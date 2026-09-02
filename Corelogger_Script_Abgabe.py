"""Python script for data formatting, plotting and cluster analysis of corelogger measurements"""
from pathlib import Path
from io import StringIO
from PIL import Image, ImageOps
import math
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Patch
from brokenaxes import brokenaxes
from sklearn.cluster import KMeans

# --- Functions for PART 1: Data preparation ---
# Read out files
def read_format_out_file(filepath):
    """read_format_out_file reads an .out file in the specified folder, 
    extracts measurement data and saves it in a pandas dataframe.
    Args:
        filepath (string): filepath to folder containing .out file
    Returns:
        df_out (dataframe): df containing measurement data from .out file (unformatted)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines() # .out file is turned into text lines

    # Find header of data (column names)
    for l, line in enumerate(lines):
        if line.strip().startswith("SB DEPTH"):
            header_line = l
            break # stop going though lines; column row found

    # Data begins after unit row
    data = lines[header_line:] # everything after header line
    df_out = pd.read_csv(
        StringIO("".join(data)),
        sep=r"\s+",
        engine="python",
        skiprows=[1])
    df_out = df_out.iloc[:,:-3] # Last 3 columns (empty) will be discarded
    df_out.columns = ["SB DEPTH [cm]", "SECT NUM", "SECT DEPTH [cm]", "CT [cm]","PW Amp",
                      "PW Vel [m/s]", "Den [g/cc]", "Imp", "FP"] # Rename columns

    # Formatting and unit correction
    df_out.loc[1, "FP"] = df_out.loc[1, "Imp"] # value assigned wrongly
    df_out.loc[1, "Imp"] = None
    if "Den [g/cc]" in df_out.columns:
        df_out["Den [kg/m3]"] = df_out["Den [g/cc]"]*1000 # Density correct unit
        df_out = df_out.drop("Den [g/cc]", axis=1) # Drop old density column

    df_out["Imp [kg/m2s]"] = df_out["Den [kg/m3]"] * df_out["PW Vel [m/s]"]
    df_out = df_out.drop("Imp", axis=1) # Drop old impedance column

    if "SB DEPTH [cm]" in df_out.columns:
        df_out["SB DEPTH [m] rel"] = df_out["SB DEPTH [cm]"] / 100 # SB depth correct unit
        df_out = df_out.drop("SB DEPTH [cm]", axis=1) # Drop old SB depth column
        # Extract depth interval from filename
        filepath = Path(filepath)
        filename = filepath.stem
        file_start, file_end = filename.split("-")
        core_start = float(file_start)
        # Convert SB depth to real depth
        df_out["SB DEPTH [m]"] = core_start + df_out["SB DEPTH [m] rel"]
        df_out = df_out.drop("SB DEPTH [m] rel", axis=1) # Drop old SB depth column

    if "SECT DEPTH [cm]" in df_out.columns:
        df_out["SECT DEPTH [m]"] = df_out["SECT DEPTH [cm]"] / 100 # SECT depth correct unit
        df_out = df_out.drop("SECT DEPTH [cm]", axis=1) # Drop old SECT depth column

    # Fixed and formatted columns rename
    df_out = df_out[["SB DEPTH [m]", "SECT NUM", "SECT DEPTH [m]", "CT [cm]", "PW Amp",
                     "PW Vel [m/s]", "Den [kg/m3]", "Imp [kg/m2s]", "FP"]]

    return df_out
# Combine out files
def combine_out_dfs(filepaths):
    """combine_out_dfs will combine the list of dataframes from all .out files into one dataframe.
    Args:
        filepaths (list): list of filepaths to .out files
    Returns:
        df_all_out (dataframe): combined dataframe containing all out measurements, sorted by depth.
    """
    all_out_dfs = []
    for file in filepaths:
        df = read_format_out_file(file)
        df["file"] = file.name
        all_out_dfs.append(df)

    df_all = pd.concat(all_out_dfs, ignore_index=True)
    # Remove rows where depth is not numeric (= column name rows)
    df_all_out = df_all[pd.to_numeric(df_all["SB DEPTH [m]"], errors="coerce").notna()]
    df_all_out = df_all_out.sort_values("SB DEPTH [m]")
    df_all_out = df_all_out.reset_index(drop=True)

    return df_all_out
# Read manual measurement excel file
def MM_assign_core_ids(series):
    """Create IDs that increment whenever values decrease. Used to distinguish in column when new core begins.
    """
    return (series.diff() < 0).cumsum()
# Function to clean numeric columns (for CT, PWVel, Den)
def MM_clean_numeric(series, dtype=float):
    """MM_clean_numeric returns a numeric version of the series.
    """
    return (pd.to_numeric(series, errors="coerce").dropna().astype(dtype).reset_index(drop=True))
# Main function to process manual measurement excel file
def process_MM(raw):
    """process_MM goes through all necessary steps of transforming the raw excel data into a correctly formatted dataframe to combine with .out corelogger data.
    Args:
        raw (dataframe): Contains the manual measurements from an excel file
    Returns:
        MM_df_end (dataframe): Contains the processed manual measurements with correct format and units
    """
    # 1. Step: Extraxt origin depths from 1st column of raw data
    OG_depth_nonan = raw.iloc[:,0].dropna()
    OG_depth_nonan = OG_depth_nonan[~OG_depth_nonan.str.contains("Core", na=False)] # remove row containing "Core"
    OG_depth = OG_depth_nonan.str.split("-").str[0] # split string at "-" and take the first part (the start depth of core)
    OG_depth = OG_depth.astype(float)

    # 2. Step: Extract position along core from 3rd column of raw data
    pos_cm = raw.iloc[:,2].dropna() # convert from cm to m
    pos_cm = pos_cm[~pos_cm.str.contains("Position", na=False)] # remove row containing word "Position"
    pos_cm = pos_cm[~pos_cm.str.contains("cm", na=False)] # remove row containing word "cm"
    pos_m = pos_cm.astype(float) / 100

    # 3. Step: Combine origin depths with their positions along core
    # add ID column to both OG_depth and pos_m to be able to merge them
    OG_depth_df = pd.DataFrame({"ID": range(len(OG_depth)), "OG_DEPTH": OG_depth})
    # pos_m grouped in intervals 0-1
    pos_m_df = pd.DataFrame({"POS_M": pos_m})
    pos_m_df["ID"] = MM_assign_core_ids(pos_m_df["POS_M"])

    # 4. Step: Extract section number
    piece_nr = raw.iloc[:,1]
    piece_nr = pd.to_numeric(piece_nr, errors="coerce") # numeric conversion maybe also good for other columns?
    piece_nr = piece_nr.dropna().astype(int)
    piece_nr_df = pd.DataFrame({"ID": np.zeros(len(piece_nr), dtype=int), "SECT NUM": piece_nr})
    piece_nr_df["ID"] = MM_assign_core_ids(piece_nr_df["SECT NUM"])
    piece_nr_df.reset_index(drop=True, inplace=True)
    # merge dataframes on ID
    merged_df = pd.merge(OG_depth_df, pos_m_df, on="ID", how="inner")
    merged_df["SB DEPTH [m]"] = merged_df["OG_DEPTH"] + merged_df["POS_M"]
    merged_df2 = pd.concat([merged_df.reset_index(drop=True), piece_nr_df["SECT NUM"].reset_index(drop=True)], axis=1)

    # 5. Step: Extract CT values
    ct = raw.iloc[:,3]
    ct = MM_clean_numeric(raw.iloc[:,3])
    ct_df = pd.DataFrame({"CT [cm]": ct})
    merged_df3 = pd.concat([merged_df2.reset_index(drop=True), ct_df], axis=1)

    # 6. Step: Extract PWVel, Den and Imp
    # first search for column with the word density in raw
    col_den_index = next(i for i, col in enumerate(raw.columns) if (raw[col] == "Density").any())
    col_pwvel_index = col_den_index + 1 # assuming PWVel is in the column right after density
    den = MM_clean_numeric(raw.iloc[:,col_den_index])
    den = den*1000 # convert from g/cm3 to kg/m3
    pwvel = MM_clean_numeric(raw.iloc[:,col_pwvel_index])
    imp = den * pwvel
    trinity_df = pd.DataFrame({"Den1 [kg/m3]": round(den,0), "PWVel [m/s]": round(pwvel,1), "Imp [kg/m2s]": round(imp,0)})
    merged_df4 = pd.concat([merged_df3.reset_index(drop=True), trinity_df], axis=1)

    # Final step: create final dataframe with all columns in the right order and fill in empty columns with NaN
    MM_df_end = pd.DataFrame({
        "SB DEPTH [m]": merged_df4["SB DEPTH [m]"],
        "SECT NUM": merged_df4["SECT NUM"],
        "SECT DEPTH [m]": merged_df4["POS_M"],
        "CT [cm]": merged_df4["CT [cm]"],
        "PW Amp": np.nan,
        "PW Vel [m/s]": merged_df4["PWVel [m/s]"],
        "Den [kg/m3]": merged_df4["Den1 [kg/m3]"],
        "Imp [kg/m2s]": merged_df4["Imp [kg/m2s]"],
        "FP": np.nan
    })

    return MM_df_end
# Combine all the data into one df
def combine_dfs(df1, df2):
    """combine_dfs will combine the dataframes from .out files and manual 
    measurements into one dataframe.
    Args:
        df1 (dataframe): dataframe containing data from .out files
        df2 (dataframe): dataframe containing manual measurements
    Returns:
        df_combined (dataframe): combined dataframe
    """
    df_list = [df1, df2]
    df_combined = pd.concat(df_list, ignore_index=True)
    df_combined = df_combined.sort_values("SB DEPTH [m]")
    df_combined = df_combined[pd.to_numeric(df_combined["SB DEPTH [m]"], errors="coerce").notna()]
    df_combined = df_combined.reset_index(drop=True)
    # Save as csv (optional)
    # path_df_final = r"C:..."
    # df_combined.to_csv(path_df_final, index=False)

    return df_combined

# --- Functions for PART 2: Plotting ---
# Load core images
def load_core_image(img_start, img_stop):
    """load_core_image loads pre-cut and renamed core images saved in a seperate image folder.
    Args:
        img_start (int): start of depth interval
        img_stop (int): end of depth interval
    Returns:
        image: core image
    """
    image_filename = f"{int(img_start)}-{int(img_stop)}.JPG"
    image_path = Path(Images) / image_filename

    if image_path.exists():
        img = Image.open(image_path)

        # Apply EXIF orientation correctly
        img = ImageOps.exif_transpose(img)

        # Convert to numpy array
        return np.array(img)

    raise FileNotFoundError(f"Could not find image: {image_path}\n")
# Depth intervals
def get_interval(filepath):
    """get_interval isolates the depth intervals from .out filenames
    Args:
        filepath (string): filepath to out file
    Returns:
        floats: start and end float seperated by ","
    """
    interval_start, interval_end = filepath.stem.split("-")

    return float(interval_start), float(interval_end)
# Merge intervals
def merge_intervals(interval_input, gap_threshold=0.5):
    """merge_intervals combines depth intervals that are close by to have less broken axes.
    Args: 
        interval_input (tuple): list of interval tuples
    Return:
        merged (tuple): merged intervals
    """
    bax_intervals = sorted(interval_input, key=lambda x: x[0])
    merged = []
    current_start, current_end = bax_intervals[0]
    for i_start, i_end in bax_intervals[1:]:
        gap = i_start - current_end

        if gap <= gap_threshold:
            # merge
            current_end = max(current_end, i_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = i_start, i_end
    merged.append((current_start, current_end))

    return merged
# First plot (depth section with core image)
def plot_section(df, section_start, section_end):
    """function for plot of specified depth section with corresponding core image.
    Args:
        df (dataframe): final dataframe
        section_start (float): start of depth of core
        section_end (float): end of depth of core
    Returns:
        fig1: plot figure
    """
    df_den = df[(df["Den [kg/m3]"] >= den_min) & (df["Den [kg/m3]"] <= den_max)]
    df_vel = df[(df["PW Vel [m/s]"] >= vel_min) & (df["PW Vel [m/s]"] <= vel_max)]
    df_imp = df[(df["Imp [kg/m2s]"] >= den_min*vel_min) & (df["Imp [kg/m2s]"] <= den_max*vel_max)]
    mask = (df["SB DEPTH [m]"] >= section_start) & (df["SB DEPTH [m]"] <= section_end)
    fig1 = plt.figure(figsize=(10, 6))
    fig1.suptitle(f"Core Section: {section_start}-{section_end} m", fontsize=14)
    # Change last value if image is shown distorted
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.4], wspace=0.3)

    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig1.add_subplot(gs[0, 2], sharey=ax1)
    ax4 = fig1.add_subplot(gs[0, 3]) # For core image
    # Hide y ticks + labels on middle and right plots
    for a in [ax2, ax3]:
        a.tick_params(axis='y', left=False, labelleft=False)

    x_labels = ["Den [kg/m3]", "PW Vel [m/s]", "Imp [kg/m2s]"]
    titles = ["Density", "P-wave Velocity", "Impedance"]

    ax1.scatter(df_den.loc[mask, "Den [kg/m3]"], df_den.loc[mask, "SB DEPTH [m]"], color="lightcoral")
    ax1.plot(df_den.loc[mask, "Den [kg/m3]"], df_den.loc[mask, "SB DEPTH [m]"], color="lightcoral")
    ax1.set_ylabel("Depth (m)")
    ax2.scatter(df_vel.loc[mask, "PW Vel [m/s]"], df_vel.loc[mask, "SB DEPTH [m]"], color="skyblue")
    ax2.plot(df_vel.loc[mask, "PW Vel [m/s]"], df_vel.loc[mask, "SB DEPTH [m]"], color="skyblue")
    ax3.scatter(df_imp.loc[mask, "Imp [kg/m2s]"], df_imp.loc[mask, "SB DEPTH [m]"], color="lightgreen")
    ax3.plot(df_imp.loc[mask, "Imp [kg/m2s]"], df_imp.loc[mask, "SB DEPTH [m]"], color="lightgreen")

    for ax_plt in [ax1, ax2, ax3]:
        ax_plt.set_xlabel(x_labels[[ax1, ax2, ax3].index(ax_plt)])
        ax_plt.set_title(titles[[ax1, ax2, ax3].index(ax_plt)])
        ax_plt.set_ylim(section_start - 0.05, section_end + 0.05)
        ax_plt.grid()

    # Image axis (no shared y)
    try:
        core_img = load_core_image(section_start, section_end)
    except FileNotFoundError:
        core_img = None
        print(f"No core image for interval {section_start}-{section_end}")
    if core_img is not None:
        ax4.imshow(np.rot90(core_img, k=3), aspect='auto')
        ax4.axis('off')
        ax4.set_title("Core Image")
    else:
        fig1.delaxes(ax4) # Removes axis if no image
    ax1.invert_yaxis()
    return fig1
#Function that performs the coresection + image plot function for all cores and saves the output plots in a seperate folder
def save_all_core_section_plots(df, output_filepath):
    """save_all_core_section_plots goes through image folder and creates the core section plot + image for all cores.
    Plots are saved (without being opened in a window) in the specified output folder.
    Args:
        df (_type_): _description_
    """
    output_folder = output_filepath
    output_folder.mkdir(exist_ok=True)
    image_files = sorted(Path(Images).glob("*.JPG"))

    for image_file in image_files:
        section_start, section_end = get_interval(image_file)
        try:
            fig = plot_section(df, section_start, section_end)
            outfile = output_folder / f"{image_file.stem}.jpg"
            fig.savefig(outfile, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {outfile.name}")
        except Exception as e:
            print(f"Could not create plot for {image_file.stem}: {e}")
# Second plot (full depth section with broken axes)
def plot_all_sections(df):
    """Function for plotting data over entire depth range with the use 
    of broken axes (segments on depth axis).
    Args:
        df (dataframe): final dataframe
    Returns:
        fig2: plot figure
    """
    intervals = [get_interval(f) for f in out_files]
    manual_intervals = {
        (math.floor(x), math.floor(x) + 1)
        for x in df["SB DEPTH [m]"]
        if pd.notna(x)}
    all_intervals = list(set(intervals) | manual_intervals)
    all_intervals.sort(key=lambda x: x[0])

    # For better visibility --> merging close-by segments
    merged_intervals = merge_intervals(all_intervals, gap_threshold=0.5)
    depths = []
    pwvelocity = []
    density = []
    impedance = []
    cl_colors = []

    for start, end in merged_intervals:
        # Mask by depth first
        mask = (df["SB DEPTH [m]"] >= start) & (df["SB DEPTH [m]"] <= end)
        d_depths = df["SB DEPTH [m]"][mask].values
        d_pwvel = np.ma.masked_invalid(df["PW Vel [m/s]"][mask].values)
        d_density = np.ma.masked_invalid(df["Den [kg/m3]"][mask].values)
        d_imp = np.ma.masked_invalid(df["Imp [kg/m2s]"][mask].values)
        # Remove NaNs
        valid_mask = (~np.isnan(d_density)) & (~np.isnan(d_pwvel)) & (~np.isnan(d_imp))
        # Combined with filter mask
        combined_mask = (d_density >= den_min) & (d_density <= den_max) & \
                        (d_pwvel >= vel_min) & (d_pwvel <= vel_max) & valid_mask
        # Apply mask to all lists
        filtered_df = df[mask].loc[combined_mask]

        depths.append(filtered_df["SB DEPTH [m]"].values)
        pwvelocity.append(filtered_df["PW Vel [m/s]"].values)
        density.append(filtered_df["Den [kg/m3]"].values)
        impedance.append(filtered_df["Imp [kg/m2s]"].values)

        colors = filtered_df["centroid"].map(cluster_colors)
        cl_colors.append(colors)

    reversed_plot_intervals = merged_intervals[::-1]  # Reverse for plotting from top to bottom
    fig2 = plt.figure(figsize=(12, 9))
    gs = GridSpec(1, 3, figure=fig2, wspace=0.3)

    # Plot configuration (data, xlabel, title, color)
    plot_config = [
        (density, "Density [kg/m3]", "Density"),
        (pwvelocity, "P-wave Velocity [m/s]", "P-wave Velocity"),
        (impedance, "Impedance [kg/m2s]", "Impedance")]

    for i, (data, xlabel, title) in enumerate(plot_config):
        bax = brokenaxes(
            ylims=reversed_plot_intervals,
            hspace=0.4,
            d=0.005,
            fig=fig2,
            despine=False,
            subplot_spec=gs[i])

        for z_segment, d_segment, colors in zip(depths,data, cl_colors):
            bax.scatter(d_segment, z_segment, c=colors, s=7)

        bax.invert_yaxis()
        bax.set_xlabel(xlabel, labelpad=20)
        bax.set_title(title)
        bax.grid(True, linestyle=':', alpha=0.6)
        handles = [Patch(facecolor=cluster_colors[label], label=f"Cluster {label}") for label in cluster_labels]
        fig2.legend(handles=handles, loc="upper right", title="Clusters")

    return fig2
# Third plot (crossplot density pwvelocity)
def plot_crossplot(df):
    """Function to plot crossplot of P-wave velocity and density 
    with an adjustable zoomed-in section (# for now).
    Args:
        df (dataframe): final dataframe
    Returns:
        fig3: plot figure
    """
    df_xplot = df.loc[density_data_range].copy()
    x = df_xplot["Den [kg/m3]"]
    y = df_xplot["PW Vel [m/s]"]

    fig3 = plt.figure(figsize=(7,5), layout="constrained")
    plt.scatter(
        x, y,
        c=df_xplot.loc[density_data_range, "SB DEPTH [m]"],
        cmap="viridis_r", marker='o', s=8)
    plt.title("P-wave Velocity vs Density")
    plt.ylabel("P-wave Velocity (m/s)")
    plt.xlabel("Density (kg/m^3)")
    plt.ylim(vel_min, vel_max)
    plt.xlim(den_min, den_max)
    plt.grid(alpha=0.2)
    plt.colorbar(label="Depth (m)")
    return fig3

# --- Functions for PART 3: Cluster Analysis ---
# Create cluster dataframe
def df_cluster_prep(df_plot):
    """df_cluster_prep prepares the dataframe for cluster analysis by 
    selecting relevant columns and applying data filters.
    Args:
        df_plot (dataframe): final dataframe
    Returns:
        df_cluster (dataframe): filtered dataframe for cluster analysis
    """
    # Select relevant columns for cluster analysis
    df = pd.DataFrame({
    'Density': df_plot['Den [kg/m3]'],
    'P-wave Velocity': df_plot['PW Vel [m/s]']})

    # Drop rows with any NaNs
    df.dropna(subset=['Density'], inplace=True)
    df.dropna(subset=['P-wave Velocity'], inplace=True)
    #df = df.reset_index(drop=True)

    # Apply data filters if needed
    df = df[(df["Density"] >= den_min) & (df["Density"] <= den_max)]
    df = df[(df["P-wave Velocity"] >= vel_min) & (df["P-wave Velocity"] <= vel_max)]

    return df
# Determine cluster number (with elbow diagram)
def determine_cluster_number(df_for_clustering, plot=False):
    """determine_cluster_number uses kmeans elbow diagram method to determine the 
    optimal number of clusters for given dataset.
    Args:
        df_for_clustering (dataframe): dataframe containing only p-wave velocity and density columns
        plot (bool, optional): Option to plot the elbow diagram. Defaults to False.
    Returns:
        optimal_elbow_k (int): optimal number of cluster
    """
    err_total = []
    n = 10

    df_elbow = df_for_clustering[['Density','P-wave Velocity']]

    for k in range(1, n+1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(df_elbow)
        err_total.append(model.inertia_)

    if plot:
        plt.figure()
        plt.plot(range(1, n+1), err_total, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Total error')

    # Elbow diagram process
    x = np.arange(1, n+1)
    y = np.array(err_total)
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)

    distances = []
    for i in range(len(x)):
        point_vec = np.array([x[i] - x[0], y[i] - y[0]])
        proj_length = np.dot(point_vec, line_vec)
        proj_point = proj_length * line_vec
        dist = np.linalg.norm(point_vec - proj_point)
        distances.append(dist)
    optimal_elbow_k = x[np.argmax(distances)]

    return optimal_elbow_k
# Plot cluster and defs
def plot_clusters(df, k):
    """plot_clusters encompasses code for plotting final clusters with cluster 
    defining ranges in a table next to the plot.
    Args:
        df (dataframe): cluster dataframe
        k (int): determined number of clusters
    Returns:
        fig4 (figure): plot and table
    """
    fig4, (ax, ax_table) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})

    # Scatter plot
    ax.scatter(df['Density'], df['P-wave Velocity'], c=df['centroid'].map(cluster_colors), s=20, alpha=0.9)
    ax.scatter(centroids['Density'], centroids['P-wave Velocity'], marker='s', s=100, c=[cluster_colors[i] for i in range(k)], edgecolor='black', label='Centroids')
    ax.set_xlabel('Density [kg/m3]', fontsize=14)
    ax.set_ylabel('P-wave Velocity [m/s]', fontsize=14)
    ax.grid(alpha=0.2)

    # Legend
    handles = [
        Patch(color=plt.get_cmap("tab20b", k)(i), label=f'Cluster {i}')
        for i in range(k)]
    ax.legend(handles=handles)

    # Cluster definitions/ranges
    cluster_defs = (
        df.groupby('centroid')
        .agg({
            'Density': ['min', 'max'],
            'P-wave Velocity': ['min', 'max']})).round(1)

    # Column names
    cluster_defs.columns = ['Den min', 'Den max', 'Vp min', 'Vp max']
    ax_table.axis('off')

    table = ax_table.table(
        cellText=cluster_defs.values,
        colLabels=cluster_defs.columns,
        rowLabels=[f'C{i}' for i in cluster_defs.index],
        loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)

    # Table coloring
    for i in range(len(cluster_defs)):
        base_color = cmap(i)
        color_with_alpha = (*base_color[:3], 0.7)
        for j in range(len(cluster_defs.columns)):
            table[(i+1, j)].set_facecolor(color_with_alpha)
        table[(i+1, -1)].set_facecolor(color_with_alpha)

    for i in range(len(cluster_defs)):
        base_color = cmap(i)
        color_with_alpha = (*base_color[:3], 0.7)
        table[(i+1, -1)].set_facecolor(color_with_alpha)

    return fig4

# ---------------------------------------------------------------------------------
# Output display setting
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
# Input window
def input_window():
    """input_window creates a configuration window where the user can fill in all necessary file/folder paths and settings for running the code.

    Returns:
        Entered variables/values
    """
    root = tk.Tk()
    root.title("CoreLogger Setup")
    root.geometry("500x750")
    root.resizable(True, True)

    # Variables
    bohrung_path = tk.StringVar()
    excel_path = tk.StringVar()
    images_path = tk.StringVar()
    sectionplot_start = tk.StringVar()
    sectionplot_end = tk.StringVar()
    den_min_var = tk.StringVar()
    den_max_var = tk.StringVar()
    vel_min_var = tk.StringVar()
    vel_max_var = tk.StringVar()
    procedure_k = tk.StringVar(value="automatic")
    k_value = tk.StringVar()
    plot_section_var = tk.BooleanVar(value=True)
    plot_all_sections_var = tk.BooleanVar(value=True)
    plot_crossplot_var = tk.BooleanVar(value=True)
    plot_cluster_var = tk.BooleanVar(value=True)
    save_all_section_plots_var = tk.BooleanVar(value=False)
    save_plots_path = tk.StringVar()

    # Functions
    def select_bohrung():
        path = filedialog.askdirectory(
            title="Select folder containing core measurements")
        if path:
            bohrung_path.set(path)
    def select_excel():
        path = filedialog.askopenfilename(
            title="Select Excel file with manual measurements",
            filetypes=[("Excel files", "*.xlsx")])
        if path:
            excel_path.set(path)
    def select_images():
        path = filedialog.askdirectory(
            title="Select folder containing core images")
        if path:
            images_path.set(path)
    def select_save_folder():
        path = filedialog.askdirectory(
            title="Select output folder for core section plot JPGs")
        if path:
            save_plots_path.set(path)
    def toggle_k_entry(*args):
        if procedure_k.get() == "manual":
            k_entry.config(state="normal")
        else:
            k_entry.config(state="disabled")
    def toggle_section_settings(*args):
        if plot_section_var.get():
            section_frame.pack(fill="x", pady=5)
        else:
            section_frame.pack_forget()
    def toggle_cluster_settings(*args):
        if plot_cluster_var.get():
            cluster_frame.pack(fill="x", pady=5)
        else:
            cluster_frame.pack_forget()
    def toggle_save_folder(*args):
        if save_all_section_plots_var.get():
            browse_save_button.config(state="normal")
        else:
            browse_save_button.config(state="disabled")
            save_plots_path.set("")

    def submit():
        root.destroy()

    procedure_k.trace_add("write", toggle_k_entry)
    plot_section_var.trace_add("write", toggle_section_settings)
    plot_cluster_var.trace_add("write", toggle_cluster_settings)
    save_all_section_plots_var.trace_add("write", toggle_save_folder)

    # Main Frame
    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)

    # Input Files
    files_frame = ttk.LabelFrame(main, text="Input Files", padding=10)
    files_frame.pack(fill="x", pady=5)
    row = ttk.Frame(files_frame)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Core measurements (folder)", width=30).pack(side="left")
    ttk.Button(row, text="Browse", command=select_bohrung).pack(side="left")
    ttk.Label(row, textvariable=bohrung_path,
              foreground="gray").pack(side="left", padx=10)
    row = ttk.Frame(files_frame)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Manual measurements (Excel file)", width=30).pack(side="left")
    ttk.Button(row, text="Browse", command=select_excel).pack(side="left")
    ttk.Label(row, textvariable=excel_path,
              foreground="gray").pack(side="left", padx=10)
    row = ttk.Frame(files_frame)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Core images (folder)", width=30).pack(side="left")
    ttk.Button(row, text="Browse", command=select_images).pack(side="left")
    ttk.Label(row, textvariable=images_path,
              foreground="gray").pack(side="left", padx=10)

    # Plot Selection
    plots_frame = ttk.LabelFrame(main, text="Plots", padding=10)
    plots_frame.pack(fill="x", pady=5)

    ttk.Checkbutton(
        plots_frame,
        text="Specific depth section plot with core image",
        variable=plot_section_var).pack(anchor="w")

    ttk.Checkbutton(
        plots_frame,
        text="Plot of all core measurements",
        variable=plot_all_sections_var).pack(anchor="w")

    ttk.Checkbutton(
        plots_frame,
        text="Density-Velocity crossplot",
        variable=plot_crossplot_var).pack(anchor="w")

    ttk.Checkbutton(
        plots_frame,
        text="Cluster plot",
        variable=plot_cluster_var).pack(anchor="w")

    row = ttk.Frame(plots_frame)
    row.pack(fill="x", pady=2)

    ttk.Checkbutton(
        row,
        text="Save all section plots as JPGs (in a folder)",
        variable=save_all_section_plots_var).pack(side="left")

    browse_save_button = ttk.Button(
        row,
        text="Browse",
        command=select_save_folder,
        state="disabled")
    browse_save_button.pack(side="left", padx=(15, 5))

    ttk.Label(
        row,
        textvariable=save_plots_path,
        foreground="gray").pack(side="left")

    # Section Settings
    section_frame = ttk.LabelFrame(
        main,
        text="Core Section Depth",
        padding=10)
    section_frame.pack(fill="x", pady=5)

    row = ttk.Frame(section_frame)
    row.pack(fill="x", pady=2)

    ttk.Label(row, text="Start depth (m)", width=20).pack(side="left")
    ttk.Entry(
        row,
        textvariable=sectionplot_start,
        width=12).pack(side="left")

    row = ttk.Frame(section_frame)
    row.pack(fill="x", pady=2)

    ttk.Label(row, text="End depth (m)", width=20).pack(side="left")
    ttk.Entry(
        row,
        textvariable=sectionplot_end,
        width=12).pack(side="left")

    # Clustering
    cluster_frame = ttk.LabelFrame(main, text="Clustering", padding=10)
    cluster_frame.pack(fill="x", pady=5)

    ttk.Radiobutton(
        cluster_frame,
        text="Automatic",
        variable=procedure_k,
        value="automatic").pack(anchor="w")

    ttk.Radiobutton(
        cluster_frame,
        text="Manual",
        variable=procedure_k,
        value="manual").pack(anchor="w")

    row = ttk.Frame(cluster_frame)
    row.pack(fill="x", pady=5)

    ttk.Label(row, text="Number of clusters (k)", width=20).pack(side="left")

    k_entry = ttk.Entry(
        row,
        textvariable=k_value,
        width=10,
        state="disabled")
    k_entry.pack(side="left")

    # Data filters
    filter_frame = ttk.LabelFrame(main, text="Data Filters", padding=10)
    filter_frame.pack(fill="x", pady=5)

    row = ttk.Frame(filter_frame)
    row.pack(fill="x", pady=2)

    ttk.Label(row, text="Density Min", width=20).pack(side="left")
    ttk.Entry(
        row,
        textvariable=den_min_var,
        width=12).pack(side="left")

    row = ttk.Frame(filter_frame)
    row.pack(fill="x", pady=2)

    ttk.Label(row, text="Density Max", width=20).pack(side="left")
    ttk.Entry(
        row,
        textvariable=den_max_var,
        width=12).pack(side="left")
    row = ttk.Frame(filter_frame)
    row.pack(fill="x", pady=2)

    ttk.Label(row, text="P-wave Velocity Min", width=20).pack(side="left")
    ttk.Entry(
        row,
        textvariable=vel_min_var,
        width=12).pack(side="left")

    row = ttk.Frame(filter_frame)
    row.pack(fill="x", pady=2)

    ttk.Label(row, text="P-wave Velocity Max", width=20).pack(side="left")
    ttk.Entry(
        row,
        textvariable=vel_max_var,
        width=12).pack(side="left")


    # Run Button
    ttk.Button(
        main,
        text="Run Analysis",
        command=submit
    ).pack(pady=15)

    root.mainloop()

    return {
        "Bohrung": Path(bohrung_path.get()),
        "out_files": list(Path(bohrung_path.get()).rglob("*.out")),
        "mm_file": Path(excel_path.get()),
        "Images": Path(images_path.get()),
        "procedure_k": procedure_k.get(),
        "optimal_k": int(k_value.get())
            if procedure_k.get() == "manual" and k_value.get()
            else None,
        "section_start": float(sectionplot_start.get())
            if sectionplot_start.get()
            else None,
        "section_end": float(sectionplot_end.get())
            if sectionplot_end.get()
            else None,
        "vel_min": float(vel_min_var.get())
            if vel_min_var.get()
            else None,
        "vel_max": float(vel_max_var.get())
            if vel_max_var.get()
            else None,
        "den_min": float(den_min_var.get())
            if den_min_var.get()
            else None,
        "den_max": float(den_max_var.get())
            if den_max_var.get()
            else None,
        "plot_section": plot_section_var.get(),
        "plot_all_sections": plot_all_sections_var.get(),
        "plot_crossplot": plot_crossplot_var.get(),
        "plot_clusters": plot_cluster_var.get(),
        "save_all_section_plots": save_all_section_plots_var.get(),
        "save_plots_folder": Path(save_plots_path.get()) if save_plots_path.get() else None}

# Renaming variables entered in configuration window
config = input_window()
Bohrung = config["Bohrung"]
out_files = config["out_files"]
mm_file = config["mm_file"]
Images = config["Images"]
procedure_k = config["procedure_k"]
optimal_k = config["optimal_k"]
core_section_start = config["section_start"]
core_section_end = config["section_end"]
vel_min = config["vel_min"]
vel_max = config["vel_max"]
den_min = config["den_min"]
den_max = config["den_max"]
# ---------------------------------------------------------------------------------
# Data preparation --> setting up the dataframes
df_out_running = combine_out_dfs(out_files)
df_mm_running = process_MM(pd.read_excel(mm_file)) #, sheet_name="Data"
df_final = combine_dfs(df_out_running, df_mm_running)

# ---------------------------------------------------------------------------------
# Data filters
df_final = df_final[df_final["PW Amp"] > 30]
#print("df_final:\n", df_final)
# df_final.to_csv(r"")

# If no filters given in configuration window, default to entire dataset range (+- buffer for better visibility)
den_min = config["den_min"] if config["den_min"] is not None else df_final["Den [kg/m3]"].min() - 100
den_max = config["den_max"] if config["den_max"] is not None else df_final["Den [kg/m3]"].max() + 100
vel_min = config["vel_min"] if config["vel_min"] is not None else df_final["PW Vel [m/s]"].min() - 100
vel_max = config["vel_max"] if config["vel_max"] is not None else df_final["PW Vel [m/s]"].max() + 100

# Define data ranges (doesn't delete outliers)
density_data_range = (df_final["Den [kg/m3]"] >= den_min) & (df_final["Den [kg/m3]"] <= den_max)
velocity_data_range = (df_final["PW Vel [m/s]"] >= vel_min) & (df_final["PW Vel [m/s]"] <= vel_max)
# ---------------------------------------------------------------------------------
# Cluster Analysis
# Define dataframe
df_cluster = df_cluster_prep(df_final)
# Determine optimal number of clusters
if procedure_k == "automatic":
    optimal_k = determine_cluster_number(df_cluster, plot=False)
    print(f"Optimal number of clusters determined by elbow method: k = {optimal_k}")

# Kmeans clustering algorithm
model_k = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
df_cl = df_cluster[['Density','P-wave Velocity']].copy()
df_cl_log = np.log10(df_cl)
model_k.fit(df_cl_log)
# Assign clusters
df_cl['centroid'] = model_k.labels_
# df_cl_log['centroid'] = model_k.labels_
# Get centroids
centroids_log = model_k.cluster_centers_
centroids = 10**centroids_log   # back to normal scale
#centroids = centroids_log


centroids = pd.DataFrame(centroids, columns=['Density','P-wave Velocity'])
#print(f"Dataframe for clustering:\n{df_cl}\n")

# Color scheme for the clusters
cluster_labels = sorted(df_cl['centroid'].unique())
# cluster_labels = sorted(df_cl_log['centroid'].unique())

cluster_map = {label: i for i, label in enumerate(cluster_labels)}
cmap = plt.get_cmap("tab20b", len(cluster_labels))
cluster_colors = {label: cmap(i) for label, i in cluster_map.items()}
# Special df for plot_all_sections with cluster colors
df_color = df_final.copy()
df_color = df_color.merge(df_cl[['centroid']], left_index=True, right_index=True, how='left')
#df_color = df_color.merge(df_cl_log[['centroid']], left_index=True, right_index=True, how='left')
# ---------------------------------------------------------------------------------
# Plotting the selected plots
if config["plot_section"]:
    plot_section(
        df_final,
        config["section_start"],
        config["section_end"])

if config["plot_all_sections"]:
    plot_all_sections(df_color)

if config["plot_crossplot"]:
    plot_crossplot(df_final)
if config["plot_clusters"]:
    plot_clusters(df_cl, optimal_k)
    #plot_clusters(df_cl_log, optimal_k)
if config["save_all_section_plots"]:
    save_all_core_section_plots(df_final, output_filepath=config["save_plots_folder"])

plt.show()
# The End
