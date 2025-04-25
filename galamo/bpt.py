import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from termcolor import colored


# Default BPT diagram
def draw(input_file, map="default", save_figure=False, output_filename="BPT_diagram.pdf"):
    """
    Draws the BPT diagram based on the input CSV file containing galaxy spectral data.
    
    Parameters:
    - input_file (str): Path to the CSV file.
    - map (str): Type of map to use for visualization ('default', 'bubble', 'soft'). Default is 'default'.
    - save_figure (bool, optional): Whether to save the figure. Default is False.
    - output_filename (str, optional): Filename to save the figure. Default is "BPT_diagram.pdf".
    """
    required_columns = ['h_alpha_flux', 'h_beta_flux', 'oiii_5007_flux', 'nii_6584_flux',
                        'oi_6300_flux', 'sii_6717_flux', 'sii_6731_flux']
    
    # Read the data
    df = pd.read_csv(input_file)
    actual_columns = df.columns.tolist()

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in actual_columns]
    
    if missing_columns:
        print(colored(f"Missing columns: {', '.join(missing_columns)}", "red"))
        print(colored(f"❌ COLUMN ERROR: Ensure your data contains the required columns.\n\nColumns must look like: {required_columns}", "red"))
        print(colored("⚠️ Check SDSS format or documentation @ www.galamo.org", "yellow"))
        exit()
    else:
        print(colored("✅ Columns matched", "green"))

    # Define classification lines
    x = np.linspace(-2, 0, 20)
    y = (0.61/(x-0.05)) + 1.3
    x1 = np.linspace(-2, 0.4, 20)
    y1 = (0.61/(x1-0.47))+1.19
    x2 = np.linspace(-2, 0.1, 20)
    y2 = (0.72/(x2 -0.32))+1.3
    sx1 = np.linspace(-0.3, 1, 20)
    sy1 = 1.89 * sx1 + 0.76
    x3 = np.linspace(-2.5, -0.75, 20)
    y3 = (0.73/(x3+0.59))+1.33
    sx2 = np.linspace(-1.1, 0, 20)
    sy2 = 1.18 * sx2 + 1.30

    # Apply selection criteria
    mask_1 = 0.73/(np.log10(df['oi_6300_flux']/df['h_alpha_flux']) + 0.59) + 1.33 < np.log10(df['oiii_5007_flux']/df['h_beta_flux'])
    agn2_1 = df[mask_1]
    mask_2 = 0.61/(np.log10(agn2_1['nii_6584_flux']/agn2_1['h_alpha_flux']) - 0.47) + 1.19 < np.log10(agn2_1['oiii_5007_flux']/agn2_1['h_beta_flux'])
    agn2_2 = agn2_1[mask_2]
    mask_3 = 0.72/(np.log10((agn2_2['sii_6717_flux'] + agn2_2['sii_6731_flux'])/agn2_2['h_alpha_flux']) - 0.32) + 1.33 < np.log10(agn2_2['oiii_5007_flux']/agn2_2['h_beta_flux'])
    agn2 = agn2_2[mask_3]

    # Classify as Seyfert or LINER
    mask_liner = np.logical_or(
        1.18 * np.log10(agn2['oi_6300_flux']/agn2['h_alpha_flux']) + 1.3 > np.log10(agn2['oiii_5007_flux']/agn2['h_beta_flux']),
        1.89 * np.log10((agn2['sii_6717_flux'] + agn2['sii_6731_flux'])/agn2['h_alpha_flux']) + 0.76 > np.log10(agn2['oiii_5007_flux']/agn2['h_beta_flux'])
    )
    liner = agn2[mask_liner]
    seyf = agn2[~mask_liner]

    
    
    # Plot BPT diagrams based on selected map
    if map == "default":
        plot_default(df, seyf, liner, x, y, x1, y1, x2, y2, sx1, sy1, x3, y3, sx2, sy2)
    elif map == "bubble":
        plot_bubble(df, seyf, x, y, x1, y1, x2, y2, sx1, sy1, x3, y3, sx2, sy2)
    elif map == "soft":
        plot_soft(df, seyf, x, y, x1, y1, x2, y2, sx1, sy1, x3, y3, sx2, sy2)
    else:
        print(colored("⚠️ Invalid map type. Choose from 'default', 'bubble', or 'soft'.", "yellow"))

    # Save figure if needed
    if save_figure:
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, transparent=True)
    
    plt.show()

# Default plot
def plot_default(df, seyf, liner, x, y, x1, y1, x2, y2, sx1, sy1, x3, y3, sx2, sy2):
    plt.figure(figsize=(15,5))

    # NII
    ax1 = plt.subplot(131)
    ax1.scatter(np.log10(df['nii_6584_flux']/df['h_alpha_flux']), np.log10(df['oiii_5007_flux']/df['h_beta_flux']), alpha =0.1, marker='x', color='k')
    ax1.scatter(np.log10(seyf['nii_6584_flux']/seyf['h_alpha_flux']), np.log10(seyf['oiii_5007_flux']/seyf['h_beta_flux']), alpha =0.5, marker='o', color='r')
    ax1.plot(x, y, c='k', linestyle='solid')
    ax1.plot(x1, y1, c='k', linestyle='dashed')
    ax1.set_xlim(-1.75,0.7)
    ax1.set_ylim(-1.25, 1.5)
    ax1.text(-1.5, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax1.text(0.5, 0.5, r'$\rm{AGN}$', ha='right')
    ax1.set_ylabel(r'$\rm{log}_{10}[OIII]/H_{\beta}$')
    ax1.set_xlabel(r'$\rm{log}_{10}[NII]/H_{\alpha}$')
    ax1.minorticks_on()
    ax1.tick_params('both', which='major', length=8, width=1)
    ax1.tick_params('both', which='minor', length=3, width=0.5)

    # SII
    ax1 = plt.subplot(132)
    ax1.scatter(np.log10((df['sii_6717_flux']+df['sii_6731_flux'])/df['h_alpha_flux']), np.log10(df['oiii_5007_flux']/df['h_beta_flux']), alpha=0.1, marker='x', color='k')
    ax1.scatter(np.log10((seyf['sii_6717_flux']+seyf['sii_6731_flux'])/seyf['h_alpha_flux']), np.log10(seyf['oiii_5007_flux']/seyf['h_beta_flux']), alpha=0.5, marker='o', color='r')
    ax1.plot(x2, y2, c='k', linestyle='dashed')
    ax1.plot(sx1, sy1, c='k', linestyle='-.', linewidth=3)
    ax1.set_xlim(-1.3,0.3)
    ax1.set_ylim(-1.25, 1.5)
    ax1.text(-1.2, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax1.set_xlabel(r'$\rm{log}_{10}[SII]/H_{\alpha}$')
    ax1.minorticks_on()
    ax1.tick_params('y', labelleft='off')
    ax1.tick_params('both', which='major', length=8, width=1)
    ax1.tick_params('both', which='minor', length=3, width=0.5)
    ax1.text(-0.5, 1.25, r'$\rm{Seyfert}$')
    ax1.text(-0.1, -0.25, r'$\rm{LINER}$')

    ax1 = plt.subplot(133)
    ax1.scatter(np.log10(df['oi_6300_flux']/df['h_alpha_flux']), np.log10(df['oiii_5007_flux']/df['h_beta_flux']), alpha=0.1, marker='x', color='k')
    ax1.scatter(np.log10(seyf['oi_6300_flux']/seyf['h_alpha_flux']), np.log10(seyf['oiii_5007_flux']/seyf['h_beta_flux']), alpha =0.5, marker='o', color='r')
    ax1.plot(x3, y3, c='k', linestyle='dashed')
    ax1.plot(sx2, sy2, c='k', linestyle='-.', linewidth=3)
    ax1.set_xlim(-2.25,-0.1) 
    ax1.set_ylim(-1.25, 1.5)
    ax1.text(-2.0, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax1.text(-1.25, 1.25, r'$\rm{Seyfert}$')
    ax1.text(-0.75, -0.25, r'$\rm{LINER}$')
    ax1.set_xlabel(r'$\rm{log}_{10}[OI]/H_{\alpha}$')
    ax1.tick_params('y', labelleft='off')
    ax1.minorticks_on()
    ax1.tick_params('both', which='major', length=8, width=1)
    ax1.tick_params('both', which='minor', length=3, width=0.5)
    
    plt.tight_layout()
    

# Bubble plot
def plot_bubble(df, seyf, x, y, x1, y1, x2, y2, sx1, sy1, x3, y3, sx2, sy2):
    plt.figure(figsize=(15, 5))

    # Panel 1: NII
    ax1 = plt.subplot(131)
    x_data_1 = np.log10(df['nii_6584_flux'] / df['h_alpha_flux'])
    y_data_1 = np.log10(df['oiii_5007_flux'] / df['h_beta_flux'])
    
    # KDE density estimate for bubble sizes
    xy = np.vstack([x_data_1, y_data_1])
    kde = gaussian_kde(xy)
    density = kde(xy)

    ax1.scatter(x_data_1, y_data_1, c=density, s=density * 500, alpha=0.5, cmap='cividis')
    ax1.scatter(np.log10(seyf['nii_6584_flux'] / seyf['h_alpha_flux']),
                np.log10(seyf['oiii_5007_flux'] / seyf['h_beta_flux']),
                alpha=0.5, color='r', edgecolors='black', s=30, label='Seyfert')
    
    ax1.plot(x, y, c='k', linestyle='solid')
    ax1.plot(x1, y1, c='k', linestyle='dashed')
    ax1.set_xlim(-1.75, 0.7)
    ax1.set_ylim(-1.25, 1.5)
    ax1.text(-1.5, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax1.text(0.5, 0.5, r'$\rm{AGN}$', ha='right')
    ax1.set_ylabel(r'$\rm{log}_{10}[OIII]/H_{\beta}$')
    ax1.set_xlabel(r'$\rm{log}_{10}[NII]/H_{\alpha}$')
    ax1.minorticks_on()
    ax1.tick_params('both', which='major', length=8, width=1)

    # Panel 2: SII
    ax2 = plt.subplot(132)
    x_data_2 = np.log10((df['sii_6717_flux'] + df['sii_6731_flux']) / df['h_alpha_flux'])
    y_data_2 = np.log10(df['oiii_5007_flux'] / df['h_beta_flux'])
    
    # KDE density estimate for bubble sizes
    xy = np.vstack([x_data_2, y_data_2])
    kde = gaussian_kde(xy)
    density = kde(xy)

    ax2.scatter(x_data_2, y_data_2, c=density, s=density * 500, alpha=0.5, cmap='cividis')
    ax2.scatter(np.log10((seyf['sii_6717_flux'] + seyf['sii_6731_flux']) / seyf['h_alpha_flux']),
                np.log10(seyf['oiii_5007_flux'] / seyf['h_beta_flux']),
                alpha=0.5, color='r', edgecolors='black', s=30, label='Seyfert')

    ax2.plot(x2, y2, c='k', linestyle='dashed')
    ax2.plot(sx1, sy1, c='k', linestyle='-.', linewidth=3)
    ax2.set_xlim(-1.3, 0.3)
    ax2.set_ylim(-1.25, 1.5)
    ax2.text(-1.2, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax2.set_xlabel(r'$\rm{log}_{10}[SII]/H_{\alpha}$')
    ax2.minorticks_on()
    ax2.tick_params('y', labelleft=False)
    ax2.tick_params('both', which='major', length=8, width=1)
    ax2.tick_params('both', which='minor', length=3, width=0.5)
    ax2.text(-0.5, 1.25, r'$\rm{Seyfert}$')
    ax2.text(-0.1, -0.25, r'$\rm{LINER}$')

    # Panel 3: OI
    ax3 = plt.subplot(133)
    x_data_3 = np.log10(df['oi_6300_flux'] / df['h_alpha_flux'])
    y_data_3 = np.log10(df['oiii_5007_flux'] / df['h_beta_flux'])
    
    # KDE density estimate for

        # Panel 3: OI
    ax3 = plt.subplot(133)
    x_data_3 = np.log10(df['oi_6300_flux'] / df['h_alpha_flux'])
    y_data_3 = np.log10(df['oiii_5007_flux'] / df['h_beta_flux'])
    
    # KDE density estimate for bubble sizes
    xy = np.vstack([x_data_3, y_data_3])
    kde = gaussian_kde(xy)
    density = kde(xy)

    ax3.scatter(x_data_3, y_data_3, c=density, s=density * 500, alpha=0.5, cmap='cividis')
    ax3.scatter(np.log10(seyf['oi_6300_flux'] / seyf['h_alpha_flux']),
                np.log10(seyf['oiii_5007_flux'] / seyf['h_beta_flux']),
                alpha=0.5, color='r', edgecolors='black', s=30, label='Seyfert')

    ax3.plot(x3, y3, c='k', linestyle='dashed')
    ax3.plot(sx2, sy2, c='k', linestyle='-.', linewidth=3)
    ax3.set_xlim(-2.25, -0.1)
    ax3.set_ylim(-1.25, 1.5)
    ax3.text(-2.0, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax3.text(-1.25, 1.25, r'$\rm{Seyfert}$')
    ax3.text(-0.75, -0.25, r'$\rm{LINER}$')
    ax3.set_xlabel(r'$\rm{log}_{10}[OI]/H_{\alpha}$')
    ax3.tick_params('y', labelleft=False)
    ax3.minorticks_on()
    ax3.tick_params('both', which='major', length=8, width=1)
    ax3.tick_params('both', which='minor', length=3, width=0.5)

    plt.tight_layout()



# Soft plot
def plot_soft(df, seyf, x, y, x1, y1, x2, y2, sx1, sy1, x3, y3, sx2, sy2):
    plt.figure(figsize=(15,5))
    ax1 = plt.subplot(131)
    x_data_1 = np.log10(df['nii_6584_flux']/df['h_alpha_flux'])
    y_data_1 = np.log10(df['oiii_5007_flux']/df['h_beta_flux'])
    sc1 = ax1.scatter(x_data_1, y_data_1, alpha=0.1, c=y_data_1, cmap='viridis', marker='x')
    ax1.scatter(np.log10(seyf['nii_6584_flux']/seyf['h_alpha_flux']),
            np.log10(seyf['oiii_5007_flux']/seyf['h_beta_flux']),
            alpha=0.3, c='r', marker='o', edgecolor='black')
    
    ax1.plot(x, y, c='k', linestyle='solid')
    ax1.plot(x1, y1, c='k', linestyle='dashed')
    ax1.set_xlim(-1.75, 0.7)
    ax1.set_ylim(-1.25, 1.5)
    ax1.text(-1.5, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax1.text(0.5, 0.5, r'$\rm{AGN}$', ha='right')
    ax1.set_ylabel(r'$\rm{log}_{10}[OIII]/H_{\beta}$')
    ax1.set_xlabel(r'$\rm{log}_{10}[NII]/H_{\alpha}$')
    ax1.minorticks_on()
    ax1.tick_params('both', which='major', length=8, width=1)
    ax1.tick_params('both', which='minor', length=3, width=0.5)

# -------- Panel 2 (SII) --------
    ax2 = plt.subplot(132)
    x_data_2 = np.log10((df['sii_6717_flux']+df['sii_6731_flux'])/df['h_alpha_flux'])
    y_data_2 = np.log10(df['oiii_5007_flux']/df['h_beta_flux'])

    sc2 = ax2.scatter(x_data_2, y_data_2, alpha=0.1, c=y_data_2, cmap='plasma', marker='x')
    ax2.scatter(np.log10((seyf['sii_6717_flux']+seyf['sii_6731_flux'])/seyf['h_alpha_flux']),
            np.log10(seyf['oiii_5007_flux']/seyf['h_beta_flux']),
            alpha=0.3, c='r', marker='o', edgecolor='black')

    ax2.plot(x, y, c='k', linestyle='solid')
    ax2.plot(x2, y2, c='k', linestyle='dashed')
    ax2.plot(sx1, sy1, c='k', linestyle='-.', linewidth=3)
    ax2.set_xlim(-1.3, 0.3)
    ax2.set_ylim(-1.25, 1.5)
    ax2.text(-1.2, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax2.set_xlabel(r'$\rm{log}_{10}[SII]/H_{\alpha}$')
    ax2.minorticks_on()
    ax2.tick_params('y', labelleft=False)
    ax2.tick_params('both', which='major', length=8, width=1)
    ax2.tick_params('both', which='minor', length=3, width=0.5)
    ax2.text(-0.5, 1.25, r'$\rm{Seyfert}$')
    ax2.text(-0.1, -0.25, r'$\rm{LINER}$')

# -------- Panel 3 (OI) --------
    ax3 = plt.subplot(133)
    x_data_3 = np.log10(df['oi_6300_flux']/df['h_alpha_flux'])
    y_data_3 = np.log10(df['oiii_5007_flux']/df['h_beta_flux'])

    sc3 = ax3.scatter(x_data_3, y_data_3, alpha=0.1, c=y_data_3, cmap='cividis', marker='x')
    ax3.scatter(np.log10(seyf['oi_6300_flux']/seyf['h_alpha_flux']),
            np.log10(seyf['oiii_5007_flux']/seyf['h_beta_flux']),
            alpha=0.3, c='r', marker='o', edgecolor='black')

    ax3.plot(x3, y3, c='k', linestyle='dashed')
    ax3.plot(sx2, sy2, c='k', linestyle='-.', linewidth=3)
    
    ax3.plot(x3, y3, c='k', linestyle='dashed')
    ax3.plot(sx2, sy2, c='k', linestyle='-.', linewidth=3)
    ax3.set_xlim(-2.25, -0.1)
    ax3.set_ylim(-1.25, 1.5)
    ax3.text(-2.0, -1.0, r'$\rm{Star}$ $\rm{Forming}$')
    ax3.text(-1.25, 1.25, r'$\rm{Seyfert}$')
    ax3.text(-0.75, -0.25, r'$\rm{LINER}$')
    ax3.set_xlabel(r'$\rm{log}_{10}[OI]/H_{\alpha}$')
    ax3.tick_params('y', labelleft=False)
    ax3.minorticks_on()
    ax3.tick_params('both', which='major', length=8, width=1)
    ax3.tick_params('both', which='minor', length=3, width=0.5) 

# -------- Layout & Save --------
    plt.tight_layout()
