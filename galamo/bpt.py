import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from termcolor import colored

def draw(input_file, save_figure=False, output_filename="BPT_diagram.pdf"):
    """
    Draws the BPT diagram based on the input CSV file containing galaxy spectral data.
    
    Parameters:
    - input_file (str): Path to the CSV file.
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
    
    # Plot BPT diagrams
    font = {'family':'serif', 'size':20}
    plt.rc('font', **font)
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rc('axes', labelsize='medium')

    plt.figure(figsize=(15,5))
#This is the first panel of Figure 2
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
# This is the second panel of Figure 2
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

# This is the third panel of Figure 2
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
    plt.subplots_adjust(wspace=0.0)
    
    plt.tight_layout()
    
    # Save figure if needed
    if save_figure:
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, transparent=True)
    
    plt.show()
