import pyAERO
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error


class pyFernando:
    def __init__(self,dataframe,x_column,y_column,c_column,axs,
                    x_label,y_label,c_label=None,vmin=None,vmax=None,cmap='viridis',color_line='red',x_lim=None,y_lim=None,
                    x_text=0.05,y_text=0.95,err_x=None,err_y=None,cbar=True):
        self.dataframe = dataframe
        self.x_column = x_column
        self.y_column = y_column
        self.c_column = c_column
        self.x_label = x_label 
        self.y_label = y_label 
        self.c_label = c_label
        self.vmin = vmin 
        self.vmax = vmax
        self.x_lim = x_lim 
        self.y_lim = y_lim 
        self.x_text = x_text
        self.y_text = y_text
        self.err_x = err_x
        self.err_y = err_y
        self.cbar = cbar
        self.cmap = cmap 
        self.color_line = color_line
        self.axs = axs
    @staticmethod
    def get_global_limits(dfs: list, column):
        """
        dfs: Lista de DataFrames [df1, df2, ...]
        columns: Lista com os nomes das colunas correspondentes [col1, col2, ...]
        """
        mins = [df[column].min() for df in dfs]
        maxs = [df[column].max() for df in dfs]
        return min(mins), max(maxs)

    @staticmethod
    def add_unified_colorbar(fig, collection, label='cbar_name',cbar_configs=[0.88, 0.25, 0.02, 0.5],right=0.85):
        """
        Adiciona um colorbar à direita de um conjunto de eixos 
        sem alterar as proporções originais dos subplots.
        cbar_configs: [esquerda, baixo, largura, altura]
        """
        fig.subplots_adjust(right=right)
        cbar_ax = fig.add_axes(cbar_configs) 
        
        cbar = fig.colorbar(collection, cax=cbar_ax)
        cbar.set_label(label, fontname='serif', fontsize=12)
        return cbar
    def plot_scattering(self):
        '''
        data: your datafram
        x_column: column x of your dataframe
        y_column: column y of your dataframe
        axis: axis of your plot
        c_column: column of your cmap
        x_label: name of var in x-axis
        y_label: name of var in y-axis
        c_label: name of cbar
        err_x: data['error_var'] input your error in x
        err_y: data['error_var] input your error in y
        '''
        x,y = self.dataframe[self.x_column],self.dataframe[self.y_column]
        self.axs.errorbar(
            x=self.dataframe[self.x_column],
            y=self.dataframe[self.y_column],
            xerr=self.err_x,
            yerr=self.err_y,
            fmt='none',
            ecolor='gray',
            elinewidth=0.8,
            alpha=0.5,
            zorder=1,
            capsize=3)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_pred = slope * x + intercept
        rmse = root_mean_squared_error(y,y_pred)
        r2,samples = r2_score(y,y_pred), len(y_pred)
        statistics = f'R: {r_value:.3f}, R²: {r2:.2f}, p-value: {p_value:.3f}\nRMSE: {rmse:.2f}, std-err: {std_err:.2f}, n: {samples}\nY = {slope:.2f}X {intercept:.2f}'
        self.axs.text(x=self.x_text,
                y=self.y_text,
                s=f'{statistics}',
                transform=self.axs.transAxes,
                fontsize=10,linespacing=1.2,
                fontdict={'fontfamily':'serif'},
                verticalalignment='top')
        
        self.axs.set_xlabel(self.x_label,
                        fontname='serif',
                        fontsize=10)
        self.axs.set_ylabel(self.y_label,
                        fontname='serif',
                        fontsize=10)
        try:
            x_min,x_max,y_min,y_max = self.x_lim[0],self.x_lim[1],self.y_lim[0],self.y_lim[1] 
            self.axs.set_xlim(x_min,x_max)
            self.axs.set_ylim(y_min,y_max)
        except: pass
        # Plot scatter
        cax = self.axs.scatter(self.dataframe[self.x_column],
                            self.dataframe[self.y_column],
                            c=self.dataframe[self.c_column],
                            vmin=self.vmin,
                            vmax=self.vmax,cmap=self.cmap,
                            edgecolors='black',
                            linewidths=0.3)
        # plot line
        self.axs.plot(x, y_pred, color=self.color_line, label='Model')
        if self.cbar == True:
            cbar = plt.colorbar(cax,shrink=0.6)
            cbar.set_label(self.c_label, fontname='serif', fontsize=12)
        return cax