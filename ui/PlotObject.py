# Plot Object Class
# Author: Shihchuan Kao (Kevin Kao)
# Contact: kaoshihchuan@gmail.com

import tkinter as tk
import typing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import pandas as pd


class Plots:

    __instance__ = None

    def __init__(self):

        self.canvas = None
        self.fig = plt.figure(figsize=(8, 7))

        # self.ax1 = plt.subplot2grid((2, 1), (0, 0))
        self.ax1 = self.fig.add_subplot()
        self.ax1.set_xlabel('Variable')
        self.ax1.set_ylabel('Count')

        self.x_range: typing.Tuple = ()
        Plots.__instance__ = self

    def config(self, root_tk):

        # ==== Scan profile display =====
        self.canvas = FigureCanvasTkAgg(self.fig, master=root_tk)  # A tk.DrawingArea.
        self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=15, columnspan=11, sticky='NSWE')
        # self.fig.tight_layout(w_pad=0.8, h_pad=0.0)

        # ##############    TOOLBAR    ###############
        toolbar_frame = tk.Frame(master=root_tk)
        toolbar_frame.grid(row=0, column=0, columnspan=6)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def add_fig_text(self, var_df: pd.Series, x: float = 0.8, y: float = 0.95):
        print(' statistic = %s' % var_df.describe())
        stat_str = '\n'.join((
            r'Count = %.0f' % (var_df.count()),
            r'$\mu=%.2f$' % (var_df.mean()),
            r'$\sigma=%.2f$' % (var_df.std()),
            r'$min=%.2f$' % (var_df.min()),
            r'$max=%.2f$' % (var_df.max())
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.ax1.text(x, y, stat_str, transform=self.ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    def add_str_var_fig_text(self, var_df: pd.Series, x: float = 0.8, y: float = 0.95):
        print(' statistic = %s' % var_df.describe())
        freq = float(var_df.describe().freq / var_df.count())
        stat_str = '\n'.join((
            r'Count = %d' % (var_df.count()),
            r'$Unique =%d$' % (var_df.nunique()),
            r'Top item =%s' % var_df.describe().top,
            r'Top freq =%.2f' % freq
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.ax1.text(x, y, stat_str, transform=self.ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    def draw_hist(self, data, n_bin: int, x_range: typing.Tuple, x_label: str, y_label: str, log_y: bool = False):

        self.ax1.cla()
        self.ax1.hist(data, n_bin, range=x_range)
        self.ax1.set_xlabel(x_label)
        self.ax1.set_ylabel(y_label)
        if log_y is True:
            plt.yscale('log')
        else:
            plt.yscale('linear')

        self.add_fig_text(pd.Series(data))
        self.canvas.draw()
        self.canvas.flush_events()

    def draw_xy_correlation(self, x, y, x_label: str = 'x', y_label: str = 'y'):
        self.ax1.cla()
        self.ax1.grid(which='major')
        self.ax1.scatter(x, y, c='green', label="Y", marker='.')
        self.ax1.set_xlabel(x_label)
        self.ax1.set_ylabel(y_label)
        self.canvas.draw()
        self.canvas.flush_events()

    def __del__(self):
        Plots.__instance__ = None

    @staticmethod
    def get_instance():

        if Plots.__instance__ is None:
            Plots()

        return Plots.__instance__
