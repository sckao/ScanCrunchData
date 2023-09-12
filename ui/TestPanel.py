# Data View Panel
# Author: Shihchuan Kao (Kevin Kao)
# Contact: kaoshihchuan@gmail.com

import tkinter as tk
import tkinter.font as tkfont
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfd
import typing

import matplotlib.pyplot as plt
import numpy as np

import data_processor.read_data as read_data
import models.RandomForestFitter as rfFitter
import ui.PlotObject as plotObj


class GuiWindow:
    __instance__ = None

    def __init__(self, rootgui):
        self.window = rootgui

        # define the scanning display
        self.frame_plot = tk.Frame(self.window)
        self.frame_plot.grid(row=0, column=0, sticky=tk.NSEW, rowspan=3)
        self.frame_var = tk.Frame(self.window, height=25, highlightbackground="gray", highlightthickness=2)
        self.frame_var.grid(row=0, column=1, sticky=tk.NSEW, padx=10, pady=(40, 0))
        self.frame_display = tk.Frame(self.window, height=20, highlightbackground="gray", highlightthickness=2)
        self.frame_display.grid(row=1, column=1, sticky=tk.NSEW, padx=10, pady=(5, 0))
        self.frame_model = tk.Frame(self.window, height=30, highlightbackground="gray", highlightthickness=2)
        self.frame_model.grid(row=2, column=1, sticky=tk.NSEW, padx=10, pady=(5, 0))
        self.frame_feature = tk.Frame(self.window, highlightbackground="gray", highlightthickness=2)
        self.frame_feature.grid(row=0, column=2, rowspan=3, sticky=tk.NSEW, padx=1, pady=(40, 0))

        self.label_font_16 = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.label_font_12 = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.report_font_12 = tkfont.Font(family="Helvetica", size=12)

        self.data_obj = read_data.DataFromCSV()
        self.train_x = None
        self.val_x = None
        self.train_y = None
        self.val_y = None

        self.model_opt_list = ['Regressor', 'Classifier']
        self.display_opt_list = ['PercentError', 'Correlation']
        self.criterion_opt_list = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
        self.str2digi_opt_list = ['All', 'Manual']
        self.combined_op_list = ['+', '-', 'AND', 'OR']
        self.filename_var = tk.StringVar(self.window, value='')
        self.nbin_var = tk.IntVar(self.window, value=10)
        self.x_min_var = tk.DoubleVar(self.window, value=-1)
        self.x_max_var = tk.DoubleVar(self.window, value=1)
        self.ntree_var = tk.IntVar(self.window, value=100)
        self.str2digi_opt_var = tk.StringVar(self.window, value=self.str2digi_opt_list[0])
        self.predict_target = tk.StringVar(self.window, value='')
        self.select_feature_a = tk.StringVar(self.window, value='')
        self.select_feature_b = tk.StringVar(self.window, value='')
        self.combined_feature_ab = tk.StringVar(self.window, value='')
        self.combined_op_var = tk.StringVar(self.window, value=self.combined_op_list[0])
        self.model_vars = {
            'n_tree': tk.IntVar(self.window, value=100),
            'model_type': tk.StringVar(self.window, value=self.model_opt_list[0]),
            'criterion': tk.StringVar(self.window, value=self.criterion_opt_list[0]),
            'max_depth': tk.IntVar(self.window, value=100),
            'max_feature': tk.IntVar(self.window, value=100),
            'min_samples_split': tk.IntVar(self.window, value=2),
            'min_samples_leaf': tk.IntVar(self.window, value=1),
            'display_option': tk.StringVar(self.window, value=self.display_opt_list[0])
        }

        self.the_model = None
        self.predict: typing.Union[None, np.ndarray] = None

        self.str_data_var = tk.StringVar(self.window, value='')
        self.str_data_list = ['No Data']
        self.num_data_var = tk.StringVar(self.window, value='')
        self.num_data_list = ['No Data']
        self.log_y_boolvar = tk.BooleanVar(self.window, value=tk.FALSE)
        self.str_var_focus: bool = True

        # ========= config matplot display =============
        self.plot = plotObj.Plots()
        self.plot.config(self.frame_plot)

        open_file_btn = tk.Button(self.frame_var, text='Load CSV', command=self.prompt_to_select_csv_file,
                                  font=self.label_font_12)
        open_file_btn.grid(row=0, column=0, sticky=tk.EW)
        filename_label = tk.Entry(self.frame_var, textvariable=self.filename_var, justify='right')
        filename_label.grid(row=1, column=0, columnspan=4, pady=1, sticky=tk.EW)

        fv_label = tk.Label(self.frame_var, text='Feature Variables', font=self.label_font_12)
        fv_label.grid(row=2, column=0, columnspan=2)

        num_var_label = tk.Label(self.frame_var, text='Numerical Variable', width=15, justify='left')
        num_var_label.grid(row=3, column=0)
        add_num_var_btn = tk.Button(self.frame_var, text='Add Feature', width=10, command=self.add_feature_num_var)
        add_num_var_btn.grid(row=3, column=3)

        str_var_label = tk.Label(self.frame_var, text='String Variable', width=15, justify='left')
        str_var_label.grid(row=4, column=0)
        add_str_var_btn = tk.Button(self.frame_var, text='Add Feature', width=10, command=self.add_feature_str_var)
        add_str_var_btn.grid(row=4, column=3)

        s2d_transfer_label = tk.Label(self.frame_var, text='Transfer Option', width=15, justify='left')
        s2d_transfer_label.grid(row=5, column=0)
        self.str2digi_opt_menu = tk.OptionMenu(
            self.frame_var,
            self.str2digi_opt_var,
            *self.str2digi_opt_list,
        )
        self.str2digi_opt_menu.grid(row=5, column=1)

        select_target = tk.Button(self.frame_var, text='Select Target', command=self.add_predict_target)
        select_target.grid(row=6, column=0, columnspan=2, sticky=tk.EW)

        # frame_display
        display_label = tk.Label(self.frame_display, text='Display Setting', font=self.label_font_12)
        display_label.grid(row=0, column=0, columnspan=2)
        log_y_check_btn = tk.Checkbutton(
            self.frame_display,
            variable=self.log_y_boolvar,
            onvalue=tk.TRUE,
            offvalue=tk.FALSE,
            text="Log Y",
            bd=0,
            command=self.toggle_y_scale
        )
        log_y_check_btn.grid(row=1, column=0)

        nbin_label = tk.Label(self.frame_display, text='N Bin', width=8, justify='left')
        nbin_label.grid(row=2, column=0, padx=2)
        nbin_entry = tk.Entry(self.frame_display, textvariable=self.nbin_var, width=10, justify='right')
        nbin_entry.grid(row=2, column=1, padx=2)

        x_range_label = tk.Label(self.frame_display, text='X (Min, Max)', width=12, justify='left')
        x_range_label.grid(row=3, column=0)
        xmin_entry = tk.Entry(self.frame_display, textvariable=self.x_min_var, width=10, justify='right')
        xmin_entry.grid(row=3, column=1, padx=1)
        xmax_entry = tk.Entry(self.frame_display, textvariable=self.x_max_var, width=10, justify='right')
        xmax_entry.grid(row=3, column=2, padx=1)

        # frame_model
        run_fitter_btn = tk.Button(self.frame_model, text='Run Model',
                                   command=self.run_model_fitter,
                                   font=self.label_font_12)
        run_fitter_btn.grid(row=0, column=0, sticky=tk.NW)

        self.model_opt_menu = tk.OptionMenu(
            self.frame_model,
            self.model_vars['model_type'],
            *self.model_opt_list,
            command=lambda selection: self.set_criterion()
        )
        self.model_opt_menu.configure(width=12)
        self.model_opt_menu.grid(row=0, column=1, padx=2)

        ntree_label = tk.Label(self.frame_model, text='N Tree', width=8, justify='left', borderwidth=3, relief="ridge")
        ntree_label.grid(row=1, column=0, sticky=tk.EW)
        ntree_entry = tk.Entry(self.frame_model, textvariable=self.model_vars['n_tree'], width=12, justify='right')
        ntree_entry.grid(row=1, column=1, padx=2)

        criterion_label = tk.Label(self.frame_model, text='Criterion',
                                   width=8, justify='left', borderwidth=3, relief="ridge")
        criterion_label.grid(row=2, column=0, sticky=tk.EW)

        self.criterion_opt_menu = tk.OptionMenu(
            self.frame_model,
            self.model_vars['criterion'],
            *self.criterion_opt_list,
        )
        self.criterion_opt_menu.configure(width=12)
        self.criterion_opt_menu.grid(row=2, column=1, padx=2)

        max_depth_label = tk.Label(self.frame_model, text='Max Depth',
                                   width=8, justify='left', borderwidth=3, relief="ridge")
        max_depth_label.grid(row=3, column=0, sticky=tk.EW)
        max_depth_entry = tk.Entry(self.frame_model, textvariable=self.model_vars['max_depth'],
                                   width=12, justify='right')
        max_depth_entry.grid(row=3, column=1, padx=2)

        max_feature_label = tk.Label(self.frame_model, text='Max Feature',
                                     width=8, justify='left', borderwidth=3, relief="ridge")
        max_feature_label.grid(row=4, column=0, sticky=tk.EW)
        max_feature_entry = tk.Entry(self.frame_model, textvariable=self.model_vars['max_feature'],
                                     width=12, justify='right')
        max_feature_entry.grid(row=4, column=1, padx=2)

        min_split_label = tk.Label(self.frame_model, text='Min Split',
                                   width=8, justify='left', borderwidth=3, relief="ridge")
        min_split_label.grid(row=5, column=0, sticky=tk.EW)
        min_split_entry = tk.Entry(self.frame_model, textvariable=self.model_vars['min_samples_split'],
                                   width=12, justify='right')
        min_split_entry.grid(row=5, column=1, padx=2)

        min_leaf_label = tk.Label(self.frame_model, text='Min Leaf',
                                  width=8, justify='left', borderwidth=3, relief="ridge")
        min_leaf_label.grid(row=6, column=0, sticky=tk.EW)
        min_leaf_entry = tk.Entry(self.frame_model, textvariable=self.model_vars['min_samples_leaf'],
                                  width=12, justify='right')
        min_leaf_entry.grid(row=6, column=1, padx=2)

        result_display_opt_label = tk.Label(self.frame_model, text='Display',
                                            width=8, justify='left', borderwidth=3, relief="ridge")
        result_display_opt_label.grid(row=7, column=0, sticky=tk.EW)
        self.result_display_opt_menu = tk.OptionMenu(
            self.frame_model,
            self.model_vars['display_option'],
            *self.display_opt_list,
            command=lambda selection: self.result_display_option()
        )
        self.result_display_opt_menu.configure(width=12)
        self.result_display_opt_menu.grid(row=7, column=1, padx=2)

        # frame_feature
        target_label = tk.Label(self.frame_feature, text='Predict Target',
                                justify='center', font=self.label_font_12)
        target_label.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        target_entry = tk.Entry(self.frame_feature, textvariable=self.predict_target,
                                justify='left')
        target_entry.grid(row=1, column=0, columnspan=2)

        feature_list_label = tk.Label(self.frame_feature, text='Selected Feature',
                                      justify='center', font=self.label_font_12)
        feature_list_label.grid(row=2, column=0, columnspan=2, sticky=tk.EW)
        feature_del_btn = tk.Button(self.frame_feature, text='Delete Selected', command=self.remove_select_feature)
        feature_del_btn.grid(row=4, column=0, columnspan=2)
        create_feature_btn = tk.Button(self.frame_feature, text='Create Feature Data',
                                       command=self.crate_feature_data)
        create_feature_btn.grid(row=5, column=0, columnspan=2, sticky=tk.EW)

        feature_a_select_btn = tk.Button(self.frame_feature, text='Select A', width=10,
                                         command=lambda: self.select_feature(self.select_feature_a))
        feature_a_select_btn.grid(row=6, column=0, sticky=tk.EW)
        feature_a_label = tk.Label(self.frame_feature, textvariable=self.select_feature_a,
                                   borderwidth=2, relief="ridge", justify='center')
        feature_a_label.grid(row=6, column=1, sticky=tk.EW)

        feature_b_select_btn = tk.Button(self.frame_feature, text='Select B', width=10,
                                         command=lambda: self.select_feature(self.select_feature_b))
        feature_b_select_btn.grid(row=7, column=0)
        feature_b_label = tk.Label(self.frame_feature, textvariable=self.select_feature_b,
                                   borderwidth=2, relief="ridge", justify='center')
        feature_b_label.grid(row=7, column=1, sticky=tk.EW)

        feature_combine_label = tk.Label(self.frame_feature, text='New Feature',
                                         borderwidth=2, relief="ridge", justify='center')
        feature_combine_label.grid(row=8, column=0, sticky=tk.EW)
        feature_combine_entry = tk.Entry(self.frame_feature, textvariable=self.combined_feature_ab, justify='right')
        feature_combine_entry.grid(row=8, column=1, sticky=tk.EW)

        feature_combine_btn = tk.Button(self.frame_feature, text='Combine', width=10,
                                        command=self.combine_feature)
        feature_combine_btn.grid(row=9, column=0)
        self.feature_op_menu = tk.OptionMenu(
            self.frame_feature,
            self.combined_op_var,
            *self.combined_op_list,
        )
        self.feature_op_menu.grid(row=9, column=1, sticky=tk.EW)

        self.str_varible_menu = None
        self.num_varible_menu = None
        self.feature_listbox = None
        self.set_option_menu()
        self.set_feature_listbox()

        self.window.update_idletasks()

        GuiWindow.__instance__ = self
    # end __init__

    def __del__(self):
        GuiWindow.__instance__ = None
    # end __del__

    def set_option_menu(self):
        self.str_varible_menu = tk.OptionMenu(self.frame_var,
                                              self.str_data_var,
                                              *self.str_data_list,
                                              command=lambda selection: self.select_str_data()
                                              )
        self.str_varible_menu.grid(row=4, column=1, columnspan=2, sticky=tk.EW)

        self.num_varible_menu = tk.OptionMenu(self.frame_var,
                                              self.num_data_var,
                                              *self.num_data_list,
                                              command=lambda selection: self.select_num_data()
                                              )
        self.num_varible_menu.grid(row=3, column=1, columnspan=2, sticky=tk.EW)

    def refresh_opt_menu(self, opt_menu: tk.OptionMenu):
        opt_menu.grid_forget()
        self.set_option_menu()

    def set_feature_listbox(self):
        self.feature_listbox = tk.Listbox(self.frame_feature, height=25,
                                          selectmode="multiple", highlightcolor='blue')
        self.feature_listbox.grid(row=3, column=0, columnspan=2)

    def refresh_feature_menu(self):
        self.feature_listbox.grid_forget()
        self.set_feature_listbox()

    def remove_select_feature(self):

        selected_checkboxs = self.feature_listbox.curselection()

        for item in selected_checkboxs[::-1]:
            print(type(item))
            print(self.feature_listbox.get(item))
            self.feature_listbox.delete(item)

        print(' size = %d' % self.feature_listbox.size())

    def toggle_y_scale(self):
        if self.str_var_focus is True:
            self.select_str_data()
        else:
            self.select_num_data()

    def prompt_to_select_csv_file(self):

        the_filename = tkfd.askopenfilename(
            initialdir="../data/", title="Select file",
            filetypes=(
                ("csv files", "*.csv"),
                ("all files", "*.*")
            )
        )

        self.filename_var.set(the_filename)
        self.data_obj.get_data(the_filename)
        self.data_obj.identify_num_str_data()
        self.str_data_list = self.data_obj.list_str_data
        self.num_data_list = self.data_obj.list_num_data
        self.refresh_opt_menu(self.str_varible_menu)
        self.refresh_opt_menu(self.num_varible_menu)
        self.data_obj.create_process_data()

        print('Selected csv file: %s! ' % the_filename)
        return the_filename

    def select_str_data(self):

        str_var = self.str_data_var.get()
        if self.data_obj.data is None:
            return

        x = self.data_obj.data[str_var].value_counts()
        self.plot.ax1.cla()
        self.plot.ax1.bar(x.index.to_list(), x.to_list())
        if bool(self.log_y_boolvar.get()) is True:
            plt.yscale('log')
        else:
            plt.yscale('linear')

        print(' str statistic : \n %s' % self.data_obj.data[str_var].describe())
        self.plot.add_str_var_fig_text(self.data_obj.data[str_var])
        self.plot.ax1.set_xlabel(str_var)
        self.plot.ax1.set_ylabel('Count')
        self.plot.canvas.draw()
        self.plot.canvas.flush_events()
        self.str_var_focus = True

    def select_num_data(self):
        num_var = self.num_data_var.get()
        if self.data_obj.data is None:
            return
        x = self.data_obj.data[num_var].to_list()
        n_bin = int(self.nbin_var.get())
        self.plot.ax1.cla()
        self.plot.ax1.hist(x, n_bin)
        self.plot.ax1.set_xlabel(num_var)
        self.plot.ax1.set_ylabel('Count')
        if bool(self.log_y_boolvar.get()) is True:
            plt.yscale('log')
        else:
            plt.yscale('linear')

        self.plot.add_fig_text(self.data_obj.data[num_var])

        self.plot.canvas.draw()
        self.plot.canvas.flush_events()
        self.str_var_focus = False

    def set_criterion(self):
        model_type = str(self.model_vars['model_type'].get())
        if model_type == 'Classifier':
            print(' 1. Model is %s ' % model_type)
            self.criterion_opt_list = ['gini', 'entropy', 'log_loss']
            self.model_vars['criterion'].set('gini')
        if model_type == 'Regressor':
            print(' 2. Model is %s ' % model_type)
            self.criterion_opt_list = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
            self.model_vars['criterion'].set('squared_error')

        self.criterion_opt_menu.grid_forget()
        self.criterion_opt_menu = tk.OptionMenu(
            self.frame_model,
            self.model_vars['criterion'],
            *self.criterion_opt_list,
        )
        self.criterion_opt_menu.configure(width=12)
        self.criterion_opt_menu.grid(row=2, column=1, padx=2)

        self.window.update_idletasks()

    def add_feature_num_var(self):
        num_var = str(self.num_data_var.get())
        feature_idx = self.feature_listbox.size()
        self.feature_listbox.insert(feature_idx, num_var)

    def add_predict_target(self):
        num_var = str(self.num_data_var.get())
        print('**> Add Predict Target as %s - %s ' % (str(self.predict_target.get()), num_var))
        self.predict_target.set(num_var)
        print('*** Add Predict Target as %s - %s ' % (str(self.predict_target.get()), num_var))

    def add_feature_str_var(self):

        str_var = str(self.str_data_var.get())
        suffix = self.data_obj.data[str_var].unique()
        if str(self.str2digi_opt_var.get()) == self.str2digi_opt_list[0]:
            self.data_obj.str_to_categorize_code([str_var])
            for sfx in suffix:
                str_var_suffix = str_var + '_' + sfx
                feature_idx = self.feature_listbox.size()
                self.feature_listbox.insert(feature_idx, str_var_suffix)

        if str(self.str2digi_opt_var.get()) == self.str2digi_opt_list[1]:
            feature_idx = self.feature_listbox.size()
            self.feature_listbox.insert(feature_idx, str_var)
            self.data_obj.digi_code_str_feature(str_var)

    def select_feature(self, feature_var: tk.StringVar):

        for it in self.feature_listbox.curselection():
            feature_str = self.feature_listbox.get(it)
            feature_var.set(feature_str)

    def combine_feature(self):
        feature_a = str(self.select_feature_a.get())
        feature_b = str(self.select_feature_b.get())
        feature_ab = str(self.combined_feature_ab.get())
        if str(self.combined_op_var.get()) == self.combined_op_list[0]:
            self.data_obj.combined_num_data_plus(feature_a, feature_b, feature_ab)
        if str(self.combined_op_var.get()) == self.combined_op_list[1]:
            self.data_obj.combined_num_data_minus(feature_a, feature_b, feature_ab)
        if str(self.combined_op_var.get()) == self.combined_op_list[2]:
            self.data_obj.combined_num_data_and(feature_a, feature_b, feature_ab)
        if str(self.combined_op_var.get()) == self.combined_op_list[3]:
            self.data_obj.combined_num_data_or(feature_a, feature_b, feature_ab)

        print(' Combined New Feature is ')
        print(self.data_obj.process_data.head())
        feature_idx = self.feature_listbox.size()
        self.feature_listbox.insert(feature_idx, feature_ab)

    def crate_feature_data(self):

        if self.feature_listbox.size() == 0:
            tkmsg.showwarning('Warning', 'No Selected Feature !')
            return
        if self.data_obj is None:
            tkmsg.showwarning('Warning', 'No Data is loaded !')
            return

        feature_content = self.feature_listbox.get(0, self.feature_listbox.size()-1)
        print(' All feature Content : ')
        print(feature_content)
        feature_list = list(feature_content)
        self.train_x, self.val_x, self.train_y, self.val_y = self.data_obj.create_feature_data(
            feature_list,
            str(self.predict_target.get())
        )

    def run_model_fitter(self):

        n_tree = int(self.model_vars['n_tree'].get())
        model_type = str(self.model_vars['model_type'].get())
        criterion = str(self.model_vars['criterion'].get())
        max_depth = int(self.model_vars['max_depth'].get())
        max_feature = int(self.model_vars['max_feature'].get())
        min_sample_split = int(self.model_vars['min_samples_split'].get())
        min_sample_leaf = int(self.model_vars['min_samples_leaf'].get())

        print(' N Tree = %d, criterion = %s ' % (n_tree, criterion))
        print(' max depth = %d , feature = %d ' % (max_depth, max_feature))
        print(' min split = %d , leaf = %d ' % (min_sample_split, min_sample_leaf))

        self.the_model = rfFitter.RandomForestFitter()
        self.the_model.config_model(n_tree,
                                    model_type=model_type,
                                    criterion=criterion,
                                    max_depth=max_depth,
                                    max_feature=max_feature,
                                    min_samples_split=min_sample_split,
                                    min_samples_leaf=min_sample_leaf)
        self.the_model.input_training_data(self.train_x, self.val_x, self.train_y, self.val_y)
        # self.the_model.get_training_data('data/train.csv', test_size=0.3, target_name='SalePrice')
        self.the_model.fit()
        # self.the_model.get_test_data('data/test.csv')
        self.predict = self.the_model.get_predict(self.the_model.val_x, self.the_model.val_y)

        self.result_display_option()

    def result_display_option(self):

        if self.predict is not None:
            if str(self.model_vars['display_option'].get()) == self.display_opt_list[0]:
                n_bin = int(self.nbin_var.get())
                x_min = float(self.x_min_var.get())
                x_max = float(self.x_max_var.get())
                self.plot.draw_hist(data=self.the_model.percent_err,
                                    n_bin=n_bin,
                                    x_range=(x_min, x_max),
                                    x_label='Percent Error',
                                    y_label='Count',
                                    log_y=bool(self.log_y_boolvar.get()))

            if str(self.model_vars['display_option'].get()) == self.display_opt_list[1]:
                self.plot.draw_xy_correlation(self.predict, self.the_model.val_y, 'Predict', 'Actual')
        else:
            print(' Model fitting failed or not performed ')
