import wx
import os

from numpy import arange, sin, pi
import matplotlib as mpl
mpl.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure

from morphology_analysis import Morphology_analysis
from haralick_analysis import Haralick_analysis
from traj_analysis import Traj_analysis
from traj_analysis import Draw_chosen_traj

class Plot_Panel(wx.Panel):
    def __init__(self, parent, id=-1, dpi=None, name='name',**kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.figure = mpl.figure.Figure(dpi=dpi, figsize=(2, 2))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        # sizer = wx.BoxSizer(wx.VERTICAL)
        Sizer_checkbox = wx.StaticBox(self, -1, name)
        nmSizer = wx.StaticBoxSizer(Sizer_checkbox, wx.VERTICAL)
        nmSizer.Add(self.canvas, 1, wx.EXPAND | wx.EXPAND)
        nmSizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(nmSizer)

class Morphologyanalysis_Panel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, size=(1000,1000))

        # create some sizers
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        grid_1 = wx.GridBagSizer(hgap=6, vgap=4)
        hSizer = wx.BoxSizer(wx.HORIZONTAL)


        # file path
        self.statictext_inputpath = wx.StaticText(self, label='input path')
        self.statictext_outputpath = wx.StaticText(self, label='output path')
        self.textctrl_inputpath = wx.TextCtrl(self, size=(140, -1), value='')
        self.textctrl_outputpath = wx.TextCtrl(self, size=(140, -1), value='')
        self.button_select_inputpath = wx.Button(self, label='choose')
        self.Bind(wx.EVT_BUTTON, self.OnClick_select_inputpath, self.button_select_inputpath)
        self.button_select_outputpath = wx.Button(self, label='choose')
        self.Bind(wx.EVT_BUTTON, self.OnClick_select_outputpath, self.button_select_outputpath)
        grid_1.Add(self.statictext_inputpath, pos=(0, 0), flag=wx.ALIGN_LEFT, )
        grid_1.Add(self.statictext_outputpath, pos=(1, 0), flag=wx.ALIGN_LEFT, )
        grid_1.Add(self.textctrl_inputpath, pos=(0, 1), span=(1, 3), flag=wx.EXPAND)
        grid_1.Add(self.textctrl_outputpath, pos=(1, 1), span=(1, 3), flag=wx.EXPAND)
        grid_1.Add(self.button_select_inputpath, pos=(0, 4), flag=wx.ALIGN_RIGHT, )
        grid_1.Add(self.button_select_outputpath, pos=(1, 4), flag=wx.ALIGN_RIGHT)
        
        self.statictext_ptsnum = wx.StaticText(self, label='ptsnum')
        self.textctrl_ptsnum = wx.TextCtrl(self, size=(140, -1), value='150')
        grid_1.Add(self.statictext_ptsnum, pos=(2, 0), flag=wx.ALIGN_LEFT, )
        grid_1.Add(self.textctrl_ptsnum, pos=(2, 1), span=(1, 1), flag=wx.ALIGN_LEFT)


        # figure box
        self.plotpanel_fig1 = Plot_Panel(self, name='contours')
        self.plotpanel_fig2 = Plot_Panel(self, name='mean cell contour')
        self.plotpanel_fig3 = Plot_Panel(self, name='Morphology PCA')
        hSizer.Add(self.plotpanel_fig1, 1, wx.EXPAND|wx.ALL, 5)
        hSizer.Add(self.plotpanel_fig2, 1, wx.EXPAND|wx.ALL, 5)
        hSizer.Add(self.plotpanel_fig3, 1, wx.EXPAND|wx.ALL, 5)

        # run
        self.button_run = wx.Button(self, label='Run')
        self.Bind(wx.EVT_BUTTON, self.OnClick_run, self.button_run)

        mainSizer.Add(grid_1, 0, wx.ALL | wx.EXPAND, 10)
        mainSizer.Add(hSizer, 1, wx.ALL | wx.EXPAND, 10)
        mainSizer.Add(self.button_run, 0, wx.ALL|wx.CENTER, 10)

        grid_1.AddGrowableCol(2, 1)
        mainSizer.SetSizeHints(self)
        self.SetSizerAndFit(mainSizer)

    # file path
    def OnClick_select_inputpath(self, event):
        self.dirname = ''
        dlg = wx.DirDialog(self, 'Choose a path', '', wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.inputpath_path = dlg.GetPath()
            self.textctrl_inputpath.SetValue(self.inputpath_path)
        dlg.Destroy()

    def OnClick_select_outputpath(self, event):
        self.dirname = ''
        dlg = wx.DirDialog(self, 'Choose a path', '', wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.outputpath_path = dlg.GetPath()
            self.textctrl_outputpath.SetValue(self.outputpath_path)
        dlg.Destroy()

    def OnClick_run(self, event):
        ''' Open a file'''
        input_path = self.textctrl_inputpath.GetValue()
        output_path = self.textctrl_outputpath.GetValue()
        pts_num = int(self.textctrl_ptsnum.GetValue())
        # if input_path[-1] != '/':
        #     input_path += '/'
        # if output_path[-1] != '/':
        #     output_path += '/'

        self.plotpanel_fig1.figure.clear()
        self.plotpanel_fig2.figure.clear()
        self.plotpanel_fig3.figure.clear()

        ax1 = self.plotpanel_fig1.figure.gca()
        ax2 = self.plotpanel_fig2.figure.gca()
        ax3 = self.plotpanel_fig3.figure.gca()

        self.plotpanel_fig1.canvas.draw()
        self.plotpanel_fig2.canvas.draw()
        self.plotpanel_fig3.canvas.draw()

        Morphology_analysis(input_path, output_path, pts_num, ax1, ax2, ax3)

        # A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
        dlg = wx.MessageDialog(self, 'Morphology_analysis complete, data written in:  \n%s/cells \n%s/mean_cell_contour \n%s/morph_pca \n' % (output_path, output_path, output_path, ), 'complete', wx.OK)
        dlg.ShowModal()  # Show it
        dlg.Destroy()  # finally destroy it when finished.

class Haralicanalysis_Panel(wx.Panel):
    def __init__(self, parent):
        
        wx.Panel.__init__(self, parent)

        # create some sizers
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        grid_1 = wx.GridBagSizer(hgap=6, vgap=4)
        # hSizer = wx.BoxSizer(wx.HORIZONTAL)

        # file path
        self.statictext_inputpath = wx.StaticText(self, label='input path')
        self.statictext_outputpath = wx.StaticText(self, label='output path')
        self.textctrl_inputpath = wx.TextCtrl(self, size=(140, -1), value='')
        self.textctrl_outputpath = wx.TextCtrl(self, size=(140, -1), value='')
        self.button_select_mianpath = wx.Button(self, label='choose')
        self.Bind(wx.EVT_BUTTON, self.OnClick_select_inputpath, self.button_select_mianpath)
        self.button_select_outputpath = wx.Button(self, label='choose')
        self.Bind(wx.EVT_BUTTON, self.OnClick_select_outputpath, self.button_select_outputpath)
        grid_1.Add(self.statictext_inputpath, pos=(0, 0), flag=wx.ALIGN_LEFT, )
        grid_1.Add(self.statictext_outputpath, pos=(1, 0), flag=wx.ALIGN_LEFT, )
        grid_1.Add(self.textctrl_inputpath, pos=(0, 1), span=(1, 3), flag=wx.EXPAND)
        grid_1.Add(self.textctrl_outputpath, pos=(1, 1), span=(1, 3), flag=wx.EXPAND)
        grid_1.Add(self.button_select_mianpath, pos=(0, 4), flag=wx.ALIGN_RIGHT, )
        grid_1.Add(self.button_select_outputpath, pos=(1, 4), flag=wx.ALIGN_RIGHT)


        # parameters
        self.statictext_flourinterval = wx.StaticText(self, label='flour interval')
        self.textctrl_flourinterval = wx.TextCtrl(self, size=(140, -1), value='2')
        grid_1.Add(self.statictext_flourinterval, pos=(2, 0), flag=wx.ALIGN_LEFT, )
        grid_1.Add(self.textctrl_flourinterval, pos=(2, 1), span=(1, 1), flag=wx.ALIGN_LEFT)

        # figure
        self.plotpanel_fig1 = Plot_Panel(self, name='Fluorescence PCA')

        # run
        self.button_run = wx.Button(self, label='Run')
        self.Bind(wx.EVT_BUTTON, self.OnClick_run, self.button_run)

        mainSizer.Add(grid_1, 0, wx.ALL | wx.EXPAND, 10)
        mainSizer.Add(self.plotpanel_fig1, 1, wx.EXPAND|wx.ALL, 5)
        mainSizer.Add(self.button_run, 0, wx.ALL | wx.CENTER, 10)

        grid_1.AddGrowableCol(2, 1)
        mainSizer.SetSizeHints(self)
        self.SetSizerAndFit(mainSizer)

    # file path
    def OnClick_select_inputpath(self, event):
        self.dirname = ''
        dlg = wx.DirDialog(self, 'Choose a path', '', wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.inputpath_path = dlg.GetPath()
            self.textctrl_inputpath.SetValue(self.inputpath_path)
        dlg.Destroy()

    def OnClick_select_outputpath(self, event):
        self.dirname = ''
        dlg = wx.DirDialog(self, 'Choose a path', '', wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.outputpath_path = dlg.GetPath()
            self.textctrl_outputpath.SetValue(self.outputpath_path)
        dlg.Destroy()

    def OnClick_run(self, event):
        ''' Open a file'''
        input_path = self.textctrl_inputpath.GetValue()
        output_path = self.textctrl_outputpath.GetValue()
        # if input_path[-1] != '/':
        #     input_path += '/'
        # if output_path[-1] != '/':
        #     output_path += '/'

        flour_interval = int(self.textctrl_flourinterval.GetValue())

        self.plotpanel_fig1.figure.clear()
        ax1 = self.plotpanel_fig1.figure.gca()

        Haralick_analysis(input_path, output_path, flour_interval, ax1,)
        self.plotpanel_fig1.canvas.draw()


        # A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
        dlg = wx.MessageDialog(self, 'Haralick_analysis complete, data written in: \n%s/fluor_pca \n%s/fluor_cells ' % (output_path, output_path), 'complete', wx.OK)
        dlg.ShowModal()  # Show it
        dlg.Destroy()  # finally destroy it when finished.

class Trajanalysis_panel(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent)

        # create some sizers
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        grid_path = wx.GridBagSizer(hgap=6, vgap=4)
        hSizer = wx.BoxSizer(wx.HORIZONTAL)
        grid_plotchosen_path = wx.GridBagSizer(hgap=6, vgap=4)
        Sizer_plotchosen = wx.StaticBox(self, -1, 'chosen trajectory to plot')
        namedSizer_plotchosen = wx.StaticBoxSizer(Sizer_plotchosen, wx.VERTICAL)
        hSizer_trajlen = wx.BoxSizer(wx.HORIZONTAL)
        hSizer_output = wx.BoxSizer(wx.HORIZONTAL)
        vSizer_fig1 = wx.BoxSizer(wx.VERTICAL)

        # file path
        self.statictext_inputpath = wx.StaticText(self, label='input path')
        self.statictext_outputpath = wx.StaticText(self, label='output path')
        self.textctrl_inputpath = wx.TextCtrl(self, size=(140, -1), value='')
        self.textctrl_outputpath = wx.TextCtrl(self, size=(140, -1), value='')
        self.button_select_inputpath = wx.Button(self, label='choose')
        self.Bind(wx.EVT_BUTTON, self.OnClick_select_inputpath, self.button_select_inputpath)
        self.button_select_outputpath = wx.Button(self, label='choose')
        self.Bind(wx.EVT_BUTTON, self.OnClick_select_outputpath, self.button_select_outputpath)
        grid_path.Add(self.statictext_inputpath, pos=(0, 0), flag=wx.ALIGN_LEFT, )
        grid_path.Add(self.statictext_outputpath, pos=(1, 0), flag=wx.ALIGN_LEFT, )
        grid_path.Add(self.textctrl_inputpath, pos=(0, 1), span=(1, 3), flag=wx.EXPAND)
        grid_path.Add(self.textctrl_outputpath, pos=(1, 1), span=(1, 3), flag=wx.EXPAND)
        grid_path.Add(self.button_select_inputpath, pos=(0, 4), flag=wx.ALIGN_RIGHT, )
        grid_path.Add(self.button_select_outputpath, pos=(1, 4), flag=wx.ALIGN_RIGHT)

        # chosen plot
        self.statictext_chosentrajpath = wx.StaticText(self, label='trajectory path')
        self.textctrl_chosentrajpath = wx.TextCtrl(self, size=(140, -1), value='')
        self.button_select_chosentrajpath = wx.Button(self, label='choose and plot')
        self.Bind(wx.EVT_BUTTON, self.OnClick_select_chosentrajpath, self.button_select_chosentrajpath)
        grid_plotchosen_path.Add(self.statictext_chosentrajpath, pos=(0, 0), flag=wx.ALIGN_LEFT, )
        grid_plotchosen_path.Add(self.textctrl_chosentrajpath, pos=(0, 1), span=(1, 3), flag=wx.EXPAND)
        grid_plotchosen_path.Add(self.button_select_chosentrajpath, pos=(0, 4), flag=wx.ALIGN_RIGHT, )

        self.plotpanel_fig2 = Plot_Panel(self, name='single cell trajectory plot')
        namedSizer_plotchosen.Add(grid_plotchosen_path, proportion=0, flag=wx.EXPAND|wx.ALL, border=5)
        namedSizer_plotchosen.Add(self.plotpanel_fig2, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)

        # parameters; output1; output text
        self.statictext_minimumtrajlength = wx.StaticText(self, label='minimum traj length')
        self.textctrl_minimumtrajlength = wx.TextCtrl(self, size=(140, -1), value='36')
        hSizer_trajlen.Add(self.statictext_minimumtrajlength, proportion=0, flag=wx.ALIGN_RIGHT, border=5)
        hSizer_trajlen.Add(self.textctrl_minimumtrajlength, proportion=1, flag=wx.ALIGN_LEFT, border=5)

        self.plotpanel_fig1_1 = Plot_Panel(self, name='scaled pricipal mode 1')
        self.plotpanel_fig1_2 = Plot_Panel(self, name='scaled pricipal mode 2')

        self.statictext_output1 = wx.StaticText(self, label='morphology PCA explained \n variance ratio:')
        self.textctrl_output1 = wx.TextCtrl(self, size=(140, -1), value='', style=wx.TE_READONLY)
        self.statictext_output2 = wx.StaticText(self, label='fluorescence PCA explained \n variance ratio:')
        self.textctrl_output2 = wx.TextCtrl(self, size=(140, -1), value='', style=wx.TE_READONLY)
        hSizer_output.Add(self.statictext_output1, proportion=0, flag=wx.ALIGN_RIGHT, border=5)
        hSizer_output.Add(self.textctrl_output1, proportion=1, flag=wx.ALIGN_LEFT, border=5)
        hSizer_output.Add(self.statictext_output2, proportion=0, flag=wx.ALIGN_RIGHT, border=5)
        hSizer_output.Add(self.textctrl_output2, proportion=1, flag=wx.ALIGN_LEFT, border=5)
        vSizer_fig1.Add(hSizer_trajlen, proportion=0, flag=wx.EXPAND|wx.ALL, border=5)
        vSizer_fig1.Add(self.plotpanel_fig1_1, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        vSizer_fig1.Add(self.plotpanel_fig1_2, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        vSizer_fig1.Add(hSizer_output, proportion=0, flag=wx.EXPAND|wx.ALL, border=5)

        hSizer.Add(vSizer_fig1, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)
        hSizer.Add(namedSizer_plotchosen, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)

        # run
        self.button_run = wx.Button(self, label='Run')
        self.Bind(wx.EVT_BUTTON, self.OnClick_run, self.button_run)

        mainSizer.Add(grid_path, 0, wx.ALL | wx.EXPAND, 10)
        mainSizer.Add(hSizer, 2, wx.ALL | wx.EXPAND, 10)
        mainSizer.Add(self.button_run, 0, wx.ALL | wx.CENTER, 10)

        grid_path.AddGrowableCol(2, 1)
        grid_plotchosen_path.AddGrowableCol(2, 1)

        mainSizer.SetSizeHints(self)
        self.SetSizerAndFit(mainSizer)

    # file path
    def OnClick_select_inputpath(self, event):
        self.dirname = ''
        dlg = wx.DirDialog(self, 'Choose a path', '', wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.inputpath_path = dlg.GetPath()
            self.textctrl_inputpath.SetValue(self.inputpath_path)
        dlg.Destroy()

    def OnClick_select_outputpath(self, event):
        self.dirname = ''
        dlg = wx.DirDialog(self, 'Choose a path', '', wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.outputpath_path = dlg.GetPath()
            self.textctrl_outputpath.SetValue(self.outputpath_path)
        dlg.Destroy()

    def OnClick_select_chosentrajpath(self, event):
        self.dirname = ''
        dlg = wx.FileDialog(self, 'Choose a file', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.choesntraj_path = dlg.GetPath()
            self.textctrl_chosentrajpath.SetValue(self.choesntraj_path)
        dlg.Destroy()

        chosentraj_path = self.textctrl_chosentrajpath.GetValue()
        self.plotpanel_fig2.figure.clear()
        message = Draw_chosen_traj(chosentraj_path, self.plotpanel_fig2.figure)


        self.plotpanel_fig2.canvas.draw()

        # A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.

        dlg = wx.MessageDialog(self, str(message),
                               'plot result', wx.OK)
        dlg.ShowModal()  # Show it
        dlg.Destroy()  # finally destroy it when finished.

    def OnClick_run(self, event):
        ''' Open a file'''
        input_path = self.textctrl_inputpath.GetValue()
        output_path = self.textctrl_outputpath.GetValue()
        # if input_path[-1] != '/':
        #     input_path += '/'
        # if output_path[-1] != '/':
        #     output_path += '/'

        minimumtrajlength = int(self.textctrl_minimumtrajlength.GetValue())

        self.plotpanel_fig1_1.figure.clear()
        self.plotpanel_fig1_2.figure.clear()
        self.plotpanel_fig2.figure.clear()
        ax1_1 = self.plotpanel_fig1_1.figure.gca()
        ax1_2 = self.plotpanel_fig1_2.figure.gca()
        fig2 = self.plotpanel_fig2.figure

        output1, output2, message = Traj_analysis(input_path, output_path, minimumtrajlength, ax1_1, ax1_2, fig2)
        
        self.plotpanel_fig1_1.canvas.draw()
        self.plotpanel_fig1_2.canvas.draw()
        self.plotpanel_fig2.canvas.draw()

        self.textctrl_output1.SetValue(str(output1))
        self.textctrl_output2.SetValue(str(output2))

        # A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
        dlg = wx.MessageDialog(self,'Trajectory_analysis complete, data written in:  \n%s/single_cell_traj \n%s' % (output_path, message), 'complete', wx.OK)
        dlg.ShowModal()  # Show it
        dlg.Destroy()  # finally destroy it when finished.

if __name__ == '__main__':
    # app = wx.App(False)
    # frame = wx.Frame(None, title='refine_mask_SNR_label_GUI')
    #
    # panel = Clustering(frame)
    # frame.Show()
    # app.MainLoop()
    #
    app = wx.App(False)
    frame = wx.Frame(None, title='Work Flow',size=(1500, 750))
    nb = wx.Notebook(frame)
    #
    # nb.AddPage(Get_training_data_Panel(nb), 'Get training data')
    # nb.AddPage(Training_CNN(nb), 'Training CNN')
    # nb.AddPage(Detect_with_CNN(nb), 'Detect with CNN')
    nb.AddPage(Morphologyanalysis_Panel(nb), 'morphology analysis')
    nb.AddPage(Haralicanalysis_Panel(nb), 'haralick analysis')
    nb.AddPage(Trajanalysis_panel(nb), 'trajectory analysis')


    frame.Show()
    app.MainLoop()
