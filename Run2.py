# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:14:50 2019

@author: Asif Towheed
"""
#import pyximport; pyximport.install()
import wx
import os
import time 
import threading
import AD_Functions as ad
import AD_Functions_22 as ad2
import SURF_Functions as surf
import SURF_Functions_2 as surf2
import GEI_Functions as gei
import GEI_Functions2 as gei2
import re

SUBJECTNAME = ""
SELECTEDALG = ""

KILL_THREAD = False
RECORDEDNUMBER = []
TRAINED = False
STOPPED = False
PATH = os.getcwd().replace('\\','/')


###################################################################################
## BEGIN NEW TRAINING SET --> SECOND WINDOW
###################################################################################
class ADThread(threading.Thread):
    def __init__(self, index):    
        threading.Thread.__init__(self)
        self.i = 0
        self.index = index
    

    #------------------------------------------------------------------------------
    def run(self):
        if self.index == 0:
            global SUBJECTNAME
            ad.AD_RESET()
            ad.AD_Begin(SUBJECTNAME)

        ad.AD_GetWalk(self.index)
#        ad.AD_GenerateDiffs()

#        global TRAINED, TRAINEDMUTEX
#        TRAINED = True
#        ad.AD_GetDCT()
#        while not TRAINEDMUTEX:
#            dummy = 0
        ad.MID_AD_RESET()
        TRAINEDMUTEX = False
            

#        while True:
#            self.i += 1
#            #print(self.i)
#            
#            global RECORDEDNUMBER
#            if RECORDEDNUMBER[self.index]:
#                break
##---------------------------------------------------------------------------------



###################################################################################
## BEGIN NEW TRAINING SET --> SECOND WINDOW
###################################################################################
class SURFThread(threading.Thread):
    def __init__(self, index):    
        threading.Thread.__init__(self)
        self.i = 0
        self.index = index
    

    #------------------------------------------------------------------------------
    def run(self):
        if self.index == 0:
            global SUBJECTNAME
            surf.SURF_RESET()
            surf.SURF_Begin(SUBJECTNAME)

        surf.SURF_GetWalk(self.index)
#        ad.AD_GenerateDiffs()

        global TRAINED, TRAINEDMUTEX
#        TRAINED = True
#        ad.AD_GetDCT()
#        while not TRAINEDMUTEX:
#            dummy = 0
        surf.MID_SURF_RESET()
        TRAINEDMUTEX = False
            

#        while True:
#            self.i += 1
#            #print(self.i)
#            
#            global RECORDEDNUMBER
#            if RECORDEDNUMBER[self.index]:
#                break
##---------------------------------------------------------------------------------
        
        

###################################################################################
## BEGIN NEW TRAINING SET --> SECOND WINDOW
###################################################################################
class GEIThread(threading.Thread):
    def __init__(self, index):    
        threading.Thread.__init__(self)
        self.i = 0
        self.index = index
    

    #------------------------------------------------------------------------------
    def run(self):
        if self.index == 0:
            global SUBJECTNAME
            gei.GEI_RESET()
            gei.GEI_Begin(SUBJECTNAME)

        gei.GEI_GetWalk(self.index)
#        ad.AD_GenerateDiffs()

        global TRAINED, TRAINEDMUTEX
#        TRAINED = True
#        ad.AD_GetDCT()
#        while not TRAINEDMUTEX:
#            dummy = 0
        gei.MID_GEI_RESET()
        TRAINEDMUTEX = False
            

#        while True:
#            self.i += 1
#            #print(self.i)
#            
#            global RECORDEDNUMBER
#            if RECORDEDNUMBER[self.index]:
#                break
##---------------------------------------------------------------------------------
    

        
###################################################################################
## BEGIN NEW TRAINING SET --> SECOND WINDOW
###################################################################################
class ADThread2(threading.Thread):
    def __init__(self, SUBJECTNAME):    
        threading.Thread.__init__(self)
        self.i = 0
        #self.index = index
        self.subjectname = SUBJECTNAME
    

    #------------------------------------------------------------------------------
    def run(self):
        
        
        ad2.AD_GenerateDiffs(self.subjectname)

        global TRAINED, TRAINEDMUTEX
        TRAINED = True
        print('ADTHREAD2 STOPPED')
#        ad.AD_GetDCT()
#        while not TRAINEDMUTEX:
#            dummy = 0
#        ad2.MID_AD_RESET()
#        TRAINEDMUTEX = False
            

#        while True:
#            self.i += 1
#            #print(self.i)
#            
#            global RECORDEDNUMBER
#            if RECORDEDNUMBER[self.index]:
#                break
##---------------------------------------------------------------------------------



###################################################################################
## BEGIN NEW TRAINING SET --> SECOND WINDOW
###################################################################################
class SURFThread2(threading.Thread):
    def __init__(self, index):    
        threading.Thread.__init__(self)
        self.i = 0
        self.index = index
    

    #------------------------------------------------------------------------------
    def run(self):
        if self.index == 0:
            global SUBJECTNAME
            surf.SURF_RESET()
            surf.SURF_Begin(SUBJECTNAME)

        surf.SURF_GetWalk(self.index)
#        ad.AD_GenerateDiffs()

        global TRAINED, TRAINEDMUTEX
#        TRAINED = True
#        ad.AD_GetDCT()
#        while not TRAINEDMUTEX:
#            dummy = 0
        surf.MID_SURF_RESET()
        TRAINEDMUTEX = False
            

#        while True:
#            self.i += 1
#            #print(self.i)
#            
#            global RECORDEDNUMBER
#            if RECORDEDNUMBER[self.index]:
#                break
##---------------------------------------------------------------------------------
        
        

###################################################################################
## BEGIN NEW TRAINING SET --> SECOND WINDOW
###################################################################################
class GEIThread2(threading.Thread):
    def __init__(self, index):    
        threading.Thread.__init__(self)
        self.i = 0
        self.index = index
    

    #------------------------------------------------------------------------------
    def run(self):
        if self.index == 0:
            global SUBJECTNAME
            gei2.GEI_RESET()
            gei2.GEI_Begin(SUBJECTNAME)

        gei2.GEI_GetWalk(self.index)
#        ad.AD_GenerateDiffs()

        global TRAINED, TRAINEDMUTEX
#        TRAINED = True
#        ad.AD_GetDCT()
#        while not TRAINEDMUTEX:
#            dummy = 0
        gei2.MID_GEI_RESET()
        TRAINEDMUTEX = False
            

#        while True:
#            self.i += 1
#            #print(self.i)
#            
#            global RECORDEDNUMBER
#            if RECORDEDNUMBER[self.index]:
#                break
##---------------------------------------------------------------------------------
    
    

###################################################################################
## BEGIN NEW TRAINING SET --> SECOND WINDOW
###################################################################################
class TrainFrame(wx.Frame):
    """
    Class used for creating frames other than the main one
    """

    #------------------------------------------------------------------------------
    def __init__(self, title, parent=None):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.count = 0 
        self.panel = wx.Panel(self) 
                
        vbox = wx.BoxSizer(wx.VERTICAL) 
        
        self.hboxlist = []
        self.gaugelist = []
        self.btnlist = []
        
        #self.SetBackgroundColour(wx.BLACK)
        
        #self.Colours()
        
        
        for i in range(10):
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            self.hboxlist.append(hbox)
            gauge = wx.Gauge(self.panel, range = 20, size = (250, 25), style =  wx.GA_HORIZONTAL|wx.GA_SMOOTH) 
#            gauge.SetForegroundColour(wx.BLUE)
#            gauge.SetLabel('Not trained')
#            gauge.SetBackgroundColour(wx.BLACK)
#            gauge.SetLabelText('Not trained')
            self.gaugelist.append(gauge)
            btn = wx.ToggleButton(self.panel, i*10, label = "Start " + str(i)) 
            self.btnlist.append(btn)
            self.Bind(wx.EVT_TOGGLEBUTTON, self.OnStart, self.btnlist[i]) 

            self.hboxlist[i].Add(self.gaugelist[i], proportion = 1, flag = wx.ALIGN_CENTRE) 
            self.hboxlist[i].Add(self.btnlist[i], proportion = 1, flag = wx.RIGHT, border = 10) 
            
            global RECORDEDNUMBER
            RECORDEDNUMBER.append(False)
            
            vbox.Add(self.hboxlist[i], proportion = 1, flag = wx.ALIGN_CENTRE) 
             
        self.begintrainingbtn = wx.ToggleButton(self.panel, label = "Begin Training!")
        self.donebtn = wx.Button(self.panel, label = "Done!")
        self.Bind(wx.EVT_BUTTON, self.FinishedTraining, self.donebtn)
        self.Bind(wx.EVT_TOGGLEBUTTON, self.BeginTraining, self.begintrainingbtn)
        hboxforbuttons = wx.BoxSizer(wx.HORIZONTAL)
        hboxforbuttons.Add(self.begintrainingbtn, proportion = 1, flag = wx.ALIGN_CENTRE) 
        hboxforbuttons.Add(self.donebtn, proportion = 1, flag = wx.ALIGN_CENTRE)
        vbox.Add(hboxforbuttons, proportion = 1, flag = wx.ALIGN_CENTRE)
        
        vbox.SetSizeHints(self)    # Resize the window to fit the buttons ONLY
        self.SetSizer(vbox)

        self.Centre() 
        self.Show()   
    		

    #------------------------------------------------------------------------------
    def FinishedTraining(self, e): 
        print('finished!!!')
        self.Close()


    #------------------------------------------------------------------------------
    def BeginTraining(self, e): 
        state = e.GetEventObject().GetValue() 
        if state == True:
            global TRAINED
            TRAINED = False
            e.GetEventObject().SetLabel('Stop Training')
            self.disableWalkButtons()
            
            for i in range(10):
                self.gaugelist[i].SetValue(0)
        
            a2 = ADThread2(SUBJECTNAME)
            #a2.start()
            
            def statusfunc():
                global TRAINEDMUTEX, TRAINED, STOPPED
                print('entered statusfunc')
                BTNTEMP = e.GetEventObject()
                while not TRAINED and not STOPPED:
                    if SELECTEDALG == 'Accumulated Differences':
                        #print('x',x)
                        #print('ad2.AD_GetStatus()',ad2.AD_GetStatus())
                        x = ad2.AD_GetWalkNumber()
                        #prev = None
                        if 0 < x < 10:
                            for i in range(x):
                                self.gaugelist[i].SetValue(20)
                        self.gaugelist[x].SetValue(ad2.AD_GetStatus())
                    elif SELECTEDALG == 'Gait Energy Image':
                        #print('x',x)
                        #print('ad2.AD_GetStatus()',ad2.AD_GetStatus())
                        x = ad2.AD_GetWalkNumber()
                        #prev = None
                        if 0 < x < 10:
                            for i in range(x):
                                self.gaugelist[i].SetValue(20)
                        self.gaugelist[x].SetValue(ad2.AD_GetStatus())
                    elif SELECTEDALG == 'SURF':
                        #print('x',x)
                        #print('ad2.AD_GetStatus()',ad2.AD_GetStatus())
                        x = ad2.AD_GetWalkNumber()
                        #prev = None
                        if 0 < x < 10:
                            for i in range(x):
                                self.gaugelist[i].SetValue(20)
                        self.gaugelist[x].SetValue(ad2.AD_GetStatus())
                TRAINEDMUTEX = True
                self.enableWalkButtons()
                if TRAINED:
                    wx.MessageBox('Training completed', 'Info', wx.OK | wx.ICON_INFORMATION)
                    BTNTEMP.SetValue(False)
                    for i in range(x):
                        self.gaugelist[i].SetValue(20)
                    BTNTEMP.SetLabel('Begin Training')
                    #self.enableWalkButtons()
                    TRAINED = False
                if STOPPED:
                    ad2.TerminateDifferences()
                    wx.MessageBox('Training stopped', 'Info', wx.OK | wx.ICON_INFORMATION)
                    STOPPED = False
                print('TRAINING FINISH')
    
            t1 = threading.Thread(target = statusfunc)
            t1.start()
        else:
            global STOPPED
            e.GetEventObject().SetLabel('Begin Training')
            STOPPED = True
            self.enableWalkButtons()

            
    #------------------------------------------------------------------------------
    def OnStart(self, e): 
        global TRAINED
        TRAINED = False
        self.e = e
        label = e.GetEventObject().GetLabel()
        print(label)
        state = e.GetEventObject().GetValue() 
        if state == True: 
            e.GetEventObject().SetLabel("Stop " + label[-1]) 
            if SELECTEDALG == 'Accumulated Differences':
                a = ADThread(int(label[-1]))
                a.start()
            elif SELECTEDALG == 'Gait Energy Image':
                g = GEIThread(int(label[-1]))
                g.start()
            elif SELECTEDALG == 'SURF':
                s = SURFThread(int(label[-1]))
                s.start()
            for btn in self.btnlist:
                if btn.GetLabel()[-1] != label[-1]:
                    print(btn.GetLabel())
#                    print(label)
                    btn.Disable()
        else: 
            if SELECTEDALG == 'Accumulated Differences':
                ad.TerminateCapture()
            elif SELECTEDALG == 'Gait Energy Image':
                gei.TerminateCapture()
            elif SELECTEDALG == 'SURF':
                surf.TerminateCapture()
#            e.GetEventObject().SetLabel("Training Walk " + label[-1])
            e.GetEventObject().Disable()            

            for btn in self.btnlist:
                if btn.GetLabel()[-1] != label[-1] and not RECORDEDNUMBER[int(btn.GetLabel()[-1])]:
                    btn.Enable()
                else:
                    RECORDEDNUMBER[int(btn.GetLabel()[-1])] = True
            e.GetEventObject().SetLabel("Walk Recorded " + label[-1])
            

    #------------------------------------------------------------------------------
    def Colours(self):
        self.panel.SetBackgroundColour(wx.BLACK)
        

    #------------------------------------------------------------------------------
    def disableWalkButtons(self):
        for btn in self.btnlist:
            btn.Disable()


    #------------------------------------------------------------------------------
    def enableWalkButtons(self):
        for btn in self.btnlist:
            btn.Enable()


##---------------------------------------------------------------------------------



###################################################################################
## BEGIN NEW TRAINING SET --> FIRST WINDOW
###################################################################################
class NewRecognitionSet(wx.Frame):
    """
    Class used for creating frames other than the main one
    """
    #------------------------------------------------------------------------------
    def __init__(self, title, parent=None):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.Show()
        panel = wx.Panel(self) 
                
        vbox = wx.BoxSizer(wx.VERTICAL) 
        
        hbox_fortext = wx.BoxSizer(wx.HORIZONTAL) 
        label1 = wx.StaticText(panel, 1234, "Enter the name of the subject")
        hbox_fortext.Add(label1, 0, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
        self.t1 = wx.TextCtrl(panel, size = (300, 24))         
        self.t1.SetFocus()
        hbox_fortext.Add(self.t1,0,wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
        self.t1.Bind(wx.EVT_TEXT,self.OnKeyTyped)        


        nm = wx.StaticBox(panel, -1, 'Please Select the Algorithm:') 
        nmSizer = wx.StaticBoxSizer(nm, wx.VERTICAL) 
        
        nmbox = wx.BoxSizer(wx.HORIZONTAL)         
        self.rb1 = wx.RadioButton(panel,-1, label = 'Accumulated Differences', style = wx.RB_GROUP)
        self.rb2 = wx.RadioButton(panel,-1, label = 'Gait Energy Image')
        self.rb3 = wx.RadioButton(panel,-1, label = 'SURF')        
        nmbox.Add(self.rb1, 0, wx.ALL|wx.CENTER, 5) 
        nmbox.Add(self.rb2, 0, wx.ALL|wx.CENTER, 5) 
        nmbox.Add(self.rb3, 0, wx.ALL|wx.CENTER, 5) 
    		
        nmSizer.Add(nmbox, 0, wx.ALL|wx.CENTER, 10)  
    		
        hbox = wx.BoxSizer(wx.HORIZONTAL) 
        okButton = wx.Button(panel, -1, 'ok')     		
        hbox.Add(okButton, 0, wx.ALL|wx.LEFT, 10) 
        cancelButton = wx.Button(panel, -1, 'cancel')     		
        hbox.Add(cancelButton, 0, wx.ALL|wx.LEFT, 10) 

        vbox.Add(hbox_fortext,0, wx.ALL|wx.CENTER, 5) 
        vbox.Add(nmSizer,0, wx.ALL|wx.CENTER, 5) 
        vbox.Add(hbox,0, wx.ALL|wx.CENTER, 5) 
        panel.SetSizer(vbox) 
        self.Centre() 
             
        panel.Fit() 
        vbox.SetSizeHints(self)    # Resize the window to fit the buttons ONLY
        self.Show()  
        
        # Binding the cancel and ok buttons to their events        
        self.Bind(wx.EVT_BUTTON, self.cancelOp, cancelButton)
        self.Bind(wx.EVT_BUTTON, self.TrainingAlgOp, okButton)

        # Binding the radio group to its event
        self.Bind(wx.EVT_RADIOBUTTON, self.OnRadiogroup)
        self.selectedAlg = "Accumulated Differences"

        
    #------------------------------------------------------------------------------
    # If the cancel button is pressed --> go back to the previous GUI screen
    def cancelOp(self, event):
        MyForm().Show()
        self.Close()


    #------------------------------------------------------------------------------
    # If the OK button is pressed --> Start the selected algorithm ***********
    def TrainingAlgOp(self, event):
        global SUBJECTNAME
        if SUBJECTNAME is '':
            wx.MessageBox("Please enter a subject's name", 'Warning', wx.OK | wx.ICON_EXCLAMATION)
        else:
            global SELECTEDALG
            ad.AD_RESET()
            surf.SURF_RESET()
            TrainFrame(title=SELECTEDALG)
            self.Close()

        
    #------------------------------------------------------------------------------
    # If a radio button is selected
    def OnRadiogroup(self, e):
       rb = e.GetEventObject()
       self.selectedAlg = rb.GetLabel()
       global SELECTEDALG
       SELECTEDALG = self.selectedAlg


    #------------------------------------------------------------------------------
    def OnKeyTyped(self, e):
        print(e.GetString())
        global SUBJECTNAME
        SUBJECTNAME = e.GetString()
##---------------------------------------------------------------------------------
    
    

###################################################################################
## BEGIN NEW TRAINING SET --> FIRST WINDOW
###################################################################################
class ViewTrainedSets(wx.Frame):
    """
    Class used for creating frames other than the main one
    """
    #------------------------------------------------------------------------------
    def __init__(self, title, parent=None):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.Show()
        self.panel = wx.Panel(self) 
                
        vbox = wx.BoxSizer(wx.VERTICAL) 

        AD_List = wx.StaticBox(self.panel, -1, 'Acumulated Differences') 
        AD_List_Sizer = wx.StaticBoxSizer(AD_List, wx.VERTICAL)
        GEI_List = wx.StaticBox(self.panel, -1, 'Gait Energy Image') 
        GEI_List_Sizer = wx.StaticBoxSizer(GEI_List, wx.VERTICAL)
        SURF_List = wx.StaticBox(self.panel, -1, 'SURF')
        SURF_List_Sizer = wx.StaticBoxSizer(SURF_List, wx.VERTICAL)
        AD_List.SetForegroundColour(wx.BLUE)
        GEI_List.SetForegroundColour(wx.BLUE)
        SURF_List.SetForegroundColour(wx.BLUE)
        
        AD_listbox = wx.BoxSizer(wx.VERTICAL)
        GEI_listbox = wx.BoxSizer(wx.VERTICAL)
        SURF_listbox = wx.BoxSizer(wx.VERTICAL)
        AD_PATH = PATH + '/AD'
        GEI_PATH = PATH + '/GEI'
        SURF_PATH = PATH + '/SURF'
        
        for subject in os.listdir(AD_PATH):
            if not re.match('trained-model', subject):
                name = wx.StaticText(self.panel, -1, subject, style = wx.ALIGN_CENTRE)
                AD_listbox.Add(name, 1, 0, 0)
        for subject in os.listdir(GEI_PATH):
            if not re.match('trained-model', subject):
                name = wx.StaticText(self.panel, -1, subject, style = wx.ALIGN_CENTRE)
                GEI_listbox.Add(name, 1, 0, 0)
        for subject in os.listdir(SURF_PATH):
            if not re.match('trained-model', subject):
                name = wx.StaticText(self.panel, -1, subject, style = wx.ALIGN_CENTRE)
                SURF_listbox.Add(name, 1, 0, 0)
    		
        AD_List_Sizer.Add(AD_listbox, 1, 0, 0)  
        GEI_List_Sizer.Add(GEI_listbox, 1, 0, 0)  
        SURF_List_Sizer.Add(SURF_listbox, 1, 0, 0)  
        
        list_hbox = wx.BoxSizer(wx.HORIZONTAL) 
        list_hbox.Add(AD_List_Sizer, 1, 0, 0) 
        list_hbox.Add(GEI_List_Sizer, 1, 0, 0) 
        list_hbox.Add(SURF_List_Sizer, 1, 0, 0) 

        hbox = wx.BoxSizer(wx.HORIZONTAL) 
        cancelButton = wx.Button(self.panel, -1, 'cancel')     		
        hbox.Add(cancelButton, 0, wx.ALL|wx.LEFT, 10) 

        vbox.Add(list_hbox,0, wx.ALL|wx.CENTER, 5)
        vbox.Add(hbox,0, wx.ALL|wx.CENTER, 5)
        self.panel.SetSizer(vbox)
        self.Centre() 
             
        self.panel.Fit() 
        vbox.SetSizeHints(self)    # Resize the window to fit the buttons ONLY
        self.Show()  
        
        # Binding the cancel and ok buttons to their events        
        self.Bind(wx.EVT_BUTTON, self.cancelOp, cancelButton)

        
    #------------------------------------------------------------------------------
    # If the cancel button is pressed --> go back to the previous GUI screen
    def cancelOp(self, event):
        MyForm().Show()
        self.Close()


##---------------------------------------------------------------------------------
    
    

###################################################################################
## BEGIN NEW TRAINING SET --> SECOND WINDOW
###################################################################################
class RetrainFrame2(wx.Frame):
    """
    Class used for creating frames other than the main one
    """

    #------------------------------------------------------------------------------
    def __init__(self, title, parent=None):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.count = 0 
        self.panel = wx.Panel(self) 
                
        vbox = wx.BoxSizer(wx.VERTICAL) 
        
        self.hboxlist = []
        self.gaugelist = []
        self.textlist = []
        
        #self.SetBackgroundColour(wx.BLACK)
        
        #self.Colours()
        
        
        for i in range(10):
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            self.hboxlist.append(hbox)
            gauge = wx.Gauge(self.panel, range = 20, size = (250, 25), style =  wx.GA_HORIZONTAL|wx.GA_SMOOTH) 
#            gauge.SetForegroundColour(wx.BLUE)
#            gauge.SetLabel('Not trained')
#            gauge.SetBackgroundColour(wx.BLACK)
#            gauge.SetLabelText('Not trained')
            self.gaugelist.append(gauge)
            nm = wx.StaticText(self.panel, -1, 'Walk {}: Not processed'.format(i), style = wx.ALIGN_CENTRE) 
            self.textlist.append(nm)
            #self.Bind(wx.EVT_TOGGLEBUTTON, self.OnStart, self.btnlist[i]) 

            self.hboxlist[i].Add(self.textlist[i], 1, 1, 0) 
            self.hboxlist[i].Add(self.gaugelist[i], proportion = 1, flag = wx.ALIGN_CENTRE) 
            
            global RECORDEDNUMBER
            RECORDEDNUMBER.append(False)
            
            vbox.Add(self.hboxlist[i], 1, 1, 0) 
             
        self.status = wx.StaticText(self.panel, -1, 'Status: Not training.', style = wx.ALIGN_CENTRE) 
        self.status.SetForegroundColour(wx.BLUE)
        vbox.Add(self.status, 2, 1, 0)
        self.begintrainingbtn = wx.ToggleButton(self.panel, label = "Begin Training!")
        self.donebtn = wx.Button(self.panel, label = "Done!")
        self.Bind(wx.EVT_BUTTON, self.FinishedTraining, self.donebtn)
        self.Bind(wx.EVT_TOGGLEBUTTON, self.BeginTraining, self.begintrainingbtn)
        hboxforbuttons = wx.BoxSizer(wx.HORIZONTAL)
        hboxforbuttons.Add(self.begintrainingbtn, proportion = 1, flag = wx.ALIGN_CENTRE) 
        hboxforbuttons.Add(self.donebtn, proportion = 1, flag = wx.ALIGN_CENTRE)
        vbox.Add(hboxforbuttons, 1, 1, 0)
        
        vbox.SetSizeHints(self)    # Resize the window to fit the buttons ONLY
        self.SetSizer(vbox)

        self.Centre() 
        self.Show()   
    		

    #------------------------------------------------------------------------------
    def FinishedTraining(self, e): 
        print('finished!!!')
        MyForm().Show()
        self.Close()


    #------------------------------------------------------------------------------
    def BeginTraining(self, e): 
        state = e.GetEventObject().GetValue() 
        if state == True:
            global TRAINED
            TRAINED = False
            e.GetEventObject().SetLabel('Stop Training')
            
            for i in range(10):
                self.gaugelist[i].SetValue(0)
            
            # TRAINING THREAD
            def trainthread():
                global SELECTEDALG
                if SELECTEDALG == 'Accumulated Differences':
                    ad2.AD_Begin()
                    ad2.AD_fvecs()
                    global TRAINED
                    TRAINED = True
                    print('trainthread finished')
                elif SELECTEDALG == 'Gait Energy Image':
                    gei2.GEI_Begin()
                    gei2.GEI_fvecs()
                    TRAINED = True
                    print('trainthread finished')
                if SELECTEDALG == 'SURF':
                    surf2.SURF_Begin()
                    surf2.SURF_train()
                    TRAINED = True
                    print('trainthread finished')

            
            # GETTING STATUS THREAD
            def statusfunc():
                global TRAINEDMUTEX, TRAINED, STOPPED
                TRAINED = False
                STOPPED = False
                print('entered statusfunc')
                BTNTEMP = e.GetEventObject()
                while not TRAINED and not STOPPED:
                    global SELECTEDALG
                    #print(SELECTEDALG)
                    #print(SELECTEDALG == 'Accumulated Differences')
                    if SELECTEDALG == 'Accumulated Differences':
                        #print('entered statusfunc3')
                        #print('x',x)
                        #print('ad2.AD_GetStatus()',ad2.AD_GetStatus())
                        x = ad2.AD_GetWalkNumber()
                        #prev = None
                        for i in range(10):
                            if i < x:
                                self.gaugelist[i].SetValue(20)
                            elif i > x:
                                self.gaugelist[i].SetValue(0)
                        self.gaugelist[x].SetValue(ad2.AD_GetStatus())
                        self.status.SetLabel(ad2.getStatus2())
                        #------------------------------------------------------
                    elif SELECTEDALG == 'Gait Energy Image':
                        #print('entered statusfunc4')
                        #print('x',x)
                        #print('ad2.AD_GetStatus()',ad2.AD_GetStatus())
                        x = gei2.GEI_GetWalkNumber()
                        #prev = None
                        for i in range(10):
                            if i < x:
                                self.gaugelist[i].SetValue(20)
                            elif i > x:
                                self.gaugelist[i].SetValue(0)
                        self.gaugelist[x].SetValue(gei2.GEI_GetStatus())
                        self.status.SetLabel(gei2.getStatus2())
                        #------------------------------------------------------                        
                    elif SELECTEDALG == 'SURF':
                        #print('entered statusfunc5')
                        #print('x',x)
                        #print('ad2.AD_GetStatus()',ad2.AD_GetStatus())
                        x = surf2.SURF_GetWalkNumber()
                        #prev = None
                        for i in range(10):
                            if i < x:
                                self.gaugelist[i].SetValue(20)
                            elif i > x:
                                self.gaugelist[i].SetValue(0)
                        self.gaugelist[x].SetValue(surf2.SURF_GetStatus())
                        self.status.SetLabel(surf2.getStatus2())
                        #------------------------------------------------------
                    self.status.SetForegroundColour(wx.RED)
                TRAINEDMUTEX = True
                #--------------------------------------------------------------
                if TRAINED:
                    wx.MessageBox('Training completed', 'Info', wx.OK | wx.ICON_INFORMATION)
                    BTNTEMP.SetValue(False)
                    for i in range(x):
                        self.gaugelist[i].SetValue(20)
                    BTNTEMP.SetLabel('Begin Training')
                    #self.enableWalkButtons()
                    TRAINED = False
                    self.status.SetLabel('Training completed!')
                    self.status.SetForegroundColour(wx.BLUE)
                #--------------------------------------------------------------
                if STOPPED:
                    if SELECTEDALG ==  'Accumulated Differences':
                        ad2.TerminateDifferences()
                        wx.MessageBox('Training stopped', 'Info', wx.OK | wx.ICON_INFORMATION)
                        STOPPED = False
                        ad2.AD_RESET()
                        #------------------------------------------------------
                    elif SELECTEDALG ==  'Gait Energy Image':
                        gei2.TerminateDifferences()
                        wx.MessageBox('Training stopped', 'Info', wx.OK | wx.ICON_INFORMATION)
                        STOPPED = False
                        gei2.GEI_RESET()
                        #------------------------------------------------------
                    elif SELECTEDALG ==  'SURF':
                        surf2.TerminateDifferences()
                        wx.MessageBox('Training stopped', 'Info', wx.OK | wx.ICON_INFORMATION)
                        STOPPED = False
                        surf2.SURF_RESET()
                        #------------------------------------------------------
                    self.status.SetLabel('Training stopped!')
                    self.status.SetForegroundColour(wx.BLUE)
                #--------------------------------------------------------------
                print('TRAINING FINISH')
    
            t0 = threading.Thread(target = trainthread)
            t1 = threading.Thread(target = statusfunc)
            t0.start()
            t1.start()
        else:
            global STOPPED
            e.GetEventObject().SetLabel('Begin Training')
            STOPPED = True

            
    #------------------------------------------------------------------------------
    def OnStart(self, e): 
        global TRAINED
        TRAINED = False
        self.e = e
        label = e.GetEventObject().GetLabel()
        print(label)
        state = e.GetEventObject().GetValue() 
        if state == True: 
            e.GetEventObject().SetLabel("Stop " + label[-1]) 
            if SELECTEDALG == 'Accumulated Differences':
                a = ADThread(int(label[-1]))
                a.start()
            elif SELECTEDALG == 'Gait Energy Image':
                g = GEIThread(int(label[-1]))
                g.start()
            elif SELECTEDALG == 'SURF':
                s = SURFThread(int(label[-1]))
                s.start()
            for btn in self.btnlist:
                if btn.GetLabel()[-1] != label[-1]:
                    print(btn.GetLabel())
#                    print(label)
                    btn.Disable()
        else: 
            if SELECTEDALG == 'Accumulated Differences':
                ad.TerminateCapture()
            elif SELECTEDALG == 'Gait Energy Image':
                gei.TerminateCapture()
            elif SELECTEDALG == 'SURF':
                surf.TerminateCapture()
#            e.GetEventObject().SetLabel("Training Walk " + label[-1])
            e.GetEventObject().Disable()            

            for btn in self.btnlist:
                if btn.GetLabel()[-1] != label[-1] and not RECORDEDNUMBER[int(btn.GetLabel()[-1])]:
                    btn.Enable()
                else:
                    RECORDEDNUMBER[int(btn.GetLabel()[-1])] = True
            e.GetEventObject().SetLabel("Walk Recorded " + label[-1])
            

    #------------------------------------------------------------------------------
    def Colours(self):
        self.panel.SetBackgroundColour(wx.BLACK)

##---------------------------------------------------------------------------------



###################################################################################
## BEGIN NEW TRAINING SET --> FIRST WINDOW
###################################################################################
class RetrainFrame(wx.Frame):
    """
    Class used for creating frames other than the main one
    """
    #------------------------------------------------------------------------------
    def __init__(self, title, parent=None):
        wx.Frame.__init__(self, parent=parent, title=title)
        self.Show()
        panel = wx.Panel(self) 
                
        vbox = wx.BoxSizer(wx.VERTICAL) 

        nm = wx.StaticBox(panel, -1, 'Please Select the Algorithm:') 
        nmSizer = wx.StaticBoxSizer(nm, wx.VERTICAL) 
        
        nmbox = wx.BoxSizer(wx.HORIZONTAL)         
        self.rb1 = wx.RadioButton(panel,-1, label = 'Accumulated Differences', style = wx.RB_GROUP)
        self.rb2 = wx.RadioButton(panel,-1, label = 'Gait Energy Image')
        self.rb3 = wx.RadioButton(panel,-1, label = 'SURF')        
        nmbox.Add(self.rb1, 0, wx.ALL|wx.CENTER, 5) 
        nmbox.Add(self.rb2, 0, wx.ALL|wx.CENTER, 5) 
        nmbox.Add(self.rb3, 0, wx.ALL|wx.CENTER, 5) 
    		
        nmSizer.Add(nmbox, 0, wx.ALL|wx.CENTER, 10)  
    		
        hbox = wx.BoxSizer(wx.HORIZONTAL) 
        okButton = wx.Button(panel, -1, 'ok')     		
        hbox.Add(okButton, 0, wx.ALL|wx.LEFT, 10) 
        cancelButton = wx.Button(panel, -1, 'cancel')     		
        hbox.Add(cancelButton, 0, wx.ALL|wx.LEFT, 10) 

        vbox.Add(nmSizer,0, wx.ALL|wx.CENTER, 5) 
        vbox.Add(hbox,0, wx.ALL|wx.CENTER, 5) 
        panel.SetSizer(vbox) 
        self.Centre() 
             
        panel.Fit() 
        vbox.SetSizeHints(self)    # Resize the window to fit the buttons ONLY
        self.Show()  
        
        # Binding the cancel and ok buttons to their events        
        self.Bind(wx.EVT_BUTTON, self.cancelOp, cancelButton)
        self.Bind(wx.EVT_BUTTON, self.TrainingAlgOp, okButton)

        # Binding the radio group to its event
        self.Bind(wx.EVT_RADIOBUTTON, self.OnRadiogroup)
        self.selectedAlg = "Accumulated Differences"
        global SELECTEDALG
        SELECTEDALG = self.selectedAlg

        
    #------------------------------------------------------------------------------
    # If the cancel button is pressed --> go back to the previous GUI screen
    def cancelOp(self, event):
        MyForm().Show()
        self.Close()


    #------------------------------------------------------------------------------
    # If the OK button is pressed --> Start the selected algorithm ***********
    def TrainingAlgOp(self, event):
        RetrainFrame2(title=SELECTEDALG)
        self.Close()

        
    #------------------------------------------------------------------------------
    # If a radio button is selected
    def OnRadiogroup(self, e):
       rb = e.GetEventObject()
       self.selectedAlg = rb.GetLabel()
       global SELECTEDALG
       SELECTEDALG = self.selectedAlg
       print('called')
##---------------------------------------------------------------------------------



###################################################################################
## MAIN FRAME --> START WINDOW
###################################################################################
class MyForm(wx.Frame):

    #------------------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, title='Gait Recognition Software!')
        
        # Create the buttons contained within the class itself with their titles
        self.NewTrainingBtn = wx.Button(self, -1, 'Begin new Training Set')
        self.RetrainBtn = wx.Button(self, -1, 'Retrain GR models')
        self.NewRecogBtn = wx.Button(self, -1, 'Run New Recognition')
        self.NewVerifyBtn = wx.Button(self, -1, 'Run New Verification')
        self.ViewTrainedBtn = wx.Button(self, -1, 'View Trained Sets')
        self.ViewHistoryBtn = wx.Button(self, -1, 'View History')
        
        # Create the sizer object to add the buttons to
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.NewTrainingBtn, 0, wx.EXPAND, 0)
        sizer.Add(self.RetrainBtn, 0, wx.EXPAND, 0)
        sizer.Add(self.NewRecogBtn, 0, wx.EXPAND, 0)
        sizer.Add(self.NewVerifyBtn, 0, wx.EXPAND, 0)
        sizer.Add(self.ViewTrainedBtn, 0, wx.EXPAND, 0)
        sizer.Add(self.ViewHistoryBtn, 0, wx.EXPAND, 0)
        sizer.SetSizeHints(self)    # Resize the window to fit the buttons ONLY
        self.SetSizer(sizer)
        
        # Create the action listeners for the buttons
        self.Bind(wx.EVT_BUTTON, self.newTraining, self.NewTrainingBtn)
        self.Bind(wx.EVT_BUTTON, self.retrain, self.RetrainBtn)
        self.Bind(wx.EVT_BUTTON, self.newRecognition, self.NewRecogBtn)
        self.Bind(wx.EVT_BUTTON, self.newTraining, self.NewVerifyBtn)
        self.Bind(wx.EVT_BUTTON, self.viewTrained, self.ViewTrainedBtn)
        self.Bind(wx.EVT_BUTTON, self.newTraining, self.ViewHistoryBtn)

        self.Centre() 


    #------------------------------------------------------------------------------
    def newTraining(self, event):
        print(os.getcwd())
        title = 'Begin New Training Set'
        NewRecognitionSet(title=title)
        self.Close()

        
    #------------------------------------------------------------------------------
    def viewTrained(self, event):
        print(os.getcwd())
        title = 'Trained Sets'
        ViewTrainedSets(title=title)
        self.Close()

        
    #------------------------------------------------------------------------------
    def newRecognition(self, event):
        # Do something
        print ('onOK handler')
        import GEI_Algorithm
        GEI_Algorithm.mainrun()

        
    #------------------------------------------------------------------------------
    def retrain(self, event):
        title = 'Retrain a model'
        RetrainFrame(title=title)
        self.Close()
#----------------------------------------------------------------------------------








###################################################################################
## DRIVER --> RUN PROGRAM
###################################################################################
if __name__ == '__main__':
    print (wx.ART_INFORMATION)
    app = wx.App()
    frame = MyForm().Show()
    app.MainLoop()
    del app
##---------------------------------------------------------------------------------
