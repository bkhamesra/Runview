import plotly
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

#semi-janky overkill
"""
class SimpleRead():
  def __init__(self,defbin="/home/rudall/Runview/TestCase/OutputDirectory/QC0_p1_|11_M192-all/data/", filepath=None, filename=None, fildir=None):
    self.dbin = defbin
    if filepath == None:
      inbin = str(input("If the file is in the default bin, enter 1. If in a subdirectory, enter 2. If outside the bin, enter 3:\n"))
      print(inbin)
      if inbin == "1": #file is in the default bin
	filename= str(input("Enter file name (in quotes!):\n"))
	filedir = defbin 
	filepath = defbin+filename
      elif inbin == "2":
	subdir = str(input("Enter subdirectory name (or path, if nested; in quotes!):\n")) #for subdirectory inside default bin
	self.subdir = subdir #for if you want to make multiple objects in the same subdirectory, and don't feel like typing it over and over
	filedir = defbin + subdir
	filename = str(input("Enter file name (in quotes!):\n"))
	filepath = filedir + filename
      elif inbin == "3":  
	filedir = str(input("Enter full file directory(in quotes!):\n")) # full form, bit of a pain to use though   
	filename = str(input("Enter file name (in quotes!):\n")) 
	filepath = filedir+filename
      else:
	print("Invalid response, please try again")
    self.fname = filename
    self.filep = filepath
    self.fdir = filedir
    with open(filepath,"r") as readfile:#pretty simple
      balist = readfile.readlines() #a list with each element being a given line of text
    self.text = balist
""" 
def readdata(bin,filename): #reading the data into a convenient list
  with open(bin+filename,"r") as readfile:
    textlist = readfile.readlines()
  return textlist

def gendatarray(datlist):
  l = 0
  while True: #find where the data starts
    if datlist[l][0] == "#":
      l += 1
    else:
      break 
  testline = datlist[l] #figuring out how many columns the data has
  testlist = testline.split()
  cols = len(testlist)
  datarray = np.zeros((len(datlist)-l,cols))
  for i in range(l,len(datlist)):
    linelist = datlist[i].split()
    for n in range(len(linelist)):
      datarray[i-l,n] = linelist[n]
  return datarray

def plotxvst(datarray1,datarray2):
  trace0 = go.Scatter(
    x = datarray1[...,1:2].transpose()[0], #where x is time and y is x
    y = datarray1[...,2:3].transpose()[0],
    mode = "lines",
    name = "BH1"
  )
  
  trace1 = go.Scatter(
    x = datarray2[...,1:2].transpose()[0], #where x is time and y is x
    y = datarray2[...,2:3].transpose()[0],
    mode = "lines",
    name = "BH2"
  )
  
  BHdata = [trace0, trace1]
  layout = go.Layout(
    title = "X vs. Time for BBH System",
    hovermode = "closest",
    xaxis = dict(
      title = "Time"
    ),
    yaxis = dict(
      title = "X"
    )
  )
  
  plot = go.Figure(data=BHdata, layout=layout)
  py.plot(plot, filename="xvstplot.html")
  

def plotyvst(datarray1,datarray2):
  trace0 = go.Scatter(
    x = datarray1[...,1:2].transpose()[0], #where x is time and y is y
    y = datarray1[...,3:4].transpose()[0],
    mode = "lines",
    name = "BH1"
  )
  
  trace1 = go.Scatter(
    x = datarray2[...,1:2].transpose()[0], #where x is time and y is y
    y = datarray2[...,3:4].transpose()[0],
    mode = "lines",
    name = "BH2"
  )
  
  BHdata = [trace0, trace1]
  layout = go.Layout(
    title = "Y vs. Time for BBH System",
    hovermode = "closest",
    xaxis = dict(
      title = "Time"
    ),
    yaxis = dict(
      title = "Y"
    )
  )
  
  plot = go.Figure(data=BHdata, layout=layout)
  py.plot(plot, filename="yvstplot.html")

def plotxvsy(datarray1,datarray2):
  trace0 = go.Scatter(
    x = datarray1[...,2:3].transpose()[0], #where x is x and y is y
    y = datarray1[...,3:4].transpose()[0],
    mode = "lines",
    name = "BH1"
  )
  
  trace1 = go.Scatter(
    x = datarray2[...,2:3].transpose()[0], #where x is x and y is y
    y = datarray2[...,3:4].transpose()[0],
    mode = "lines",
    name = "BH2"
  )
  
  BHdata = [trace0, trace1]
  layout = go.Layout(
    title = "X vs. Y for BBH System",
    hovermode = "closest",
    xaxis = dict(
      title = "X"
    ),
    yaxis = dict(
      title = "Y"
    )
  )
 
  plot = go.Figure(data=BHdata, layout=layout)
  py.plot(plot, filename="xvsyplot.html")
    
def plotsepvst(datarray1,datarray2):
  
  x1 = datarray1[...,2:3].transpose()[0]
  x2 = datarray2[...,2:3].transpose()[0]
  y1 = datarray1[...,3:4].transpose()[0]
  y2 = datarray2[...,3:4].transpose()[0] 
  
  length = x1.shape[0]
  separray = np.zeros((1,length))
  for num in range(length):
    xsepsq = (x1[num] - x2[num])**2
    ysepsq = (y1[num] - y2[num])**2
    r = np.sqrt(xsepsq + ysepsq)
    separray[0,num] = r
    
  trace0 = go.Scatter(
    x = datarray1[...,1:2].transpose()[0], 
    y = separray[0],
    mode = "lines",
    name = "BH1"
  )
  
  BHdata = [trace0]
  layout = go.Layout(
    title = "Separation vs. Time for BBH System",
    hovermode = "closest",
    xaxis = dict(
      title = "Time"
    ),
    yaxis = dict(
      title = "Separation"
    )
  )
  
  plot = go.Figure(data=BHdata, layout=layout)
  py.plot(plot, filename="sepvstplot.html") 

  
   

binQC0 = "/home/rudall/Runview/TestCase/OutputDirectory/QC0_p1_l11_M192-all/data/"

BH1 = readdata(binQC0, "ShiftTracker0.asc")
BH2 = readdata(binQC0, "ShiftTracker1.asc")

BH1A = gendatarray(BH1)
BH2A = gendatarray(BH2)
  
plotxvst(BH1A,BH2A)
plotyvst(BH1A,BH2A)
plotxvsy(BH1A,BH2A)
plotsepvst(BH1A,BH2A)