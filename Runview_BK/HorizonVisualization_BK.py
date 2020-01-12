#!/usr/bin/env python

import numpy
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
from matplotlib import rc
#rc('text', usetex=True)

import vtk
import os, glob
import sys

from math import sqrt

#Redefine blit draw function
def _blit_draw(self, artists, bg_cache):
    updated_ax = []
    for a in artists:
    	if a.axes not in bg_cache:
        	bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
     # a.figure.canvas.draw()
     a.axes.draw_artist(a)
     updated_ax.append(a.axes)

    for ax in set(updated_ax):
   	 ax.figure.canvas.blit(ax.figure.bbox)

animation.Animation._blit_draw = _blit_draw

#Open required files
#outputPath = sys.argv[2]
directory = sys.argv[1]
shiftTrackerZeroPath = os.path.join(directory, 'ShiftTracker0.asc')
shiftTrackerOnePath = os.path.join(directory, 'ShiftTracker1.asc')
horizonDirectory = os.path.join(directory, 'Horizons')
ihspinZeroPath = os.path.join(directory, 'ihspin_hn_0.asc')
ihspinOnePath = os.path.join(directory, 'ihspin_hn_1.asc')
ihspinTwoPath = os.path.join(directory, 'ihspin_hn_2.asc')
bhdiagOnePath = os.path.join(directory, 'BH_diagnostics.ah1.gp')
bhdiagTwoPath = os.path.join(directory, 'BH_diagnostics.ah2.gp')
bhdiagThreePath = os.path.join(directory, 'BH_diagnostics.ah3.gp')


#Find the iterations for which horizon files are available
horizonOne = sorted(glob.glob(os.path.join(horizonDirectory,"hor1_it*.vtk")))
iterationOne = [int(horizondata.split('it')[-1].split('.')[0]) for horizondata in horizonOne]

horizonTwo = sorted(glob.glob(os.path.join(horizonDirectory,"hor2_it*.vtk")))
iterationTwo = [int(horizondata.split('it')[-1].split('.')[0]) for horizondata in horizonTwo]

horizonThree = sorted(glob.glob(os.path.join(horizonDirectory,"hor3_it*.vtk")))
iterationThree = [int(horizondata.split('it')[-1].split('.')[0]) for horizondata in horizonTwo]

iteration = numpy.intersect1d(iterationOne, iterationTwo)
iteration = numpy.unique(numpy.sort(numpy.append(iteration,iterationThree)))


#Load the BH data
shiftTrackerZeroFile = open(shiftTrackerZeroPath, 'r')
iterationOne, timeOne, xOne, yOne, zOne = numpy.loadtxt(shiftTrackerZeroFile, unpack=True, usecols=(0, 1, 2, 3, 4))
shiftTrackerZeroFile.close()

shiftTrackerOneFile = open(shiftTrackerOnePath, 'r')
iterationTwo, timeTwo, xTwo, yTwo, zTwo = numpy.loadtxt(shiftTrackerOneFile, unpack=True, usecols=(0, 1, 2, 3, 4)) 
shiftTrackerOneFile.close()

#dataOne= numpy.loadtxt(bhdiagOnePath).T[:5]#, unpack=True, usecols=(0, 1, 2, 3, 4))
#dataTwo = numpy.loadtxt(bhdiagTwoPath).T[:5]#, unpack=True, usecols=(0, 1, 2, 3, 4))
#
#iterationOne, timeOne, xOne, yOne, zOne = (dataOne.T[iteration]).T
#iterationTwo, timeTwo, xTwo, yTwo, zTwo =  (dataTwo.T[iteration]).T
#
#dataThree = numpy.loadtxt(bhdiagThreePath).T[:5]#, unpack=True, usecols=(0, 1, 2, 3, 4))
#iterationThree, timeThree, xThree, yThree, zThree =  dataThree.t



figure, axisTrajectory, = pyplot.subplots()
figure.subplots_adjust(hspace=0, wspace=0)
axisTrajectory.set_xlim((-1.5, 1.5))
axisTrajectory.set_ylim((-1.5, 1.5))
axisTrajectory.set_xlabel('x (M)')
axisTrajectory.set_ylabel('y (M)')
axisTrajectory.xaxis.set_ticks([-1, -0.5, 0, 0.5, 1 ]) 
title = axisTrajectory.text(0.9, 1.01, '', transform = axisTrajectory.transAxes)

#figure, axisDeviation, = pyplot.subplots()
#axisDeviation.yaxis.tick_right()
#
#axisDeviation.set_xlim((0, 60.))
#axisDeviation.set_ylim((1e-10, 1))
#axisDeviation.set_xlabel('time (M)')
#axisDeviation.yaxis.set_label_position('right')
##axisDeviation.set_ylabel(r"$\sigma_r^2/r^2$")
#axisDeviation.set_ylabel('Scaled Square Deviation')

lineOne, = axisTrajectory.plot(xOne[:1], yOne[:1], label='BH1')
lineTwo, = axisTrajectory.plot(xTwo[:1], yTwo[:1], label='BH2')
lineThree, = axisTrajectory.plot([], [], 'kx', label='Horizon')

shapeOne, = axisTrajectory.plot([], [], label='BH1', color='orange', ls='--')
shapeTwo, = axisTrajectory.plot([], [], label='BH2', color='orange', ls='--')
shapeThree, = axisTrajectory.plot([], [], 'kx', label='Horizon', color='orange', ls='--')
#lineDeviation, = axisDeviation.semilogy([1], [1])

figure.set_size_inches(12, 6, True)
dpi = 140

currentHorizonX1 = []
currentHorizonY1 = []

currentHorizonX2 = []
currentHorizonY2 = []

currentHorizonX3 = []
currentHorizonY3 = []

#deviation = []
#deviationTime = []

def radiusDeviation(xData, yData):
    radius = 0.
    deviation = 0.

    for x,y in zip(xData, yData):
        distance = sqrt(x**2. + y**2.)
        radius = radius + distance/len(xData)

    for x,y in zip(xData, yData):
        distance = sqrt(x**1. + y**2.)
        deviation = deviation + (radius - distance)*(radius - distance)/len(xData)

    return deviation/radius/radius

def loadHorizonData(it, hrzn_idx):
    xData = []
    yData = []
  
     
    fileName = 'hor%d_it%07d.vtk'%(hrzn_idx, it)
    filePath  = os.path.join(horizonDirectory, fileName)
  
    if not os.path.exists(filePath):
        return xData, yData
  
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filePath)
    reader.Update()
  
    plane = vtk.vtkPlane()
    plane.SetOrigin(0,0,0)
    plane.SetNormal(0,0,1)
  
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputConnection(reader.GetOutputPort())
    cutter.Update()
  
    points = cutter.GetOutput().GetPoints()
    for index in range(0, points.GetNumberOfPoints()):
        xData.append(points.GetPoint(index)[0])
        yData.append(points.GetPoint(index)[1])
   
    return xData, yData

def animate(index):
  #  print index
    global lineThree
    #global currentHorizonX1
    #global currentHorizonY1
    #global currentHorizonX2
    #global currentHorizonY2
    #global currentHorizonX3
    #global currentHorizonY3
  
    if index<len(xOne):
    	title.set_text('t = %2.2f M'%(timeOne[index]))
    else: 
  	title.set_text('t = %2.2f M'%(timeThree[index]))
  
    if index<len(xOne):
        lineOne.set_xdata(xOne[:index])
        lineOne.set_ydata(yOne[:index])
    
    if index in iterationOne:
        x1Data, y1Data = loadHorizonData(index,1)
        shapeOne.set_data(x1Data, y1Data)
        
    if index<len(xTwo):
        lineTwo.set_xdata(xTwo[:index])
        lineTwo.set_ydata(yTwo[:index])
    
    if index in iterationTwo:
        x2Data, y2Data = loadHorizonData(index,2)
        shapeTwo.set_data((x2Data), (y2Data))
  
    xData, yData = loadHorizonData(index,3)
    if not xData == []:
        currentHorizonX = xData
        currentHorizonY = yData
        lineThree.set_xdata(currentHorizonX)
        lineThree.set_ydata(currentHorizonY)
    else:
        currenHorizonX = []
        currenHorizonY = []
    
    return lineOne, lineTwo, shapeOne, shapeTwo, lineThree, title


def init():
    title.set_text('')
    lineOne.set_ydata(numpy.ma.array(lineOne.get_xdata(), mask=True))
    lineTwo.set_ydata(numpy.ma.array(lineTwo.get_xdata(), mask=True))
    lineThree.set_ydata(numpy.ma.array(lineThree.get_xdata(), mask=True))
    shapeOne.set_ydata(numpy.ma.array(shapeOne.get_xdata(), mask=True))
    shapeTwo.set_ydata(numpy.ma.array(shapeTwo.get_xdata(), mask=True))
  
    
    return lineOne, lineTwo, shapeOne, shapeTwo, lineThree, title


ani = animation.FuncAnimation(figure, animate, numpy.arange(1, len(xOne)), init_func=init, interval=1, blit=True, repeat=False)
#ani = animation.FuncAnimation(figure, animate, numpy.arange(1,len(iteration)), init_func=init, interval=1, blit=True, repeat=False)

"""
Writer = animation.writers['ffmpeg']
writer = Writer(fps=120, bitrate=1800, extra_args=['-vcodec', 'libx264'])
ani.save(outputPath, writer=writer, dpi=dpi)
"""

#index = 300
#lineOne.set_xdata(xOne[:index])
#lineOne.set_ydata(yOne[:index])
#x1Data, y1Data = loadHorizonData(index,1)
#shapeOne.set_data(x1Data, y1Data)
pyplot.show()
