import matplotlib as mpl
import functools
mpl.rc('lines', linewidth=2, color='r')
mpl.rc('font', size=16)
mpl.rc('axes', labelsize=18, grid=True)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
import matplotlib.pyplot as plt

#Create a decorator for remaining plots
def plot_skeleton( plot_func):
    ''' This is a decorator function which can be used as wrapper to extend any plot functions
        plot_func -  specify the plot function with arguments
        plot_type - type string ('Single Panel'/'Multi Panel'
        'Single Panel' - plot all the datasets in single plot
        'Multi Panel'   - plot each dataset in a subplot
    '''
    @functools.wraps(plot_func)    #preserves the information of plot_func
    def plot_wrapper(self, *args, **kwargs):
        
        if self.plot_type=='Single Panel': #0th argument is the class instance
            figlength = 10   
            figwidth  = 10            
            nrows     = 1
            ncols     = 1
        else:
            numfigs = len((self.x).keys())
            figlength = 14
            figwidth  = 7*(numfigs//2)
            nrows   = numfigs//2
            ncols   = 2


        if not self.figsize==None:
            figlength = self.figsize[0]
            figwidth  = self.figsize[1]

        fig = plt.figure(figsize=(figlength, figwidth))
        
        plot_func(self, *args, **kwargs)
        fig.suptitle(self.figure_title, fontsize=20)

        if self.figname!=None: plt.savefig(self.figname, dpi=300)
        plt.close()
        return
    return plot_wrapper

class create_plot():
   
    def __init__(x, y, xlabel, ylabel, legend=None,  fig_properties=None):
        '''
        x : type dict
    	Dictionary with keys x1, x2,  .... where each xi's are x-data sets
        
    	y : type dict
    	Dictionary with keys y1, y2, .... where each yi's are y-data sets
    	
    	xlabel: type string
    	Label of x axis
    	
    	ylabel: type string
    	Label of y axis
    
    	legend: type dict
    	Legend description for each x-y dataset with keys label1, label2....
    
    	fig_properties: type dict
    	keys of dictionary - plot_title (list of string), figure_title (string), figure_name (string), 
        figure_size (list/array), panel_type - 'Single Panel'/'Multi Panel'
        '''
    
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend

        
	    
        if fig_properties!=None:
            
            if 'figure_size' in fig_properties.keys():
                self.figsize=fig_properties['figsize']  
            
            if 'figure_title' in fig_properties.keys():
            	self.fig_title=fig_properties['figure-title']
            
            if 'figure_name' in fig_properties.keys():
            	self.figname=fig_properties['figname']
            
            if 'panel_type' in fig_properties.keys():
                self.panel_type=fig_properties['panel_type']
        else:
            self.figure_title=None
            self.figname=None
            self.figsize=None
            self.panel_type='Single Panel'
    
    

    
    @plot_skeleton
    def plot_xy(self, plot_properties=None):
        """ 
        Generates y vs x plots and subplots for given datasets. 
        
        Parameters :
    
        plot_properties - Type dict
        Contain following keys and values - 
            1. 'xlim' = list/array of type [xmin, xmax]
            2. 'ylim' = list/array of type [ymin, ymax]
            3. 'xscale' = 'log'
            4. 'yscale' = 'log'
            5. 'title'  = list of string  for each subplot title
            6. 'plot-type' = 'line'/'scatter'
    
        """ 
        lines = ["-", "--", ":", "-."]
        linecycler = cycle(lines)
        
        num_datasets = len(self.x.keys())
        if self.panel_type=='Single Panel':
            nrows, ncols = 1, 1
        else: 
            nrows, ncols = num_datasets//2,2 
    
        for i in range(num_datasets):
            
            j = i+1 
            xj, yj = self.x['x%d'%j], self.y['y%d'%j]
    
            assert(len(xj)==len(yj)), (Fore.red + 'Length of x%d and y%d datasets are not the same. Please check the data'%(j, j))
            
            if self.panel_type=='Multi Panel':
                plt.subplot(nrows, ncols, i)
                plt.xlabel(self.xlabel)
                plt.ylabel(self.ylabel)
        
            plt.plot(xj, yj, next(linecycler), legend=self.legend['label%d'%j])
            plt.legend()
            plt.grid(True)            
        
       
