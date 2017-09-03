from metadata import *
from shutil import copytree
from CommonFunctions import *

def webpage_data(metadata, parfile, locate_merger):
	
	
	html_txt = """

<!DOCTYPE html> \n 
<html lang=\"en-US\"> \n 
<head> \n 
<style> \n 
/* body {{ \n   background-image: url(http://wallpapercave.com/wp/zPuoYm9.jpg); \n   width: 1000%; \n  }}-->*/\n
h1 {{color:rgb(0, 132, 255);margin: 50px;}} \n 
h2 {{color:rgb(110, 145, 236);margin: 50px;}} \n 
p {{color:ref;margin: 50px;}}\n  
ul {{margin: 50px}}\n  </style>\n  
<title> Simulation </title> \n 
</head>\n <body>\n \n 
<h1 style=\"text-align:center; font-size:300% \n \"> {} </h1>""".format('BBH')

	html_txt = html_txt + """
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> <a href=\"Metadata.html\">Metadata </a></h2>
\n <ul>
\n <li> Simulation Name = {}</li>
\n <li> Sim-Type = {}</li> 
\n <li> Mass Ratio = {}</li>
\n <li> Initial Separation = {}</li>
\n <li> Spin of BH1 = {}</li>
\n <li> Spin of BH2 = {}</li>
\n </ul>	""".format(metadata['alternative-names'], metadata['simulation-type'], metadata['mass-ratio'], metadata['init_sep'], metadata['spin1'], metadata['spin2'])


	if locate_merger:
		html_txt = html_txt + """
		\n <ul>
		\n <li> Final Horizon detected (in center of mass frame) at t = {}</li>
		\n <li> Final Horizon detected (at r=75M) at t = {}</li>
		\n <li> Maximum Amplitude in Psi4 (at r=75M) achieved at t = {}</li>
		\n </ul>	""".format(metadata['final_horizon'], metadata['final_horizon']+75, metadata['max_amp'])

	html_txt = html_txt + """

\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\">Trajectory Plots</a> </h2>
\n 
\n <p> <a href=\"figures/Trajectory_xvstime.png\"><img src=\"figures/Trajectory_xvstime.png\" alt=\"HTML/Trajectory_xvstime\" width=\"700\" height=\"600\" hspace=20></a>
\n  <a href=\"figures/Trajectory_yvstime.png\"><img src=\"figures/Trajectory_yvstime.png\" alt=\"Trajectory_yvstime\" width=\"700\" height=\"600\" ></a></p>
\n <p> <a href=\"figures/Trajectory_xy.png\"><img src=\"figures/Trajectory_xy.png\" alt=\"HTML/Trajectory_xy\" width=\"700\" height=\"600\" hspace=20></a>
\n  <a href=\"figures/Trajectory_separation.png\"><img src=\"figures/Trajectory_separation.png\" alt=\"Trajectory_separation\" width=\"700\" height=\"600\" ></a></p>
\n <p> <a href=\"figures/Trajectory_phase.png\"><img src=\"figures/Trajectory_phase.png\" alt=\"HTML/Trajectory_phase\" width=\"700\" height=\"600\" hspace=20></a>
\n  <a href=\"figures/Trajectory_rdot.png\"><img src=\"figures/Trajectory_rdot.png\" alt=\"Trajectory_rdot\" width=\"700\" height=\"600\" ></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Mass and Area Plots </a></h2>
\n <p> <a href=\"figures/IrreducibleMasses.png\"><img src=\"figures/IrreducibleMasses.png\" alt=\"IrreducibleMasses.png\" width=\"700\" height=\"600\" hspace=20></a>
\n  <a href=\"figures/Area.png\"><img src=\"figures/Area.png\" alt=\"Area.png\" width=\"700\" height=\"600\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Radius Plots </a></h2>
\n <p> <a href=\"figures/CoordRadius.png\"><img src=\"figures/CoordRadius.png\" alt=\"CoordRadius.png\" width=\"700\" height=\"600\" hspace=20></a>
\n  <a href=\"figures/ArealRadius.png\"><img src=\"figures/ArealRadius.png\" alt=\"ArealRadius.png\" width=\"700\" height=\"600\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Proper Distance </a></h2>
\n <p> <a href=\"figures/ProperDistance.png\"><img src=\"figures/ProperDistance.png\" alt=\"ProperDistance.png\" width=\"700\" height=\"600\" hspace=20></a>
\n  <a href=\"figures/Trumpets.png\"><img src=\"figures/Trumpets.png\" alt=\"Trumpets.png\" width=\"700\" height=\"600\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Runstats </h2>
\n <p> <a href=\"figures/Runstats.png\"><img src=\"figures/Runstats.png\" alt=\"Runstats.png\" width=\"700\" height=\"600\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Expansion Plots </h2>
\n <p> <a href=\"figures/InwardExpansion.png\"><img src=\"figures/InwardExpansion.png\" alt=\"InwardExpansion.png\" width=\"700\" height=\"600\" hspace=20></a>
\n  <a href=\"figures/OutwardExpansion.png\"><img src=\"figures/OutwardExpansion.png\" alt=\"OutwardExpansion.png\" width=\"700\" height=\"600\"></a></p>
\n 
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Spin Plots </a></h2>
\n <p> <a href=\"figures/Spinx.png\"><img src=\"figures/Spinx.png\" alt=\"Spin-x.png\" width=\"700\" height=\"600\" hspace=20></a>
\n  <a href=\"figures/Spiny.png\"><img src=\"figures/Spiny.png\" alt=\"Spin-y.png\" width=\"700\" height=\"600\"></a></p>
\n  <p> <a href=\"figures/Spinz.png\"><img src=\"figures/Spinz.png\" alt=\"Spin-z.png\" width=\"700\" height=\"600\"></a>
\n <a href=\"figures/Spinmag.png\"><img src=\"figures/Spinmag.png\" alt=\"Spin_Magnitude.png\" width=\"700\" height=\"600\" hspace=20></a></p>
\n 
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Links to Files </h2>
\n <ul>
\n <li><a href=\"data/{}\">Parameter File</a> </li>
\n <li><a href=\"data/Separation.txt\">Separation-OrbitalPhase File</a> </li>
\n </ul>
\n </body>
\n </html>
\n 
	""".format(os.path.basename(parfile))
	return html_txt

def webpage(wfdir, outdir,locate_merger):
	
	datadir = DataDir(wfdir, outdir) 
	filename = open(os.path.join(datadir+'/..','webpage.html'),'w+')

	meta_data, parfile = metadata(wfdir, outdir, locate_merger=False)
	webdata = webpage_data(meta_data, parfile, locate_merger=False)
	filename.write(webdata)
	filename.close()
	#if not os.path.exists(datadir+"/../HTML"):		
	#	copytree('HTML', datadir+'/../HTML')
	
