#Incomplete

def webpage_data(metadata):
	
	
	html_txt = "

<!DOCTYPE html> \n
<html lang=\"en-US\"> \n
<head> \n
<style> \n
/* body { \n
  background-image: url(http://wallpapercave.com/wp/zPuoYm9.jpg); \n
  width: 1000%; \n
  }-->*/\n
h1 {color:rgb(0, 132, 255);margin: 50px;} \n
h2 {color:rgb(110, 145, 236);margin: 50px;} \n
p {color:ref;margin: 50px;}\n 
ul {margin: 50px}\n 
</style>\n 
<title> Simulation </title> \n
</head>\n
<body>\n 
\n
<h1 style=\"text-align:center; font-size:300% \n \"> %s </h1>" %filename


	html_txt = html_txt + "
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> <a href=\"Metadata.html\">Metadata </a></h2>
\n <ul>
\n <li> Simulation Name = {}</li>
\n <li> Sim-Type = {}</li> 
\n <li> Mass Ratio = {}</li>
\n <li> Initial Separation = {}</li>
\n <li> Spin of BH1 = {}</li>
\n <li> Spin of BH2 = {}</li>
		".format(metadata['Simulation Name'], metadata['Sim-type'], metadata['Mass-Ratio'], metadata['Initial Separation'], metadata['Spin1'], metadata['Spin2'])



	html_txt = html_txt + "

\n <h2><a href=\"Trajectory.html\">Trajectory Plots</a> </h2>
\n 
\n <p> <a href=\"figures/Trajectory_xy.png\"><img src=\"figures/Trajectory_xy.png\" alt=\"Trajectory_xy\" width=\"500\" height=\"400\" hspace=20></a>
\n  <a href=\"figures/Trajectory_separation.png\"><img src=\"figures/Trajectory_separation.png\" alt=\"Trajectory_separation\" width=\"500\" height=\"400\" ></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"><a href=\"Psi4.html\"> Psi4 Plots </a></h2>
\n <p> <a href=\"figures/Psi4_amp.png\"><img src=\"figures/Psi4_amp.png\" alt=\"Psi4_amp.png\" width=\"500\" height=\"400\" hspace=20></a>
\n  <a href=\"figures/Psi4_phase.png\"><img src=\"figures/Psi4_phase.png\" alt=\"Psi4_phase.png\" width=\"500\" height=\"400\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"><a href=\"Momentum.html\"> Momentum Plots </a></h2>
\n <p> <a href=\"figures/Momentum_components.png\"><img src=\"figures/Momentum_components.png\" alt=\"Momentum_components.png\" width=\"500\" height=\"400\" hspace=20></a>
\n  <a href=\"figures/Pmag.png\"><img src=\"figures/Pmag.png\" alt=\"Pmag.png\" width=\"500\" height=\"400\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"><a href=\"Energy.html\"> Angular Momentum and Energy Plots </a></h2>
\n <p> <a href=\"figures/Energy.png\"><img src=\"figures/Energy.png\" alt=\"Energy.png\" width=\"500\" height=\"400\" hspace=20></a>
\n  <a href=\"figures/Energy_derivative.png\"><img src=\"figures/Energy_derivative.png\" alt=\"Energy_derivative.png\" width=\"500\" height=\"400\"></a></p>
\n <p> <a href=\"figures/AngMom.png\"><img src=\"figures/AngMom.png\" alt=\"Angular_Momentum.png\" width=\"500\" height=\"400\" hspace=20></a>
\n  <a href=\"figures/AngMomDer.png\"><img src=\"figures/AngMomDer.png\" alt=\"Angular_Momentum_Derivative.png\" width=\"500\" height=\"400\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Runstats </h2>
\n <p> <a href=\"figures/Runstats.png\"><img src=\"figures/Runstats.png\" alt=\"Runstats.png\" width=\"500\" height=\"400\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"><a href=\"Spin.html\"> Spin Plots </a></h2>
\n <p> <a href=\"figures/Spinmag.png\"><img src=\"figures/Spinmag.png\" alt=\"Spin_Magnitude.png\" width=\"500\" height=\"400\" hspace=20></a>
\n  <a href=\"figures/Spinz.png\"><img src=\"figures/Spinz.png\" alt=\"Spin-z.png\" width=\"500\" height=\"400\"></a></p>
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Kicks Plots </h2>
\n 
\n 
\n <h2 style=\"text-align:left;font-size:150%;font-family:verdana\"> Links to Files </h2>
\n <ul>
\n <li><a href=\"data/D09_q1.00_a0.35_Q20.par\">Parameter File</a> </li>
\n <li><a href=\"data/Separation.txt\">Separation-OrbitalPhase File</a> </li>
\n <li><a href=\"data/Psi4_l2m2_r75.txt\">Psi4 File</a> </li>
\n <li><a href=\"data/psi4analysis_r75.00.asc\">Psi4_Analysis File</a> </li>
\n </ul>
\n </body>
\n </html>
\n 
	"""
	return html_txt

def webpage(wfdir, outdir):
	
	datadir = DataDir(wfdir, outdir) 
	filename = open('webpage.html','a+')

	metadata = {}
	metadata = metadata(wfdir, datadir)
	webdata = webpage_data(metadata
	filename.write(webdata)
	filename.close()

	
