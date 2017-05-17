def webpage(metadata):
	html_txt = """
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <link href="style.css" rel="stylesheet" type="text/css" />
</head>
<body>
	
	<div id="main-content">
		<div class="box">
        	<h1>{}</h1>
        	<h2>Simulation Details </h2>
			<ul style="margin-top:10px;">
				<li>Simulation Type - BBH</li>
				<li>s1 = {}</li>
				<li>s2 = {}</li>
				<li>q = {}</li>
				<li>D = {}</li>
			</ul>
		</div>
		<h2> Trajectory Plots</h2>
		<p>
			<img  src="Trajectory_xy.png" width="600" height="500" style="margin: 0 10px 10px 0;float:left;" />
			<img src="Trajectory_xyvstime.png" width="500" height="500" style="margin: 0 10px 10px 0;float:center;" />
		</p>
		<p>
			<img style = "display:block;" src="Trajectory_separation.png" width="700" height="500" style="margin: 0 10px 10px 0;float:left;" />
		<br/>	
		</p>
		<p>
		<h2>Psi4 Plots</h2>
		</p>
		<p>
			<img src="Psi4_plot.png" width="600" height="500" style="margin: 0 10px 10px 0;float:left;" />
			<img src="Psi4_plot_zoomed.png" width="600" height="500" style="margin: 0 10px 10px 0;float:center;" />
		</p>
		<p>
			<img src="Psi4_amp.png" width="600" height="500" style="margin: 0 10px 10px 0;float:left;" />
			<img src="Psi4_phase.png" width="600" height="500" style="margin: 0 10px 10px 0;float:left;" /
		</p>
	</div>
<!--</div>	-->

</body>
</html>
	""".format(simname, simname, s1,s2,q,sep)
