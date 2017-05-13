from inspect import getframeinfo, stack
import os
def debuginfo(message):

	caller = getframeinfo(stack()[1][0])
	print "Warning: %s:%d - %s" % (caller.filename, caller.lineno, message)

def DataDir(dirpath, outdir):

	filename = dirpath.split("/")[-1]
  	outputdir = os.path.join(outdir, filename)
	datadir = os.path.join(outputdir,'data')

	if not os.path.exists(datadir):
		os.makedirs(datadir)
	return datadir


def FigDir(dirpath, outdir):

	filename = dirpath.split("/")[-1]
  	outputdir = os.path.join(outdir, filename)
	figdir = os.path.join(outputdir,'figures')

	if not os.path.exists(figdir):
		os.makedirs(figdir)
	return figdir


