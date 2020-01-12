import os

def sync(remotepath, wfdir )
    os.system('rsync -avtr --exclude="*check*" %s %s'%(remotepath, wfdir)
