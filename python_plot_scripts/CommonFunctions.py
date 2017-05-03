from inspect import getframeinfo, stack

def debuginfo(message):
    caller = getframeinfo(stack()[1][0])
    print "Warning: %s:%d - %s" % (caller.filename, caller.lineno, message)

