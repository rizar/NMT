
import sys
import cPickle as pkl
import numpy as np
from matplotlib import pyplot as plt

# Give path to log file
logfile = sys.argv[1]

# Change this if necessary
items = ['decoder_cost_cost_0', 'decoder_cost_cost_1', 'decoder_cost_cost_2']

try:
    d = {}
    log = pkl.load(open(logfile))
    for iter_, dc in log.iteritems():
        if dc.keys()[0] in items:
            idx = items.index(dc.keys()[0])
            if items[idx] not in d:
                d[items[idx]] = []
            d[items[idx]].append(dc.values()[0])

    for i, (key, val) in enumerate(d.iteritems()):
        plt.subplot(len(d), 1, i)
        plt.plot(np.asarray(xrange(len(val))), np.asarray(val))
        plt.xlabel('Iter')
        plt.ylabel('Cost')
        plt.title(key)

    plt.show()

except Exception as e:
    print str(e)
