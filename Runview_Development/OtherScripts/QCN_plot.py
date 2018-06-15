import numpy as np
import matplotlib.pyplot as plt

qcn_spmcs = ("BNS_SPMCS/QCN.txt")
rs, qs, cs, ns = np.loadtxt(qcn_spmcs, comments ="#", unpack = True, usecols = (0, 4,5,6))


qcn_bowen = ("BNS_Bowen/tov_internaldata_0.asc")
rb, qb, cb, nb = np.loadtxt(qcn_bowen, comments ="#", unpack = True, usecols = (0, 4,5,6))

plt.plot(rs, cs, 'k', label = "BK", linewidth = 4)
plt.plot(rb, cb, 'g', label = "MC", linewidth = 2)
plt.ylabel("Q")
plt.xlabel("r")
plt.legend()
plt.show()
