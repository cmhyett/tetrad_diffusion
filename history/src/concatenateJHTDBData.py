# because of server-side constraints, it is common to request many small datasets
# this script concatenates them into a single dataset
import numpy as np;
import os;

basepath = "/xdisk/chertkov/cmhyett/jhtdbIsotropic4096/snapshot/"
outpath = basepath;
concatAxis = 0;

ph = np.load(basepath+"ph0.npy");
vgt = np.load(basepath+"vgt0.npy");
i = 1;

while (os.path.isfile(basepath+"ph{}.npy".format(i))):
    ph = np.concatenate((ph, np.load(basepath+"ph{}.npy".format(i))), axis=concatAxis);
    vgt = np.concatenate((vgt, np.load(basepath+"vgt{}.npy".format(i))), axis=concatAxis);
    i = i + 1;

np.save(basepath + "ph.npy", ph);
np.save(basepath + "vgt.npy", vgt);

