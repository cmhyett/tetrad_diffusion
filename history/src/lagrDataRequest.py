
import numpy as np
import pyJHTDB
import time
from pyJHTDB.dbinfo import isotropic1024coarse as info
import matplotlib.pyplot as plt
import threading
from threading import Thread
from pyJHTDB import libJHTDB

def getJHTDBParticles(radius=np.pi/512, L=np.pi/128, numTsteps=5, numSamples=100):
    interp_info = (44, 'FD4Lag4');
    dt = info['time'][1] - info['time'][0];
    deltaT = numTsteps*dt;

    start = time.time();
    t = 0.0;

    x = np.random.uniform(low=-radius, high=radius, size=(numSamples,3)).astype(np.float32);
        
    positions, times = lJHTDB.getPosition(starttime=t,
                                          endtime=t+deltaT,
                                          dt = dt,
                                          point_coords = x,
                                          steps_to_keep = numTsteps)

    vgt = np.zeros((numTsteps+1, numSamples, 9))
    ph = np.zeros((numTsteps+1, numSamples, 6))
    for j in range(0,numTsteps+1):
        vgt[j,:,:] = lJHTDB.getData(times[j],
                                    positions[j,:,:],
                                    sinterp = interp_info[0],
                                    tinterp = 0,
                                    data_set = info['name'],
                                    getFunction = 'getVelocityGradient');
        print("got vgt at time {}\n".format(times[j]));

        ph[j,:,:] = lJHTDB.getData(times[j],
                                   positions[j,:,:],
                                   sinterp = interp_info[0],
                                   tinterp = 0,
                                   data_set = info['name'],
                                   getFunction = 'getPressureHessian');
        print("got ph at time {}\n".format(times[j]));
    end = time.time();
    executionTime = end-start;
    print(executionTime);
    return positions, vgt, ph, times;


lJHTDB = libJHTDB()
lJHTDB.initialize()
auth_token=open("auth_token.txt").read().strip()
lJHTDB.add_token(auth_token)

basepath = '/xdisk/chertkov/cmhyett/jhtdbIsotropic1024/lagrData/'
numTsteps = 1000;
numSamples = 10000;

for i in range(0,10):
    positions, vgt, ph, times = getJHTDBParticles(numTsteps=numTsteps, numSamples=numSamples);
    np.save(basepath + 'positions{}.npy'.format(i), positions);
    np.save(basepath + 'vgt{}.npy'.format(i), vgt);
    np.save(basepath + 'ph{}.npy'.format(i), ph);
    np.save(basepath + 'times{}.npy'.format(i), times);
    
lJHTDB.finalize()


