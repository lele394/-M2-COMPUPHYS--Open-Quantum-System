import matplotlib.pyplot as plt
import numpy as np



#amp phase amp phase .......

def GetData(file_path):
    file = open(file_path).read()
    data = file.splitlines()[4:]
    data = [[float(x) for x in i.split()] for i in data]
    return data

def ExtractValues(data):
    # Create amp and phase 
    amp = [[line[i] for i in range(len(line)) if i%2 == 0] for line in data]
    phase = [[line[i] for i in range(len(line)) if (i+1)%2 == 0] for line in data]
    return {"amp": amp, "phase": phase}


def ShowMatrix(mat):
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()



def Extract(path):    
    data = GetData(path)
    return ExtractValues(data)

#Get data for all components
def GetComplexFields(monitor):
    ex = Extract(f'./data/fwtmp_{monitor}_f1_ex.dat')
    ey = Extract(f'./data/fwtmp_{monitor}_f1_ey.dat')
    ez = Extract(f'./data/fwtmp_{monitor}_f1_ez.dat')

    hx = Extract(f'./data/fwtmp_{monitor}_f1_hx.dat')
    hy = Extract(f'./data/fwtmp_{monitor}_f1_hy.dat')
    hz = Extract(f'./data/fwtmp_{monitor}_f1_hz.dat')

    E = np.zeros((len(ex["amp"]), len(ex["amp"][0]),3), dtype=np.complex128)
    H = np.zeros((len(ex["amp"]), len(ex["amp"][0]),3), dtype=np.complex128)


    # Z0 = np.sqrt(4*np.pi*1e-7 / (1/(36*np.pi*1e-7)))*1e-6

    for x in range(len(ex["amp"])):
        for y in range(len(ex["amp"][0])):
            E[x, y] = np.array([ex["amp"][x][y] * np.exp(1j*(ex["phase"][x][y]/180 *np.pi)),
                                ey["amp"][x][y] * np.exp(1j*(ey["phase"][x][y]/180 *np.pi)),
                                ez["amp"][x][y] * np.exp(1j*(ez["phase"][x][y]/180 *np.pi))])
            
            H[x, y] = np.array([hx["amp"][x][y] * np.exp(1j*(hx["phase"][x][y]/180 *np.pi)),
                                hy["amp"][x][y] * np.exp(1j*(hy["phase"][x][y]/180 *np.pi)),
                                hz["amp"][x][y] * np.exp(1j*(hz["phase"][x][y]/180 *np.pi))])
            
            # print(E[x, y])
            # quit()
            

    # Ds = 0.02 mu
    return E,H




              #X   Y   Z

def DoMonitor(monitor, ds):
    E,H = GetComplexFields(monitor)

    # Calculate Z0
    Z0 = np.sqrt(4 * np.pi * 1e-7 / (1 / (36 * np.pi * 1e-7)))
    
    # Calculate the Poynting vector (time-averaged) using complex conjugates
    PoyntingIG = 0.5 * np.cross(E, np.conj(H)).real
    
    # Calculate the dot product of Poynting vector with the surface normal vector (ds)
    # to account for the flux through that surface
    scalar_products = np.einsum('ijk,k->ij', PoyntingIG, ds)
    
    # Sum up all the contributions for this monitor/surface
    sum_flux = np.sum(scalar_products)
    
    # Compute the power passing through this surface
    power = sum_flux * np.linalg.norm(ds) / Z0 * 1e12
    
    return power



monitors = ["X-", "X+",
           "Y-", "Y+",
           "Z-", "Z+"]

dss = np.array([[ -0.02e-12,  0,     0],   # X-
                [  0.02e-12,  0,     0],   # X+
                [  0,    -0.02e-12,  0],   # Y-
                [  0,     0.02e-12,  0],   # Y+
                [  0,     0,    -0.02e-12], # Z-
                [  0,     0,     0.02e-12]]) # Z+


values = [DoMonitor(monitors[i], dss[i]) for i in range(len(monitors))]
print(values)
print(np.sum(values))

        





