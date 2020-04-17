# -*- coding: utf-8 -*-
"""
@author: Luke Parry
"""

import time
import matplotlib.pyplot as plt
import numpy as np

import pyopencl as cl
import pyocl

# note intel gpu (8 exec units x3 slices = 24 WG * 7 threads)

class TransientHeatSim(pyocl.OpenCLSimBase):
    """
    Class that holds data for the heat equation using OpenCL
    """
    def __init__(self, u0):
        # Make sure that the data is single precision floating point
        assert (np.issubdtype(u0.dtype, np.float32))

        self._u0 = u0  # raw data
        self._u1 = None

        # Simulation Parameters
        self.rho = 0
        self.cp = 0
        self.k = 0

        self.dt = 0.005
        self.t = 0
        self.dx = 1e-3
        self.dy = 1e-3

        self.initialiseCL()
        self.initialiseData(u0)

        # Find number of cells
        self.nx = u0.shape[0]
        self.ny = u0.shape[1]

    def initialiseData(self, u0):
        """
        Uploads initial data to the CL device
        """

        # PyOpenCL flags for transferring data
        mf = cl.mem_flags

        """
        Good  description of flags
        ##### Memory access in Kernel ####
        mf.READ_ONLY  - kernel read only acess
        mf.WRITE_ONLY - kernel write only access
        mf.READ_WRITE - kernel RW access

        ### Memory Transfer / Creations Options #####
        mf.COPY_HOST_PTR - Allocate memory on device and initialise - not efficient transfer  The memory object will set the memory region specified by the host pointer.
        mf.USE_HOST_PTR - memory not easily availble (fast transfers) - maintains a reference to that memory area and t might access it directly  during kernel invocation. Used for shared memory devices -Intel, Mali -
        mf.ALLOC_HOST_PTR. Allocate memory at host, accessible from host. Used for shared memory devices -Intel, Mali -
        """

        transferFlag = mf.USE_HOST_PTR if self.ocl.hasSharedMemory() else mf.COPY_HOST_PTR

        # Upload data to the device

        self.u0 = cl.Buffer(self.ocl.context, mf.READ_WRITE | transferFlag, hostbuf=u0)

        # Allocate output buffers
        self.u1 = cl.Buffer(self.ocl.context, mf.READ_WRITE, u0.nbytes)
        # Map the host memory to  the buffer object

    @property
    def kernel(self):
        from mako.template import Template
        with open('./heat_eq_2D.cl') as f:
            code = str(Template(f.read()).render())

        return code

    @property
    def alpha(self) -> np.float32:
        return np.float32(self.k / (self.rho * self.cp))

    def maxTimestep(self) -> np.float32:
        # returns the max timestep to satisfy the CFL condition
        return 0.5 * min(self.dx * self.dx / (2.0 * self.alpha),
                         self.dy * self.dy / (2.0 * self.alpha))

    def step(self):
        # Execute program on device

        #        ev = self.program.copy(self.queue, (self.nx, self.ny), self.workGroupSize,
        #                             self.u1, self.u0)

        #        ev = self.program.heat_eq_2D_shared(self.queue, (self.nx, self.ny), self.workGroupSize,
        #                                     self.u1, self.u0, cl.LocalMemory(4*18*18),
        #                                     np.float32(self.alpha), np.float32(self.dt), np.float32(self.dx), np.float32(self.dy))

        #        ev = self.program.heat_eq_2D_shared4(self.queue, (self.nx, self.ny), self.workGroupSize,
        #                                     self.u1, self.u0, cl.LocalMemory(4*18*24),
        #                                     np.float32(self.alpha), np.float32(self.dt), np.float32(self.dx), np.float32(self.dy))

        ev = self.program.heat_eq_2D(self.queue, (self.nx, self.ny), self.workGroupSize,
                                     self.u1, self.u0,
                                     np.float32(self.alpha), np.float32(self.dt), np.float32(self.dx),
                                     np.float32(self.dy))

        ev.wait()  # wait for kernel to finish
        # print('Time taken {:.5f}'.format((ev.profile.end - ev.profile.start)*1e-9))
        # Swap the buffers ( this involves pointers so no copying is actually performed!)
        self.u0, self.u1 = self.u1, self.u0

    def download(self):
        """
        Enables downloading data from CL device to Python
        """

        # Allocate data on the host for result - this does not exist by default
        u1 = np.empty((self.nx, self.ny), dtype=np.float32)

        #  Transfers a copy from the device buffer (u0) to host array (self.u0)
        # is_blocking is by default true on transfers between the host
        cl.enqueue_copy(self.queue, u1, self.u1, is_blocking=True)

        # Return
        return u1


"""
OpenCL Transient Heat Transfer Kernel
Computes the heat equation using an explicit finite difference scheme with OpenCL
"""

# Note that the dimensions for this kernel should be a multiple of the work group size
nx = 12800
ny = 12800

# Initialise a random numpy array on the host device  
u0 = np.random.rand(ny, nx).astype(np.float32) * 1000

# Create the OpenCL Transient Heat Kernel
heatsim = TransientHeatSim(u0)
heatsim.k = 10.0  # W/kgK
heatsim.rho = 2700.0  # kg/m^3
heatsim.cp = 920.0  # J/kgK

heatsim.dx = heatsim.dy = 1e-3  # [m]
heatsim.dt = heatsim.maxTimestep()

# Set the work group size for the kernel
heatsim.workGroupSize = (16, 16)

avgIncTime = []
for i in range(1, 10):
    timesteps_per_plot = 2
    # Simulate 10 timesteps

    startTime = time.time()
    for j in range(0, timesteps_per_plot):
        # Perform a kernal launch. Note these are synchronously launched.
        heatsim.step()

    endTime = float((time.time() - startTime)) / float(timesteps_per_plot)

    avgIncTime += [endTime]

# Iterations finished. Now visualise the result
# Note that on shared memory devices (Intel) The result can be used immediately

if heatsim.ocl.hasSharedMemory():
    plt.imshow(u0)
else:
    # Download data if using a non-shared memory device e.g. GPU
    u1 = heatsim.download()
    plt.imshow(u1)

print('Average iteration time {:.5f} ± {:.3f} ms '.format(np.mean(avgIncTime) * 1e3, np.std(avgIncTime) * 1e3))
print('Average iteration time {:.5f} MPix '.format(nx * ny / np.mean(avgIncTime) / 1e6))
