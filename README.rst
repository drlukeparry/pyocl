.. image:: https://github.com/drlukeparry/pyocl/workflows/Python%20application/badge.svg
  :target: https://github.com/drlukeparry/pyocl/actions

.. image:: https://readthedocs.org/projects/pyocl/badge/?version=latest
    :target: https://pyocl.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

PyOCL Module Repository
========================

Provides convenience methods and boiler-plate code for interacting and setting up the OpenCL compute environment based on the capabilities provided
by `PyOpenCL <https://documen.tician.de/pyopencl/>`_. This is to aid setting up simulations for use in development. Common routines and flags are enabled for convenience to improve productivity and provide a consistent
environment to develop in.

Installation
*************

Installation is currently supported on Windows and Linux. PyOCL can be installed along with dependencies using

.. code:: bash

    pip install pyocl


Depending on your environment, you will need to install the latest version of PyOpenCL with support dependencies. This can be done through
the Anaconda distribution,

.. code:: bash

    conda install -c conda-forge pyopencl


For Windows platforms, PyOpenCL generally do not have the GlInterop capabilties compiled in. Binary python package for PyOpenCL that has been
compiled with the GlSharing is available via `Christoph Gohlke's repository <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl>`_.


USAGE
******

The basic usage for PyOCL is relatively straightforward once an OpenCL environment is available and pyopencl has been installed.
The project aims to compliment the pyopencl library to provide additional convenience methods and a concise framework for
prototyping and developing simulations using OpenCL. The following example demonstrates initiliasing a default GPU
device and accessing the basic components from pyopencl for creating and launching a kernel.

.. code:: python

    import pyocl

    ocl = pyocl.Core(useGpu = True)

    # Get the current selected platform
    platform = ocl.platform

    # Identify the GPU Devices available
    gpuDevices = ocl.gpuDevices

    # Number of compute units available on
    numComputeUnits = ocl.computeUnits

    # Get the current OpenCL device
    computeDevice = ocl.device

    # Get the OpenCL context for Initialising the Kernel and Command Queue
    context = ocl.context

The further usage can be built upon subclassing OpenCLSimBase, to provide the basic mechanism for launching and

.. code:: python

    import pyopencl as cl

    class Sim(pyocl.OpenCLSimBase):

        def __init__(self, u0):

            # Launch the Kernel Generation
            self.initialiseCL()

            self.initialiseData(u0)

            # Dimensions of the work item
            self.nx = u0.shape[0]
            self.ny = u0.shape[1]

        def initialiseData(self, u0):
            """  Uploads initial data to the CL device """

            # PyOpenCL flags for transferring data
            mf = cl.mem_flags

            # Copy from host memory to the device
            transferFlag = mf.COPY_HOST_PTR

            # Upload data to the device
            self.u0 = cl.Buffer(self.ocl.context, mf.READ_WRITE | transferFlag, hostbuf=u0)

        @property
        def kernel(self):
            # The kernel can be an inbuilt string or be from an external file which is formatted via the Mako Python Library

            from mako.template import Template
            with open('./kernelSource.cl') as f:
                code = str(Template(f.read()).render())

            return code

        def launch(self):
            # Execute program on device

            # Launch the kernel and put on the OpenCL queue for processing
            ev = self.program.myKernel(self.queue, (self.nx, self.ny), self.workGroupSize, self.u1, self.u0,)

            ev.wait()  # wait for kernel to finish (i.e. synchronous execution)


        def download(self):
            """ Enables downloading data from CL device to Python """

            # Allocate data on the host for result - this does not exist by default
            u1 = np.empty((self.nx, self.ny), dtype=np.float32)

            #  Transfers a copy from the device buffer (u0) to host array (self.u0)
            # is_blocking is by default true on transfers between the host
            cl.enqueue_copy(self.queue, u1, self.u1, is_blocking=True)

            # Return
            return u1


Further examples can be found in documented  `examples <https://github.com/drlukeparry/pyocl/tree/master/examples>`_  and also via
the project `documentation <https://pyocl.readthedocs.io/en/latest/>`_.

