.. image:: https://github.com/drlukeparry/pyocl/workflows/Python%20application/badge.svg
  :target: https://github.com/drlukeparry/pyocl/actions

PyOCL Module Repository
========================

Provides convenience methods and boiler-plate code for interacting and setting up the OpenCL compute environment based on the capabilities provided
by PyOpenCL. This is to aid setting up simulations for use in development. Common routines and flags are enabled for convenience to improve productivity and provide a consistent
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
compiled with the GlSharing is available via `y Christoph Gohlke repo <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl>`_.
`Learn more <http://lukeparry.uk/>`_.
