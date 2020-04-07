# -*- coding: utf-8 -*-
import os
from enum import Enum, auto
from typing import List
import logging
import pyopencl as cl



class OpenCLFlags(Enum):
    """
    Enums for setting options for the OpenCL Class
    """
    ENABLE_DEBUG = auto()
    ENABLE_COMPILER_OUTPUT = auto()
    ENABLE_CACHE = auto()  # 'PYOPENCL_NO_CACHE=1'
    DISABLE_NON_FINITE_MATH = auto()  # '-cl-finite-math-only
    DISABLE_OPTIMISATIONS = auto()  # '-cl-opt-disable'

class Core:
    """
    PyOCL Core Class

    Provides methods for interacting and setting up the OpenCL compute environment based on the capabilities provided
    by PyOpenCL. Common routines and flags are enabled for convenience to improve productivity and provide a consistent
    environment to develop in.
    """

    def __init__(self, useGPU: bool = True) -> None:
        self._isUsingGPU = useGPU
        self._gl_interop = False
        self._clDevice = None

        # Show compiled output by setting envrionment flag
        self.enableCompilerOutput(OpenCLFlags.ENABLE_COMPILER_OUTPUT)
        self.enableCompilerCache(OpenCLFlags.ENABLE_CACHE)

        os.environ["AMPLXE_LOG_LEVEL"] = "TRACE"
        os.environ["TPSS_DEBUG"] = "1"

        # Get the OpenCL Platforms
        self._platform = cl.get_platforms()[0]

        logging.debug('Initialising OpenCL Runtime - {:s}'.format(self.platform.name))

        gpuDevices = self.gpuDevices  # get GPU devices of selected platform
        cpuDevices = self.cpuDevices  # get GPU devices of selected platform

        if len(gpuDevices) > 0 and useGPU:
            self._device = gpuDevices[0]  # take first GPU
            logging.debug('Using GPU - {:s} ({:s}) for OpenCL'.format(self.device.name, self.device.version))

        elif len(cpuDevices) > 0:
            self._device = cpuDevices[0]  # take first CPU
            logging.debug('Using CPU for OpenCL')
        else:
            raise RuntimeError('No OpenCL device currently available')

        if useGPU and cl.have_gl() and self.hasGLShare():
            from pyopencl.tools import get_gl_sharing_context_properties

            try:
                self._context = cl.Context(
                        properties=[(cl.context_properties.PLATFORM,
                                     self.platform)] + get_gl_sharing_context_properties())
                self._gl_interop = True
            except:
                logging.warning('Issue with GL Sharing at runtime')
                self._context = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)],
                                           devices=[self.device])

        else:
            self._context = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)],
                                       devices=[self.device])

    def enableCompilerCache(self, state : int) -> None:
        """
        :param state: provide a OpenCLFlag
        """
        os.environ["PYOPENCL_NO_CACHE"] = '1' if (state == OpenCLFlags.ENABLE_CACHE) else '1'

    def enableCompilerOutput(self, state: int) -> None:
        """
        :param state: provide a OpenCLFlag
        """
        os.environ["PYOPENCL_COMPILER_OUTPUT"] = '' if (state == OpenCLFlags. ENABLE_COMPILER_OUTPUT) else '1'

    @property
    def device(self) -> cl.Device:
        """
        Returns the current selected OpenCL Device
        """
        return self._device

    @property
    def platform(self) -> cl.Platform:
        """
        Returns the current OpenCL Device

        :rtype: cl.Platform
        """
        return self._platform

    @property
    def context(self) -> cl.Context:
        """
        Returns the chosen generated OpenCL context

        :return: cl.Context:
        """
        return self._context

    def useGLInterop(self) -> bool:
        return self._gl_interop

    def isUsingGPU(self) -> bool:
        return self._isUsingGPU

    def hasSharedMemory(self) -> bool:
        """
        Returns if the currently selected Compute Device has shared memory capability (e.g. APU, Intel Platforms)

        :return: bool
        """
        if self.deviceType() == 'Intel GPU':
            return True
        else:
            return False

    def deviceType(self) -> str:
        if "Intel(R) UHD Graphics" in self.device.name:
            return 'Intel GPU'

    ### Helper functions ###
    @property
    def gpuDevices(self) -> List[cl.Device]:

        """
        Returns list of GPU devices based on the currently chosen platform

        :return: list of clDevices
        """
        return self.platform.get_devices(cl.device_type.GPU)

    @property
    def cpuDevices(self) -> List[cl.Device]:

        """
        Returns list of CPU devices based on the currently chosen platform

        :return: list of clDevices
        """
        return self.platform.get_devices(cl.device_type.CPU)

    @property
    def computeUnits(self) -> int:
        """
        Returns the number of compute units available on the selected compute device

        :return: number of compute units available
        """
        return self.device.max_compute_units

    @property
    def localMemorySize(self) -> int:
        """
        Returns the local size of memory available on the compute device

        :return: Size of local memory available [bytes]
        """
        return self.device.local_mem_size

    @property
    def globalMemorySize(self) -> int:
        """
        Returns the global size of memory available on the compute device
        
        :return: Size of global memory available [bytes]
        """
        return self.device.global_mem_size

    def hasFloat16(self) -> bool:
        """
         Returns if the compute device has native float16 support

         :return: bool: Native float16 support available
         """
        return 'cl_khr_fp16' in self.platform.extensions

    def hasDouble(self) -> bool:
        """
         Returns if the compute device has native float64 (double) support

         :return: bool: Native float64 support available
         """
        return 'cl_khr_fp64' in self.platform.extensions

    def hasGLShare(self) -> bool:
        """
         Returns if the compute device has native GL Sharing Capabilities (within driver)

         :return: bool: Native GLInterop Sharing support available
         """
        return 'cl_khr_gl_sharing' in self.platform.extensions
