# -*- coding: utf-8 -*-
import os
from enum import Enum, auto
from typing import List, Tuple
import logging
import pyopencl as cl


class OpenCLFlags(Enum):
    """
    Enums for setting options for the OpenCL Class
    """
    ENABLE_DEBUG = auto()
    ENABLE_COMPILER_OUTPUT = auto()
    ENABLE_CACHE = auto()  # 'PYOPENCL_NO_CACHE=1'
    DISABLE_NON_FINITE_MATH = asuto()  # '-cl-finite-math-only
    DISABLE_OPTIMISATIONS = auto()  # '-cl-opt-disable'


class Core:
    """
    PyOCL Core Class

    Provides methods for interacting and setting up the OpenCL compute environment based on the capabilities provided
    by PyOpenCL. Common routines and flags are enabled for convenience to improve productivity and provide a consistent
    environment to develop in.
    """

    def __init__(self, device = None, useGPU: bool = True) -> None:

        self._isUsingGPU = useGPU
        self._gl_interop = False
        self._clDevice = None

        # Show compiled output by setting envrionment flag
        self.enableCompilerOutput(OpenCLFlags.ENABLE_COMPILER_OUTPUT)
        self.enableCompilerCache(OpenCLFlags.ENABLE_CACHE)

        os.environ["AMPLXE_LOG_LEVEL"] = "TRACE"
        os.environ["TPSS_DEBUG"] = "1"

        if device:
            self._device = device
            self._platform = device.platform

        else:
            # Preselect the default platform and device

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

        if self.isUsingGpu() and cl.have_gl() and self.hasGLShareExtension():
            from pyopencl.tools import get_gl_sharing_context_properties

            # Try setting up an OpenCL context with a Gl_Sharing
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

    @staticmethod
    def enableCompilerCache(state : int) -> None:
        """
        Sets PyOpenCL to use compiler caching to improve warm-up time

        :param state: provide a OpenCLFlag
        """
        os.environ["PYOPENCL_NO_CACHE"] = '1' if (state == OpenCLFlags.ENABLE_CACHE) else '1'

    @staticmethod
    def enableCompilerOutput(state: int) -> None:
        """
        Sets PyOpenCL to enable the compiler output including errors

        :param state: provide a OpenCLFlag
        """
        os.environ["PYOPENCL_COMPILER_OUTPUT"] = '' if (state == OpenCLFlags. ENABLE_COMPILER_OUTPUT) else '1'

    def isUsingGPU(self) -> bool:
        """
        Returns if the device is a GPU
        """
        return self.device.type == cl.device_type.GPU

    def isUsingCPU(self) -> bool:
        """
        Returns if the device is a GPU
        """
        return self.device.type == cl.device_type.CPU

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

        :return: The OpenCL Platform currently selected
        """
        return self._platform

    @property
    def context(self) -> cl.Context:
        """
        Returns the chosen generated OpenCL context

        :return: The OpenCL Context
        """
        return self._context

    def useGLInterop(self) -> bool:
        return self._gl_interop

    def hasSharedMemory(self) -> bool:
        """
        Returns if the currently selected Compute Device has shared memory capability (e.g. APU, Intel Platforms)

        :return:
        """
        if self.deviceType() == 'Intel GPU':
            return True
        else:
            return False

    def deviceType(self) -> str:
        if "Intel" in self.device.vendor and self.device.type == cl.device_type.GPU:
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
    def maxWorkGroupSize(self) -> int:
        """
        The maximum work group size for the device
        """
        return self.device.max_work_group_size()

    @property
    def max2DImageSize(self) -> Tuple[int,int]:
        """
        The maximum 2D image size buffer
        """
        return self.device.image2d_max_width,  self.device.image2d_max_height

    @property
    def max3DImageSize(self) -> Tuple[int,int, int]:
        """
        The maximum 3D image size buffer
        """
        return self.device.image3d_max_width, self.device.image3d_max_height, self.device.image3d_max_depth

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

         :return: Native float16 support available
         """
        return 'cl_khr_fp16' in self.platform.extensions

    def hasDouble(self) -> bool:
        """
         Returns if the compute device has native float64 (double) support

         :return: Native float64 support available
         """
        return 'cl_khr_fp64' in self.platform.extensions

    def hasGLShareExtension(self) -> bool:
        """
         Returns if the compute device has native GL Sharing Capabilities (within driver)

         :return: Native GLInterop Sharing support available
         """
        return 'cl_khr_gl_sharing' in self.platform.extensions
