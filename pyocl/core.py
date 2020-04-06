# -*- coding: utf-8 -*-


from enum import Enum, auto
from typing import List
import logging
import numpy
import time
import pyopencl as cl

from . import helpers


def get_hmm():
    """Get a thought."""
    return 'hmmm...'


def hmm():
    """Contemplation..."""
    if helpers.get_answer():
        print(get_hmm())


class Core:
    """

    """

    class OpenCLFlags(Enum):
        """
        Enums for setting options for the OpenCL Class
        """
        ENABLE_DEBUG = auto()
        DISABLE_CACHE = auto()  # 'PYOPENCL_NO_CACHE=1'
        DISABLE_NON_FINITE_MATH = auto()  # '-cl-finite-math-only, -cl-finite-math-only'

    def __init__(self, useGPU: bool = True) -> None:
        self._isUsingGPU = useGPU
        self._gl_interop = False
        self._clDevice = None

        # Show compiled output by setting envrionment flag
        import os

        os.environ["PYOPENCL_COMPILER_OUTPUT"] = '1'
        os.environ["PYOPENCL_NO_CACHE"] = "1"
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
                    properties = [(cl.context_properties.PLATFORM, self.platform)] + get_gl_sharing_context_properties())
                self._gl_interop = True
            except:
                logging.warning('Issue with GL Sharing at runtime')
                self._context = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)],
                                           devices=[self.device])

        else:
            self._context = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)],
                                       devices=[self.device])

    @property
    def device(self):
        return self._device

    @property
    def platform(self) -> cl.Platform:
        return self._platform

    @property
    def context(self):
        return self._context

    def useGLInterop(self) -> bool:
        return self._gl_interop

    def isUsingGPU(self) -> bool:
        return self._isUsingGPU

    def hasSharedMemory(self) -> bool:
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
        return self.platform.get_devices(cl.device_type.GPU)

    @property
    def cpuDevices(self) -> List[cl.Device]:
        return self.platform.get_devices(cl.device_type.CPU)

    @property
    def computeUnits(self) -> int:
        return self.device.max_compute_units

    @property
    def localMemorySize(self) -> int:
        return self.device.local_mem_size

    @property
    def globalMemorySize(self) -> int:
        return self.device.global_mem_size

    def hasFloat16(self) -> bool:
        return 'cl_khr_fp16' in self.platform.extensions

    def hasDouble(self) -> bool:
        return 'cl_khr_fp64' in self.platform.extensions

    def hasGLShare(self) -> bool:
        return 'cl_khr_gl_sharing' in self.platform.extensions
