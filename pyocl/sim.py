from enum import Enum, auto
from typing import List
import logging
import pyopencl as cl
from .core import Core


class OpenCLSimBase:
    """
    OpenCL Sim class for creating the runtime. Other classes should derive from this class and set both the kernel.
    The derived class should specify the kernel input as a string
    """

    def __init__(self):
        self.myocl = None
        self.queue = None
        self.program = None
        self.kernel = None
        self._isDebugBuild = False
        self._workGroupSize = (64, 1)
        self._dims = 2  # dimension of problem

    def initialiseCL(self) -> None:
        """
        Create the OpenCL context, generates the compiled kernel.
        """

        self.myocl = Core()

        # Create a command queue
        self.queue = cl.CommandQueue(self.myocl.context, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Compile and build the openCL program
        buildOptions = []

        if self.isDebugBuild:
            buildOptions += ['-g']  # -g

        self.program = cl.Program(self.myocl.context, self.kernel).build(options=buildOptions)


    #        try:
    #            self.program = cl.Program(self.myocl.context, self.kernel).build()
    #        except:
    #            logging.error("Error:")
    #            print(self.program.get_build_info(self.myocl.device, cl.program_build_info.ERROR))
    #            raise

def setDebugBuild(self, state: bool):
    """
    Sets the compilation and build of the OpenCL kernel to use debug flags
    :param state:
    """
    self._isDebugBuild = state


def isDebugBuild(self) -> bool:
    """
    Returns if the build includes debug flags

    :return: bool: build
    """
    return self._isDebugBuild


@property
def dimensions(self) -> int:
    """
    Return the dimensions of the kernel launched

    :return: int: The kernel dimensions
    """
    return self._dims


@dimensions.setter
def dimensions(self, dims: int):
    """
    Return the dimensions of the kernel launched

    :param dims: int: Returns the dimensions for the kernel
    """
    self._dims = dims


def isKernelAvailable(self):
    return self.program and len(self.program.all_kernels())


@property
def workGroupSize(self) -> int:
    """
    The chosen workgroup size for the kernel launch

    :return: tuple(int,int): The workgroup size
    """
    return self._workGroupSize


@workGroupSize.setter
def workGroupSize(self, wgSize: tuple()):
    """
    Sets the work group size to be used by the kernel.

    :param wgSize: tuple():  The work group size
    """
    self._workGroupSize = wgSize


def getLocalMemorySize(self) -> int:
    """
    Returns the calculated local memory size based oen the compiled kernel.
     A return of -1 indicates the kernel is not available.

    :return: int - Calculated memory size in [bytes]
    """
    if not self.isKernelAvailable():
        return -1

    return self.program.all_kernels()[0].get_work_group_info(cl.kernel_work_group_info.LOCAL_MEM_SIZE,
                                                             self.myocl.device)


def getRecommendedWorkGroupSizeMultiple(self) -> int:
    """
    Returns the recommended mulitple of workgroup size for the device.
    A return of -1 indicates the kernel is not available.

    :return: int - Returns the work groupsize recommendations
    """

    if not self.isKernelAvailable():
        return -1

    return self.program.all_kernels()[0].get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        self.myocl.device)


def getMaximumWorkGroupSize(self) -> int:
    """
    Returns the maximum workgroup size for the device.
    A return of -1 indicates the kernel is not available.

    :return: int - Returns the maximum work group size
    """
    if not self.isKernelAvailable():
        return -1

    return self.program.all_kernels()[0].get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE,
                                                             self.myocl.device)
