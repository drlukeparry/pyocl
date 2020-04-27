from enum import Enum, auto
import abc
from typing import Any, List, Tuple
import logging
import pyopencl as cl

from .core import Core


class OpenCLSimBase(abc.ABC):
    """
    OpenCL Sim class for creating the runtime. Other classes should derive from this class and set both the kernel.
    The derived class should specify the kernel input as a string
    """

    def __init__(self):
        self.ocl = None
        self.queue = None
        self.program = None
        self.kernel = None
        self._workGroupSize = (64, 1)
        self._dims = 2  # dimension of problem

    def initialiseCL(self) -> None:
        """
        Create the OpenCL context, generates the compiled kernel.
        """

        self.ocl = Core()

        # Create a command queue
        self.queue = cl.CommandQueue(self.ocl.context, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Compile and build the openCL program
        buildOptions = []

        if self.ocl.isDebugBuild():
            buildOptions += ['-g']

        if self.ocl.isUsingOpenCL2():
            buildOptions += ['-cl-std=CL2.0']

        self.program = cl.Program(self.ocl.context, self.kernel).build(options=buildOptions)

    #        try:
    #            self.program = cl.Program(self.ocl.context, self.kernel).build()
    #        except:
    #            logging.error("Error:")
    #            print(self.program.get_build_info(self.ocl.device, cl.program_build_info.ERROR))
    #            raise


    @property
    def dimensions(self) -> int:
        """
        Return the dimensions of the kernel launched

        :return: The kernel dimensions
        """
        return self._dims


    @dimensions.setter
    def dimensions(self, dims: int):
        """
        Return the dimensions of the kernel launched

        :param dims: Returns the dimensions for the kernel
        """
        self._dims = dims

    def isKernelAvailable(self) -> bool:
        return self.program and len(self.program.all_kernels())

    @property
    @abc.abstractmethod
    def kernel(self):
        raise NotImplementedError()

    @property
    def workGroupSize(self) -> Tuple[int, ...]:
        """
        The chosen workgroup size for the kernel launch

        :return: The workgroup size
        """
        return self._workGroupSize


    @workGroupSize.setter
    def workGroupSize(self, wgSize: Any):
        """
        Sets the work group size to be used by the kernel.

        :param wgSize: The work group size
        """
        if isinstance(wgSize,int):
            wgSize = tuple(wgSize)

        self._workGroupSize = wgSize


    def getLocalMemorySize(self) -> int:
        """
        Returns the calculated local memory size based oen the compiled kernel.
         A return of -1 indicates the kernel is not available.

        :return: Calculated memory size in [bytes]
        """
        if not self.isKernelAvailable():
            return -1

        return self.program.all_kernels()[0].get_work_group_info(cl.kernel_work_group_info.LOCAL_MEM_SIZE,
                                                                 self.ocl.device)


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
                self.ocl.device)


    def getMaximumWorkGroupSize(self) -> int:
        """
        Returns the maximum workgroup size for the device.
        A return of -1 indicates the kernel is not available.

        :return: Returns the maximum work group size
        """
        if not self.isKernelAvailable():
            return -1

        return self.program.all_kernels()[0].get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE,
                                                                 self.ocl.device)
