package cl

/*
#cgo CFLAGS: -DCL_TARGET_OPENCL_VERSION=300
#cgo windows LDFLAGS: -lOpenCL
#cgo darwin LDFLAGS: -framework OpenCL
#cgo linux pkg-config: OpenCL
#include <CL/cl.h>
#include <stdlib.h>
*/
import "C"

type (
	PlatformID   C.cl_platform_id
	DeviceID     C.cl_device_id
	Context      C.cl_context
	CommandQueue C.cl_command_queue
	Program      C.cl_program
	Kernel       C.cl_kernel
	MemObject    C.cl_mem
	Event        C.cl_event
	Bool         C.cl_bool
	UInt         C.cl_uint
	Int          C.cl_int
)

const (
	Success       = C.CL_SUCCESS
	DeviceTypeAll = C.CL_DEVICE_TYPE_ALL
	DeviceTypeCPU = C.CL_DEVICE_TYPE_CPU
	DeviceTypeGPU = C.CL_DEVICE_TYPE_GPU
	DeviceTypeACC = C.CL_DEVICE_TYPE_ACCELERATOR
)

func ErrorString(err Int) string {
	switch err {
	case C.CL_SUCCESS:
		return "Success"
	case C.CL_DEVICE_NOT_FOUND:
		return "Device not found"
	case C.CL_DEVICE_NOT_AVAILABLE:
		return "Device not available"
	case C.CL_OUT_OF_RESOURCES:
		return "Out of resources"
	case C.CL_OUT_OF_HOST_MEMORY:
		return "Out of host memory"
	case C.CL_INVALID_PLATFORM:
		return "Invalid platform"
	case C.CL_INVALID_DEVICE_TYPE:
		return "Invalid device type"
	case C.CL_INVALID_VALUE:
		return "Invalid value"
	default:
		return "Unknown error"
	}
}
