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

// 平台信息结构
type PlatformInfo struct {
	ID         PlatformID
	Name       string
	Vendor     string
	Version    string
	Profile    string
	Extensions string
}

// 设备信息结构
type DeviceInfo struct {
	ID           DeviceID
	Name         string
	Vendor       string
	Version      string
	Type         uint64
	MaxMemAlloc  uint64
	MaxWorkGroup Size
}

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
	Size         C.size_t
)

const (
	Success       = C.CL_SUCCESS
	DeviceTypeAll = C.CL_DEVICE_TYPE_ALL
	DeviceTypeCPU = C.CL_DEVICE_TYPE_CPU
	DeviceTypeGPU = C.CL_DEVICE_TYPE_GPU
	DeviceTypeACC = C.CL_DEVICE_TYPE_ACCELERATOR
)

// 平台信息类型
const (
	PlatformProfile    = C.CL_PLATFORM_PROFILE
	PlatformVersion    = C.CL_PLATFORM_VERSION
	PlatformName       = C.CL_PLATFORM_NAME
	PlatformVendor     = C.CL_PLATFORM_VENDOR
	PlatformExtensions = C.CL_PLATFORM_EXTENSIONS
)

// 设备信息类型
const (
	DeviceName         = C.CL_DEVICE_NAME
	DeviceVendor       = C.CL_DEVICE_VENDOR
	DeviceVersion      = C.CL_DEVICE_VERSION
	DeviceType         = C.CL_DEVICE_TYPE
	DeviceMaxMemAlloc  = C.CL_DEVICE_MAX_MEM_ALLOC_SIZE
	DeviceMaxWorkGroup = C.CL_DEVICE_MAX_WORK_GROUP_SIZE
)

// 上下文属性
const (
	ContextPlatform = C.CL_CONTEXT_PLATFORM
)

// 命令队列属性
const (
	QueueProperties = C.CL_QUEUE_PROPERTIES
)

// 命令队列属性值
const (
	QueueOutOfOrderExecModeEnable = C.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
	QueueProfilingEnable          = C.CL_QUEUE_PROFILING_ENABLE
)

// OpenCLError 自定义错误类型
type OpenCLError struct {
	Code Int
}

func (e OpenCLError) Error() string {
	return ErrorString(e.Code)
}

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
