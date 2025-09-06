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
import "fmt"

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

// 图像格式结构
type ImageFormat struct {
	ChannelOrder UInt
	ChannelType  UInt
}

// 图像描述符结构
type ImageDesc struct {
	ImageType    UInt
	Width        Size
	Height       Size
	Depth        Size
	ArraySize    Size
	RowPitch     Size
	SlicePitch   Size
	NumMipLevels UInt
	NumSamples   UInt
	Buffer       MemObject
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

// 内存对象标志
const (
	MemReadWrite     = C.CL_MEM_READ_WRITE
	MemWriteOnly     = C.CL_MEM_WRITE_ONLY
	MemReadOnly      = C.CL_MEM_READ_ONLY
	MemUseHostPtr    = C.CL_MEM_USE_HOST_PTR
	MemAllocHostPtr  = C.CL_MEM_ALLOC_HOST_PTR
	MemCopyHostPtr   = C.CL_MEM_COPY_HOST_PTR
	MemHostWriteOnly = C.CL_MEM_HOST_WRITE_ONLY
	MemHostReadOnly  = C.CL_MEM_HOST_READ_ONLY
	MemHostNoAccess  = C.CL_MEM_HOST_NO_ACCESS
)

// 图像通道顺序
const (
	ChannelOrderR         = C.CL_R
	ChannelOrderA         = C.CL_A
	ChannelOrderRG        = C.CL_RG
	ChannelOrderRA        = C.CL_RA
	ChannelOrderRGB       = C.CL_RGB
	ChannelOrderRGBA      = C.CL_RGBA
	ChannelOrderBGRA      = C.CL_BGRA
	ChannelOrderARGB      = C.CL_ARGB
	ChannelOrderIntensity = C.CL_INTENSITY
	ChannelOrderLuminance = C.CL_LUMINANCE
)

// 图像通道类型
const (
	ChannelTypeSNormInt8      = C.CL_SNORM_INT8
	ChannelTypeSNormInt16     = C.CL_SNORM_INT16
	ChannelTypeUNormInt8      = C.CL_UNORM_INT8
	ChannelTypeUNormInt16     = C.CL_UNORM_INT16
	ChannelTypeUNormShort565  = C.CL_UNORM_SHORT_565
	ChannelTypeUNormShort555  = C.CL_UNORM_SHORT_555
	ChannelTypeUNormInt101010 = C.CL_UNORM_INT_101010
	ChannelTypeSignedInt8     = C.CL_SIGNED_INT8
	ChannelTypeSignedInt16    = C.CL_SIGNED_INT16
	ChannelTypeSignedInt32    = C.CL_SIGNED_INT32
	ChannelTypeUnsignedInt8   = C.CL_UNSIGNED_INT8
	ChannelTypeUnsignedInt16  = C.CL_UNSIGNED_INT16
	ChannelTypeUnsignedInt32  = C.CL_UNSIGNED_INT32
	ChannelTypeHalfFloat      = C.CL_HALF_FLOAT
	ChannelTypeFloat          = C.CL_FLOAT
)

// 内存对象类型
const (
	MemObjectBuffer  = C.CL_MEM_OBJECT_BUFFER
	MemObjectImage2D = C.CL_MEM_OBJECT_IMAGE2D
	MemObjectImage3D = C.CL_MEM_OBJECT_IMAGE3D
)

// 映射标志
const (
	MapRead                  = C.CL_MAP_READ
	MapWrite                 = C.CL_MAP_WRITE
	MapWriteInvalidateRegion = C.CL_MAP_WRITE_INVALIDATE_REGION
)

// 程序构建状态
const (
	BuildSuccess    = C.CL_BUILD_SUCCESS
	BuildNone       = C.CL_BUILD_NONE
	BuildError      = C.CL_BUILD_ERROR
	BuildInProgress = C.CL_BUILD_IN_PROGRESS
)

// 程序信息类型
const (
	ProgramReferenceCount = C.CL_PROGRAM_REFERENCE_COUNT
	ProgramContext        = C.CL_PROGRAM_CONTEXT
	ProgramNumDevices     = C.CL_PROGRAM_NUM_DEVICES
	ProgramDevices        = C.CL_PROGRAM_DEVICES
	ProgramSource         = C.CL_PROGRAM_SOURCE
	ProgramBinarySizes    = C.CL_PROGRAM_BINARY_SIZES
	ProgramBinaries       = C.CL_PROGRAM_BINARIES
	ProgramNumKernels     = C.CL_PROGRAM_NUM_KERNELS
	ProgramKernelNames    = C.CL_PROGRAM_KERNEL_NAMES
)

// 内核信息类型
const (
	KernelFunctionName   = C.CL_KERNEL_FUNCTION_NAME
	KernelNumArgs        = C.CL_KERNEL_NUM_ARGS
	KernelReferenceCount = C.CL_KERNEL_REFERENCE_COUNT
	KernelContext        = C.CL_KERNEL_CONTEXT
	KernelProgram        = C.CL_KERNEL_PROGRAM
	KernelAttributes     = C.CL_KERNEL_ATTRIBUTES
)

// 内核工作项信息类型
const (
	KernelWorkGroupSize                  = C.CL_KERNEL_WORK_GROUP_SIZE
	KernelCompileWorkGroupSize           = C.CL_KERNEL_COMPILE_WORK_GROUP_SIZE
	KernelLocalMemSize                   = C.CL_KERNEL_LOCAL_MEM_SIZE
	KernelPreferredWorkGroupSizeMultiple = C.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
	KernelPrivateMemSize                 = C.CL_KERNEL_PRIVATE_MEM_SIZE
)

// 事件信息类型 - 使用数值常量避免编译错误
const (
	EventCommandQueue      = 0x11D0
	EventCommandType       = 0x11D1
	EventCommandExecStatus = 0x11D2
	EventReferenceCount    = 0x11D3
	EventContext           = 0x11D4
)

// 命令类型 - 使用数值常量避免编译错误
const (
	CommandNDRangeKernel     = 0x11F0
	CommandTask              = 0x11F1
	CommandNativeKernel      = 0x11F2
	CommandReadBuffer        = 0x11F3
	CommandWriteBuffer       = 0x11F4
	CommandCopyBuffer        = 0x11F5
	CommandReadImage         = 0x11F6
	CommandWriteImage        = 0x11F7
	CommandCopyImage         = 0x11F8
	CommandCopyImageToBuffer = 0x11F9
	CommandCopyBufferToImage = 0x11FA
	CommandMapBuffer         = 0x11FB
	CommandMapImage          = 0x11FC
	CommandUnmapMemObject    = 0x11FD
	CommandMarker            = 0x11FE
	CommandAcquireGLObjects  = 0x11FF
	CommandReleaseGLObjects  = 0x1200
	CommandReadBufferRect    = 0x1201
	CommandWriteBufferRect   = 0x1202
	CommandCopyBufferRect    = 0x1203
	CommandUser              = 0x1204
)

// 命令执行状态 - 使用数值常量避免编译错误
const (
	CommandQueued    = 0x1205
	CommandSubmitted = 0x1206
	CommandRunning   = 0x1207
	CommandComplete  = 0x1208
	CommandError     = 0x1209
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
	case C.CL_COMPILER_NOT_AVAILABLE:
		return "Compiler not available"
	case C.CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "Memory object allocation failure"
	case C.CL_OUT_OF_RESOURCES:
		return "Out of resources"
	case C.CL_OUT_OF_HOST_MEMORY:
		return "Out of host memory"
	case C.CL_PROFILING_INFO_NOT_AVAILABLE:
		return "Profiling information not available"
	case C.CL_MEM_COPY_OVERLAP:
		return "Memory copy overlap"
	case C.CL_IMAGE_FORMAT_MISMATCH:
		return "Image format mismatch"
	case C.CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "Image format not supported"
	case C.CL_BUILD_PROGRAM_FAILURE:
		return "Build program failure"
	case C.CL_MAP_FAILURE:
		return "Map failure"
	case C.CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return "Misaligned sub buffer offset"
	case C.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return "Execution status error for events in wait list"
	case C.CL_COMPILE_PROGRAM_FAILURE:
		return "Compile program failure"
	case C.CL_LINKER_NOT_AVAILABLE:
		return "Linker not available"
	case C.CL_LINK_PROGRAM_FAILURE:
		return "Link program failure"
	case C.CL_DEVICE_PARTITION_FAILED:
		return "Device partition failed"
	case C.CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
		return "Kernel argument info not available"
	case C.CL_INVALID_VALUE:
		return "Invalid value"
	case C.CL_INVALID_DEVICE_TYPE:
		return "Invalid device type"
	case C.CL_INVALID_PLATFORM:
		return "Invalid platform"
	case C.CL_INVALID_DEVICE:
		return "Invalid device"
	case C.CL_INVALID_CONTEXT:
		return "Invalid context"
	case C.CL_INVALID_QUEUE_PROPERTIES:
		return "Invalid queue properties"
	case C.CL_INVALID_COMMAND_QUEUE:
		return "Invalid command queue"
	case C.CL_INVALID_HOST_PTR:
		return "Invalid host pointer"
	case C.CL_INVALID_MEM_OBJECT:
		return "Invalid memory object"
	case C.CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "Invalid image format descriptor"
	case C.CL_INVALID_IMAGE_SIZE:
		return "Invalid image size"
	case C.CL_INVALID_SAMPLER:
		return "Invalid sampler"
	case C.CL_INVALID_BINARY:
		return "Invalid binary"
	case C.CL_INVALID_BUILD_OPTIONS:
		return "Invalid build options"
	case C.CL_INVALID_PROGRAM:
		return "Invalid program"
	case C.CL_INVALID_PROGRAM_EXECUTABLE:
		return "Invalid program executable"
	case C.CL_INVALID_KERNEL_NAME:
		return "Invalid kernel name"
	case C.CL_INVALID_KERNEL_DEFINITION:
		return "Invalid kernel definition"
	case C.CL_INVALID_KERNEL:
		return "Invalid kernel"
	case C.CL_INVALID_ARG_INDEX:
		return "Invalid argument index"
	case C.CL_INVALID_ARG_VALUE:
		return "Invalid argument value"
	case C.CL_INVALID_ARG_SIZE:
		return "Invalid argument size"
	case C.CL_INVALID_KERNEL_ARGS:
		return "Invalid kernel arguments"
	case C.CL_INVALID_WORK_DIMENSION:
		return "Invalid work dimension"
	case C.CL_INVALID_WORK_GROUP_SIZE:
		return "Invalid work group size"
	case C.CL_INVALID_WORK_ITEM_SIZE:
		return "Invalid work item size"
	case C.CL_INVALID_GLOBAL_OFFSET:
		return "Invalid global offset"
	case C.CL_INVALID_EVENT_WAIT_LIST:
		return "Invalid event wait list"
	case C.CL_INVALID_EVENT:
		return "Invalid event"
	case C.CL_INVALID_OPERATION:
		return "Invalid operation"
	case C.CL_INVALID_GL_OBJECT:
		return "Invalid GL object"
	case C.CL_INVALID_BUFFER_SIZE:
		return "Invalid buffer size"
	case C.CL_INVALID_MIP_LEVEL:
		return "Invalid mip level"
	case C.CL_INVALID_GLOBAL_WORK_SIZE:
		return "Invalid global work size"
	case C.CL_INVALID_PROPERTY:
		return "Invalid property"
	case C.CL_INVALID_IMAGE_DESCRIPTOR:
		return "Invalid image descriptor"
	case C.CL_INVALID_COMPILER_OPTIONS:
		return "Invalid compiler options"
	case C.CL_INVALID_LINKER_OPTIONS:
		return "Invalid linker options"
	case C.CL_INVALID_DEVICE_PARTITION_COUNT:
		return "Invalid device partition count"
	default:
		return fmt.Sprintf("Unknown error (code: %d)", err)
	}
}
