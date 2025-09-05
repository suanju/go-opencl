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
import (
	"unsafe"
)

func CreateCommandQueue(context Context, device DeviceID, properties UInt) (CommandQueue, error) {
	var err C.cl_int

	queue := C.clCreateCommandQueue(
		C.cl_context(context),
		C.cl_device_id(device),
		C.cl_command_queue_properties(properties),
		&err,
	)

	if err != C.CL_SUCCESS {
		return CommandQueue(nil), OpenCLError{Code: Int(err)}
	}

	return CommandQueue(queue), nil
}

func CreateCommandQueueWithProperties(context Context, device DeviceID, properties map[UInt]interface{}) (CommandQueue, error) {
	var err C.cl_int
	var queue C.cl_command_queue

	// 准备属性数组
	var propertiesArray []C.cl_queue_properties
	if properties != nil {
		for key, value := range properties {
			propertiesArray = append(propertiesArray, C.cl_queue_properties(key))
			switch v := value.(type) {
			case UInt:
				propertiesArray = append(propertiesArray, C.cl_queue_properties(v))
			default:
				return CommandQueue(nil), OpenCLError{Code: Int(C.CL_INVALID_VALUE)}
			}
		}
		propertiesArray = append(propertiesArray, 0) // 结束标记
	}

	// 创建命令队列
	if len(propertiesArray) > 0 {
		queue = C.clCreateCommandQueueWithProperties(
			C.cl_context(context),
			C.cl_device_id(device),
			&propertiesArray[0],
			&err,
		)
	} else {
		queue = C.clCreateCommandQueueWithProperties(
			C.cl_context(context),
			C.cl_device_id(device),
			nil,
			&err,
		)
	}

	if err != C.CL_SUCCESS {
		return CommandQueue(nil), OpenCLError{Code: Int(err)}
	}

	return CommandQueue(queue), nil
}
func ReleaseCommandQueue(queue CommandQueue) error {
	err := C.clReleaseCommandQueue(C.cl_command_queue(queue))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

func RetainCommandQueue(queue CommandQueue) error {
	err := C.clRetainCommandQueue(C.cl_command_queue(queue))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}
func GetCommandQueueInfo(queue CommandQueue, paramName UInt, paramValueSize Size) ([]byte, error) {
	var paramValueSizeRet C.size_t
	paramValue := make([]byte, paramValueSize)

	err := C.clGetCommandQueueInfo(
		C.cl_command_queue(queue),
		C.cl_command_queue_info(paramName),
		C.size_t(paramValueSize),
		unsafe.Pointer(&paramValue[0]),
		&paramValueSizeRet,
	)

	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	return paramValue[:paramValueSizeRet], nil
}

func Flush(queue CommandQueue) error {
	err := C.clFlush(C.cl_command_queue(queue))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}
func Finish(queue CommandQueue) error {
	err := C.clFinish(C.cl_command_queue(queue))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

func GetCommandQueueContext(queue CommandQueue) (Context, error) {
	var contextPtr C.cl_context
	info, err := GetCommandQueueInfo(queue, C.CL_QUEUE_CONTEXT, Size(unsafe.Sizeof(contextPtr)))
	if err != nil {
		return Context(nil), err
	}

	context := *(*C.cl_context)(unsafe.Pointer(&info[0]))
	return Context(context), nil
}
func GetCommandQueueDevice(queue CommandQueue) (DeviceID, error) {
	var devicePtr C.cl_device_id
	info, err := GetCommandQueueInfo(queue, C.CL_QUEUE_DEVICE, Size(unsafe.Sizeof(devicePtr)))
	if err != nil {
		return DeviceID(nil), err
	}

	device := *(*C.cl_device_id)(unsafe.Pointer(&info[0]))
	return DeviceID(device), nil
}

func GetCommandQueueProperties(queue CommandQueue) (UInt, error) {
	info, err := GetCommandQueueInfo(queue, C.CL_QUEUE_PROPERTIES, Size(unsafe.Sizeof(C.cl_command_queue_properties(0))))
	if err != nil {
		return 0, err
	}

	properties := *(*C.cl_command_queue_properties)(unsafe.Pointer(&info[0]))
	return UInt(properties), nil
}
