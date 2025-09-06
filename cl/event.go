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

// CreateUserEvent 创建用户事件
func CreateUserEvent(context Context) (Event, error) {
	var err C.cl_int

	event := C.clCreateUserEvent(
		C.cl_context(context),
		&err,
	)

	if err != C.CL_SUCCESS {
		return Event(nil), OpenCLError{Code: Int(err)}
	}

	return Event(event), nil
}

// ReleaseEvent 释放事件
func ReleaseEvent(event Event) error {
	err := C.clReleaseEvent(C.cl_event(event))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

// RetainEvent 增加事件的引用计数
func RetainEvent(event Event) error {
	err := C.clRetainEvent(C.cl_event(event))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

// SetUserEventStatus 设置用户事件状态
func SetUserEventStatus(event Event, executionStatus Int) error {
	err := C.clSetUserEventStatus(
		C.cl_event(event),
		C.cl_int(executionStatus),
	)

	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}

	return nil
}

// WaitForEvents 等待事件列表中的所有事件完成
func WaitForEvents(eventList []Event) error {
	if len(eventList) == 0 {
		return nil
	}

	// 将Go的Event切片转换为C的cl_event数组
	events := make([]C.cl_event, len(eventList))
	for i, event := range eventList {
		events[i] = C.cl_event(event)
	}

	err := C.clWaitForEvents(
		C.cl_uint(len(eventList)),
		&events[0],
	)

	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}

	return nil
}

// GetEventInfo 获取事件信息
func GetEventInfo(event Event, paramName UInt, paramValueSize Size) ([]byte, error) {
	var paramValueSizeRet C.size_t
	paramValue := make([]byte, paramValueSize)

	err := C.clGetEventInfo(
		C.cl_event(event),
		C.cl_event_info(paramName),
		C.size_t(paramValueSize),
		unsafe.Pointer(&paramValue[0]),
		&paramValueSizeRet,
	)

	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	return paramValue[:paramValueSizeRet], nil
}

// GetEventCommandQueue 获取事件关联的命令队列
func GetEventCommandQueue(event Event) (CommandQueue, error) {
	var queuePtr C.cl_command_queue
	info, err := GetEventInfo(event, C.CL_EVENT_COMMAND_QUEUE, Size(unsafe.Sizeof(queuePtr)))
	if err != nil {
		return CommandQueue(nil), err
	}

	queue := *(*C.cl_command_queue)(unsafe.Pointer(&info[0]))
	return CommandQueue(queue), nil
}

// GetEventCommandType 获取事件命令类型
func GetEventCommandType(event Event) (UInt, error) {
	info, err := GetEventInfo(event, C.CL_EVENT_COMMAND_TYPE, Size(unsafe.Sizeof(C.cl_command_type(0))))
	if err != nil {
		return 0, err
	}

	commandType := *(*C.cl_command_type)(unsafe.Pointer(&info[0]))
	return UInt(commandType), nil
}

// GetEventCommandExecStatus 获取事件命令执行状态
func GetEventCommandExecStatus(event Event) (Int, error) {
	info, err := GetEventInfo(event, C.CL_EVENT_COMMAND_EXECUTION_STATUS, Size(unsafe.Sizeof(C.cl_int(0))))
	if err != nil {
		return 0, err
	}

	execStatus := *(*C.cl_int)(unsafe.Pointer(&info[0]))
	return Int(execStatus), nil
}

// GetEventContext 获取事件关联的上下文
func GetEventContext(event Event) (Context, error) {
	var contextPtr C.cl_context
	info, err := GetEventInfo(event, C.CL_EVENT_CONTEXT, Size(unsafe.Sizeof(contextPtr)))
	if err != nil {
		return Context(nil), err
	}

	context := *(*C.cl_context)(unsafe.Pointer(&info[0]))
	return Context(context), nil
}

// GetEventReferenceCount 获取事件引用计数
func GetEventReferenceCount(event Event) (UInt, error) {
	info, err := GetEventInfo(event, C.CL_EVENT_REFERENCE_COUNT, Size(unsafe.Sizeof(C.cl_uint(0))))
	if err != nil {
		return 0, err
	}

	refCount := *(*C.cl_uint)(unsafe.Pointer(&info[0]))
	return UInt(refCount), nil
}

// SetEventCallback 设置事件回调函数
func SetEventCallback(event Event, commandExecCallbackType UInt, callback func(Event, Int, unsafe.Pointer), userData unsafe.Pointer) error {
	// 注意：这里需要将Go的回调函数转换为C函数指针
	// 由于Go和C之间的回调函数转换比较复杂，这里提供一个简化版本
	// 实际使用时可能需要更复杂的C包装函数

	err := C.clSetEventCallback(
		C.cl_event(event),
		C.cl_int(commandExecCallbackType),
		nil, // 这里需要C函数指针，实际实现时需要C包装函数
		userData,
	)

	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}

	return nil
}
