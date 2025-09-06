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

func CreateBuffer(context Context, flags UInt, size Size, hostPtr unsafe.Pointer) (MemObject, error) {
	var err C.cl_int

	buffer := C.clCreateBuffer(
		C.cl_context(context),
		C.cl_mem_flags(flags),
		C.size_t(size),
		hostPtr,
		&err,
	)

	if err != C.CL_SUCCESS {
		return MemObject(nil), OpenCLError{Code: Int(err)}
	}

	return MemObject(buffer), nil
}

func CreateSubBuffer(buffer MemObject, flags UInt, bufferCreateType UInt, bufferCreateInfo unsafe.Pointer) (MemObject, error) {
	var err C.cl_int

	subBuffer := C.clCreateSubBuffer(
		C.cl_mem(buffer),
		C.cl_mem_flags(flags),
		C.cl_buffer_create_type(bufferCreateType),
		bufferCreateInfo,
		&err,
	)

	if err != C.CL_SUCCESS {
		return MemObject(nil), OpenCLError{Code: Int(err)}
	}

	return MemObject(subBuffer), nil
}

func ReleaseMemObject(memObj MemObject) error {
	err := C.clReleaseMemObject(C.cl_mem(memObj))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

func RetainMemObject(memObj MemObject) error {
	err := C.clRetainMemObject(C.cl_mem(memObj))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

func GetMemObjectInfo(memObj MemObject, paramName UInt) ([]byte, error) {
	var paramValueSizeRet C.size_t

	// 第一次调用，获取需要的大小
	err := C.clGetMemObjectInfo(
		C.cl_mem(memObj),
		C.cl_mem_info(paramName),
		0,
		nil,
		&paramValueSizeRet,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	if paramValueSizeRet == 0 {
		return nil, nil
	}

	paramValue := make([]byte, paramValueSizeRet)

	// 第二次调用，真正获取数据
	err = C.clGetMemObjectInfo(
		C.cl_mem(memObj),
		C.cl_mem_info(paramName),
		paramValueSizeRet,
		unsafe.Pointer(&paramValue[0]),
		nil,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	return paramValue, nil
}

func GetMemObjectSize(memObj MemObject) (Size, error) {
	info, err := GetMemObjectInfo(memObj, C.CL_MEM_SIZE)
	if err != nil {
		return 0, err
	}

	size := *(*C.size_t)(unsafe.Pointer(&info[0]))
	return Size(size), nil
}

func GetMemObjectFlags(memObj MemObject) (UInt, error) {
	info, err := GetMemObjectInfo(memObj, C.CL_MEM_FLAGS)
	if err != nil {
		return 0, err
	}

	flags := *(*C.cl_mem_flags)(unsafe.Pointer(&info[0]))
	return UInt(flags), nil
}

func GetMemObjectContext(memObj MemObject) (Context, error) {
	info, err := GetMemObjectInfo(memObj, C.CL_MEM_CONTEXT)
	if err != nil {
		return Context(nil), err
	}

	context := *(*C.cl_context)(unsafe.Pointer(&info[0]))
	return Context(context), nil
}

func EnqueueReadBuffer(queue CommandQueue, buffer MemObject, blocking Bool, offset Size, size Size, ptr unsafe.Pointer, eventWaitList []Event) (Event, error) {
	var err C.cl_int
	var event C.cl_event

	var waitList *C.cl_event
	var waitListSize C.cl_uint
	if len(eventWaitList) > 0 {
		waitListSize = C.cl_uint(len(eventWaitList))
		waitListArray := make([]C.cl_event, len(eventWaitList))
		for i, e := range eventWaitList {
			waitListArray[i] = C.cl_event(e)
		}
		waitList = &waitListArray[0]
	}

	err = C.clEnqueueReadBuffer(
		C.cl_command_queue(queue),
		C.cl_mem(buffer),
		C.cl_bool(blocking),
		C.size_t(offset),
		C.size_t(size),
		ptr,
		waitListSize,
		waitList,
		&event,
	)

	if err != C.CL_SUCCESS {
		return Event(nil), OpenCLError{Code: Int(err)}
	}

	return Event(event), nil
}

func EnqueueWriteBuffer(queue CommandQueue, buffer MemObject, blocking Bool, offset Size, size Size, ptr unsafe.Pointer, eventWaitList []Event) (Event, error) {
	var err C.cl_int
	var event C.cl_event

	var waitList *C.cl_event
	var waitListSize C.cl_uint
	if len(eventWaitList) > 0 {
		waitListSize = C.cl_uint(len(eventWaitList))
		waitListArray := make([]C.cl_event, len(eventWaitList))
		for i, e := range eventWaitList {
			waitListArray[i] = C.cl_event(e)
		}
		waitList = &waitListArray[0]
	}

	err = C.clEnqueueWriteBuffer(
		C.cl_command_queue(queue),
		C.cl_mem(buffer),
		C.cl_bool(blocking),
		C.size_t(offset),
		C.size_t(size),
		ptr,
		waitListSize,
		waitList,
		&event,
	)

	if err != C.CL_SUCCESS {
		return Event(nil), OpenCLError{Code: Int(err)}
	}

	return Event(event), nil
}

func EnqueueCopyBuffer(queue CommandQueue, srcBuffer MemObject, dstBuffer MemObject, srcOffset Size, dstOffset Size, size Size, eventWaitList []Event) (Event, error) {
	var err C.cl_int
	var event C.cl_event

	var waitList *C.cl_event
	var waitListSize C.cl_uint
	if len(eventWaitList) > 0 {
		waitListSize = C.cl_uint(len(eventWaitList))
		waitListArray := make([]C.cl_event, len(eventWaitList))
		for i, e := range eventWaitList {
			waitListArray[i] = C.cl_event(e)
		}
		waitList = &waitListArray[0]
	}

	err = C.clEnqueueCopyBuffer(
		C.cl_command_queue(queue),
		C.cl_mem(srcBuffer),
		C.cl_mem(dstBuffer),
		C.size_t(srcOffset),
		C.size_t(dstOffset),
		C.size_t(size),
		waitListSize,
		waitList,
		&event,
	)

	if err != C.CL_SUCCESS {
		return Event(nil), OpenCLError{Code: Int(err)}
	}

	return Event(event), nil
}

func EnqueueMapBuffer(queue CommandQueue, buffer MemObject, blocking Bool, mapFlags UInt, offset Size, size Size, eventWaitList []Event) (unsafe.Pointer, Event, error) {
	var err C.cl_int
	var event C.cl_event

	var waitList *C.cl_event
	var waitListSize C.cl_uint
	if len(eventWaitList) > 0 {
		waitListSize = C.cl_uint(len(eventWaitList))
		waitListArray := make([]C.cl_event, len(eventWaitList))
		for i, e := range eventWaitList {
			waitListArray[i] = C.cl_event(e)
		}
		waitList = &waitListArray[0]
	}

	ptr := C.clEnqueueMapBuffer(
		C.cl_command_queue(queue),
		C.cl_mem(buffer),
		C.cl_bool(blocking),
		C.cl_map_flags(mapFlags),
		C.size_t(offset),
		C.size_t(size),
		waitListSize,
		waitList,
		&event,
		&err,
	)

	if err != C.CL_SUCCESS {
		return nil, Event(nil), OpenCLError{Code: Int(err)}
	}

	return ptr, Event(event), nil
}

func EnqueueUnmapMemObject(queue CommandQueue, memObj MemObject, mappedPtr unsafe.Pointer, eventWaitList []Event) (Event, error) {
	var err C.cl_int
	var event C.cl_event

	var waitList *C.cl_event
	var waitListSize C.cl_uint
	if len(eventWaitList) > 0 {
		waitListSize = C.cl_uint(len(eventWaitList))
		waitListArray := make([]C.cl_event, len(eventWaitList))
		for i, e := range eventWaitList {
			waitListArray[i] = C.cl_event(e)
		}
		waitList = &waitListArray[0]
	}

	err = C.clEnqueueUnmapMemObject(
		C.cl_command_queue(queue),
		C.cl_mem(memObj),
		mappedPtr,
		waitListSize,
		waitList,
		&event,
	)

	if err != C.CL_SUCCESS {
		return Event(nil), OpenCLError{Code: Int(err)}
	}

	return Event(event), nil
}
