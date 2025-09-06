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
	props := map[UInt]any{}
	if properties != 0 {
		props[C.CL_QUEUE_PROPERTIES] = properties
	}
	return CreateCommandQueueWithProperties(context, device, props)
}
func CreateCommandQueueWithProperties(
	context Context,
	device DeviceID,
	properties map[UInt]any,
) (CommandQueue, error) {
	var err C.cl_int
	var queue C.cl_command_queue
	var propsPtr *C.cl_queue_properties

	if len(properties) > 0 {
		// 每个属性是 key+value 两个元素，最后还要加一个 0 作为结束符
		propertiesArray := make([]C.cl_queue_properties, 0, len(properties)*2+1)

		for key, value := range properties {
			propertiesArray = append(propertiesArray, C.cl_queue_properties(key))
			switch v := value.(type) {
			case UInt:
				propertiesArray = append(propertiesArray, C.cl_queue_properties(v))
			case int:
				propertiesArray = append(propertiesArray, C.cl_queue_properties(v))
			case uint64:
				propertiesArray = append(propertiesArray, C.cl_queue_properties(v))
			default:
				return CommandQueue(nil), OpenCLError{Code: Int(C.CL_INVALID_VALUE)}
			}
		}

		// 结束标记
		propertiesArray = append(propertiesArray, 0)
		propsPtr = &propertiesArray[0]
	}

	queue = C.clCreateCommandQueueWithProperties(
		C.cl_context(context),
		C.cl_device_id(device),
		propsPtr,
		&err,
	)

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

// EnqueueNDRangeKernel 提交内核执行任务，指定工作项维度
func EnqueueNDRangeKernel(
	queue CommandQueue,
	kernel Kernel,
	workDim UInt,
	globalWorkOffset []Size,
	globalWorkSize []Size,
	localWorkSize []Size,
	eventWaitList []Event,
	event *Event,
) error {
	var err C.cl_int
	var eventPtr *C.cl_event

	// 处理事件等待列表
	var eventWaitListPtr *C.cl_event
	var numEventsInWaitList C.cl_uint
	if len(eventWaitList) > 0 {
		events := make([]C.cl_event, len(eventWaitList))
		for i, event := range eventWaitList {
			events[i] = C.cl_event(event)
		}
		eventWaitListPtr = &events[0]
		numEventsInWaitList = C.cl_uint(len(eventWaitList))
	}

	// 处理输出事件
	if event != nil {
		eventPtr = (*C.cl_event)(unsafe.Pointer(event))
	}

	// 处理全局工作偏移
	var globalWorkOffsetPtr *C.size_t
	if len(globalWorkOffset) > 0 {
		offsets := make([]C.size_t, len(globalWorkOffset))
		for i, offset := range globalWorkOffset {
			offsets[i] = C.size_t(offset)
		}
		globalWorkOffsetPtr = &offsets[0]
	}

	// 处理全局工作大小
	var globalWorkSizePtr *C.size_t
	if len(globalWorkSize) > 0 {
		sizes := make([]C.size_t, len(globalWorkSize))
		for i, size := range globalWorkSize {
			sizes[i] = C.size_t(size)
		}
		globalWorkSizePtr = &sizes[0]
	}

	// 处理本地工作大小
	var localWorkSizePtr *C.size_t
	if len(localWorkSize) > 0 {
		localSizes := make([]C.size_t, len(localWorkSize))
		for i, size := range localWorkSize {
			localSizes[i] = C.size_t(size)
		}
		localWorkSizePtr = &localSizes[0]
	}

	err = C.clEnqueueNDRangeKernel(
		C.cl_command_queue(queue),
		C.cl_kernel(kernel),
		C.cl_uint(workDim),
		globalWorkOffsetPtr,
		globalWorkSizePtr,
		localWorkSizePtr,
		numEventsInWaitList,
		eventWaitListPtr,
		eventPtr,
	)

	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}

	return nil
}

// EnqueueTask 提交任务执行（1D工作项）
func EnqueueTask(
	queue CommandQueue,
	kernel Kernel,
	eventWaitList []Event,
	event *Event,
) error {
	var err C.cl_int
	var eventPtr *C.cl_event

	// 处理事件等待列表
	var eventWaitListPtr *C.cl_event
	var numEventsInWaitList C.cl_uint
	if len(eventWaitList) > 0 {
		events := make([]C.cl_event, len(eventWaitList))
		for i, e := range eventWaitList {
			events[i] = C.cl_event(e)
		}
		eventWaitListPtr = &events[0]
		numEventsInWaitList = C.cl_uint(len(eventWaitList))
	}

	// 输出事件
	if event != nil {
		eventPtr = (*C.cl_event)(unsafe.Pointer(event))
	}

	// 使用 NDRange 提交一个单工作项（等效于旧的 clEnqueueTask）
	var globalSize C.size_t = 1
	err = C.clEnqueueNDRangeKernel(
		C.cl_command_queue(queue),
		C.cl_kernel(kernel),
		C.cl_uint(1), // workDim = 1
		nil,          // global_work_offset
		&globalSize,  // global_work_size = [1]
		nil,          // local_work_size = NULL
		numEventsInWaitList,
		eventWaitListPtr,
		eventPtr,
	)

	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

// EnqueueMarker 在命令队列中插入标记
func EnqueueMarker(queue CommandQueue, event *Event) error {
	var err C.cl_int
	var eventPtr *C.cl_event

	if event != nil {
		eventPtr = (*C.cl_event)(unsafe.Pointer(event))
	}

	// 使用带等待列表的接口，传 0 和 NULL 表示无等待
	err = C.clEnqueueMarkerWithWaitList(
		C.cl_command_queue(queue),
		0,
		nil,
		eventPtr,
	)

	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

// EnqueueBarrier 在命令队列中插入屏障
func EnqueueBarrier(queue CommandQueue) error {
	// 新接口同样接受等待列表；这里直接传 0 / NULL
	err := C.clEnqueueBarrierWithWaitList(
		C.cl_command_queue(queue),
		0,
		nil,
		nil,
	)
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}
