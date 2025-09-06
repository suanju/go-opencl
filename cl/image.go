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

func CreateImage2D(
	context Context,
	flags UInt,
	imageFormat ImageFormat,
	imageWidth Size,
	imageHeight Size,
	imageRowPitch Size,
	hostPtr unsafe.Pointer,
) (MemObject, error) {
	var err C.cl_int

	format := C.cl_image_format{
		image_channel_order:     C.cl_channel_order(imageFormat.ChannelOrder),
		image_channel_data_type: C.cl_channel_type(imageFormat.ChannelType),
	}

	desc := C.cl_image_desc{
		image_type:        C.CL_MEM_OBJECT_IMAGE2D,
		image_width:       C.size_t(imageWidth),
		image_height:      C.size_t(imageHeight),
		image_depth:       1, // 2D 必须为 1
		image_array_size:  0, // 非数组图像
		image_row_pitch:   C.size_t(imageRowPitch),
		image_slice_pitch: 0, // 2D 无 slice pitch
		num_mip_levels:    0,
		num_samples:       0,
	}

	image := C.clCreateImage(
		C.cl_context(context),
		C.cl_mem_flags(flags),
		&format,
		&desc,
		hostPtr,
		&err,
	)
	if err != C.CL_SUCCESS {
		return MemObject(nil), OpenCLError{Code: Int(err)}
	}
	return MemObject(image), nil
}

func CreateImage3D(
	context Context,
	flags UInt,
	imageFormat ImageFormat,
	imageWidth Size,
	imageHeight Size,
	imageDepth Size,
	imageRowPitch Size,
	imageSlicePitch Size,
	hostPtr unsafe.Pointer,
) (MemObject, error) {
	var err C.cl_int

	format := C.cl_image_format{
		image_channel_order:     C.cl_channel_order(imageFormat.ChannelOrder),
		image_channel_data_type: C.cl_channel_type(imageFormat.ChannelType),
	}
	desc := C.cl_image_desc{
		image_type:        C.CL_MEM_OBJECT_IMAGE3D,
		image_width:       C.size_t(imageWidth),
		image_height:      C.size_t(imageHeight),
		image_depth:       C.size_t(imageDepth),
		image_array_size:  0,
		image_row_pitch:   C.size_t(imageRowPitch),
		image_slice_pitch: C.size_t(imageSlicePitch),
		num_mip_levels:    0,
		num_samples:       0,
		// buffer 零值 = NULL（如字段存在）
	}

	image := C.clCreateImage(
		C.cl_context(context),
		C.cl_mem_flags(flags),
		&format,
		&desc,
		hostPtr,
		&err,
	)
	if err != C.CL_SUCCESS {
		return MemObject(nil), OpenCLError{Code: Int(err)}
	}
	return MemObject(image), nil
}

func CreateImage(context Context, flags UInt, imageFormat ImageFormat, imageDesc ImageDesc, hostPtr unsafe.Pointer) (MemObject, error) {
	var err C.cl_int

	format := C.cl_image_format{
		image_channel_order:     C.cl_channel_order(imageFormat.ChannelOrder),
		image_channel_data_type: C.cl_channel_type(imageFormat.ChannelType),
	}

	desc := C.cl_image_desc{
		image_type:        C.cl_mem_object_type(imageDesc.ImageType),
		image_width:       C.size_t(imageDesc.Width),
		image_height:      C.size_t(imageDesc.Height),
		image_depth:       C.size_t(imageDesc.Depth),
		image_array_size:  C.size_t(imageDesc.ArraySize),
		image_row_pitch:   C.size_t(imageDesc.RowPitch),
		image_slice_pitch: C.size_t(imageDesc.SlicePitch),
		num_mip_levels:    C.cl_uint(imageDesc.NumMipLevels),
		num_samples:       C.cl_uint(imageDesc.NumSamples),
	}

	image := C.clCreateImage(
		C.cl_context(context),
		C.cl_mem_flags(flags),
		&format,
		&desc,
		hostPtr,
		&err,
	)

	if err != C.CL_SUCCESS {
		return MemObject(nil), OpenCLError{Code: Int(err)}
	}

	return MemObject(image), nil
}

func GetSupportedImageFormats(context Context, flags UInt, imageType UInt) ([]ImageFormat, error) {
	var numFormats C.cl_uint

	// 第一次调用，获取格式数量
	err := C.clGetSupportedImageFormats(
		C.cl_context(context),
		C.cl_mem_flags(flags),
		C.cl_mem_object_type(imageType),
		0,
		nil,
		&numFormats,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	if numFormats == 0 {
		return []ImageFormat{}, nil
	}

	// 分配格式数组
	formats := make([]C.cl_image_format, numFormats)

	// 第二次调用，获取格式数据
	err = C.clGetSupportedImageFormats(
		C.cl_context(context),
		C.cl_mem_flags(flags),
		C.cl_mem_object_type(imageType),
		numFormats,
		&formats[0],
		nil,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	// 转换为Go格式
	result := make([]ImageFormat, numFormats)
	for i, format := range formats {
		result[i] = ImageFormat{
			ChannelOrder: UInt(format.image_channel_order),
			ChannelType:  UInt(format.image_channel_data_type),
		}
	}

	return result, nil
}

func EnqueueReadImage(queue CommandQueue, image MemObject, blocking Bool, origin [3]Size, region [3]Size, rowPitch Size, slicePitch Size, ptr unsafe.Pointer, eventWaitList []Event) (Event, error) {
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

	originArray := [3]C.size_t{C.size_t(origin[0]), C.size_t(origin[1]), C.size_t(origin[2])}
	regionArray := [3]C.size_t{C.size_t(region[0]), C.size_t(region[1]), C.size_t(region[2])}

	err = C.clEnqueueReadImage(
		C.cl_command_queue(queue),
		C.cl_mem(image),
		C.cl_bool(blocking),
		&originArray[0],
		&regionArray[0],
		C.size_t(rowPitch),
		C.size_t(slicePitch),
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

func EnqueueWriteImage(queue CommandQueue, image MemObject, blocking Bool, origin [3]Size, region [3]Size, rowPitch Size, slicePitch Size, ptr unsafe.Pointer, eventWaitList []Event) (Event, error) {
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

	originArray := [3]C.size_t{C.size_t(origin[0]), C.size_t(origin[1]), C.size_t(origin[2])}
	regionArray := [3]C.size_t{C.size_t(region[0]), C.size_t(region[1]), C.size_t(region[2])}

	err = C.clEnqueueWriteImage(
		C.cl_command_queue(queue),
		C.cl_mem(image),
		C.cl_bool(blocking),
		&originArray[0],
		&regionArray[0],
		C.size_t(rowPitch),
		C.size_t(slicePitch),
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

func EnqueueCopyImage(queue CommandQueue, srcImage MemObject, dstImage MemObject, srcOrigin [3]Size, dstOrigin [3]Size, region [3]Size, eventWaitList []Event) (Event, error) {
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

	srcOriginArray := [3]C.size_t{C.size_t(srcOrigin[0]), C.size_t(srcOrigin[1]), C.size_t(srcOrigin[2])}
	dstOriginArray := [3]C.size_t{C.size_t(dstOrigin[0]), C.size_t(dstOrigin[1]), C.size_t(dstOrigin[2])}
	regionArray := [3]C.size_t{C.size_t(region[0]), C.size_t(region[1]), C.size_t(region[2])}

	err = C.clEnqueueCopyImage(
		C.cl_command_queue(queue),
		C.cl_mem(srcImage),
		C.cl_mem(dstImage),
		&srcOriginArray[0],
		&dstOriginArray[0],
		&regionArray[0],
		waitListSize,
		waitList,
		&event,
	)

	if err != C.CL_SUCCESS {
		return Event(nil), OpenCLError{Code: Int(err)}
	}

	return Event(event), nil
}

func EnqueueMapImage(queue CommandQueue, image MemObject, blocking Bool, mapFlags UInt, origin [3]Size, region [3]Size, imageRowPitch *Size, imageSlicePitch *Size, eventWaitList []Event) (unsafe.Pointer, Size, Size, Event, error) {
	var err C.cl_int
	var event C.cl_event
	var rowPitch C.size_t
	var slicePitch C.size_t

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

	originArray := [3]C.size_t{C.size_t(origin[0]), C.size_t(origin[1]), C.size_t(origin[2])}
	regionArray := [3]C.size_t{C.size_t(region[0]), C.size_t(region[1]), C.size_t(region[2])}

	ptr := C.clEnqueueMapImage(
		C.cl_command_queue(queue),
		C.cl_mem(image),
		C.cl_bool(blocking),
		C.cl_map_flags(mapFlags),
		&originArray[0],
		&regionArray[0],
		&rowPitch,
		&slicePitch,
		waitListSize,
		waitList,
		&event,
		&err,
	)

	if err != C.CL_SUCCESS {
		return nil, 0, 0, Event(nil), OpenCLError{Code: Int(err)}
	}

	return ptr, Size(rowPitch), Size(slicePitch), Event(event), nil
}
