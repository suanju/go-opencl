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

// CreateContext 创建OpenCL上下文
// 参数:
//   - platform: 平台ID
//   - devices: 设备ID列表
//   - properties: 上下文属性（可选）
//
// 返回:
//   - Context: 创建的上下文
//   - error: 错误信息
func CreateContext(platform PlatformID, devices []DeviceID, properties map[UInt]interface{}) (Context, error) {
	var err C.cl_int
	var context C.cl_context

	// 准备设备数组
	deviceCount := C.cl_uint(len(devices))
	if deviceCount == 0 {
		return Context(nil), OpenCLError{Code: Int(C.CL_INVALID_VALUE)}
	}

	deviceArray := make([]C.cl_device_id, deviceCount)
	for i, device := range devices {
		deviceArray[i] = C.cl_device_id(device)
	}

	// 准备属性数组
	var propertiesArray []C.cl_context_properties
	if properties != nil {
		for key, value := range properties {
			propertiesArray = append(propertiesArray, C.cl_context_properties(key))
			switch v := value.(type) {
			case PlatformID:
				propertiesArray = append(propertiesArray, C.cl_context_properties(uintptr(unsafe.Pointer(v))))
			case UInt:
				propertiesArray = append(propertiesArray, C.cl_context_properties(v))
			default:
				return Context(nil), OpenCLError{Code: Int(C.CL_INVALID_VALUE)}
			}
		}
		propertiesArray = append(propertiesArray, 0) // 结束标记
	}

	// 创建上下文
	if len(propertiesArray) > 0 {
		context = C.clCreateContext(
			&propertiesArray[0],
			deviceCount,
			&deviceArray[0],
			nil, // 回调函数
			nil, // 用户数据
			&err,
		)
	} else {
		context = C.clCreateContext(
			nil,
			deviceCount,
			&deviceArray[0],
			nil, // 回调函数
			nil, // 用户数据
			&err,
		)
	}

	if err != C.CL_SUCCESS {
		return Context(nil), OpenCLError{Code: Int(err)}
	}

	return Context(context), nil
}

// CreateContextFromType 根据设备类型创建上下文
// 参数:
//   - platform: 平台ID
//   - deviceType: 设备类型（如DeviceTypeGPU）
//   - properties: 上下文属性（可选）
//
// 返回:
//   - Context: 创建的上下文
//   - error: 错误信息
func CreateContextFromType(platform PlatformID, deviceType UInt, properties map[UInt]interface{}) (Context, error) {
	var err C.cl_int
	var context C.cl_context

	// 准备属性数组
	var propertiesArray []C.cl_context_properties
	if properties != nil {
		for key, value := range properties {
			propertiesArray = append(propertiesArray, C.cl_context_properties(key))
			switch v := value.(type) {
			case PlatformID:
				propertiesArray = append(propertiesArray, C.cl_context_properties(uintptr(unsafe.Pointer(v))))
			case UInt:
				propertiesArray = append(propertiesArray, C.cl_context_properties(v))
			default:
				return Context(nil), OpenCLError{Code: Int(C.CL_INVALID_VALUE)}
			}
		}
		propertiesArray = append(propertiesArray, 0) // 结束标记
	}

	// 创建上下文
	if len(propertiesArray) > 0 {
		context = C.clCreateContextFromType(
			&propertiesArray[0],
			C.cl_device_type(deviceType),
			nil, // 回调函数
			nil, // 用户数据
			&err,
		)
	} else {
		context = C.clCreateContextFromType(
			nil,
			C.cl_device_type(deviceType),
			nil, // 回调函数
			nil, // 用户数据
			&err,
		)
	}

	if err != C.CL_SUCCESS {
		return Context(nil), OpenCLError{Code: Int(err)}
	}

	return Context(context), nil
}

// ReleaseContext 释放上下文资源
// 参数:
//   - context: 要释放的上下文
//
// 返回:
//   - error: 错误信息
func ReleaseContext(context Context) error {
	err := C.clReleaseContext(C.cl_context(context))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

// RetainContext 增加上下文的引用计数
// 参数:
//   - context: 上下文
//
// 返回:
//   - error: 错误信息
func RetainContext(context Context) error {
	err := C.clRetainContext(C.cl_context(context))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

// GetContextInfo 获取上下文信息
// 参数:
//   - context: 上下文
//   - paramName: 参数名称
//   - paramValueSize: 参数值缓冲区大小
//
// 返回:
//   - []byte: 参数值
//   - error: 错误信息
func GetContextInfo(context Context, paramName UInt, paramValueSize Size) ([]byte, error) {
	var paramValueSizeRet C.size_t

	// 第一次调用，获取需要的大小
	err := C.clGetContextInfo(
		C.cl_context(context),
		C.cl_context_info(paramName),
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

	// 分配缓冲区
	paramValue := make([]byte, paramValueSizeRet)

	// 第二次调用，真正获取数据
	err = C.clGetContextInfo(
		C.cl_context(context),
		C.cl_context_info(paramName),
		paramValueSizeRet,
		unsafe.Pointer(&paramValue[0]),
		nil,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	return paramValue, nil
}

// GetContextDevices 获取上下文中的设备列表
// 参数:
//   - context: 上下文
//
// 返回:
//   - []DeviceID: 设备ID列表
//   - error: 错误信息
func GetContextDevices(context Context) ([]DeviceID, error) {
	// 首先获取设备数量
	devicesInfo, err := GetContextInfo(context, C.CL_CONTEXT_DEVICES, 0)
	if err != nil {
		return nil, err
	}

	var devicePtr C.cl_device_id
	deviceCount := len(devicesInfo) / int(unsafe.Sizeof(devicePtr))
	if deviceCount == 0 {
		return []DeviceID{}, nil
	}

	// 获取设备列表
	devicesInfo, err = GetContextInfo(context, C.CL_CONTEXT_DEVICES, Size(deviceCount*int(unsafe.Sizeof(devicePtr))))
	if err != nil {
		return nil, err
	}

	devices := make([]DeviceID, deviceCount)
	for i := 0; i < deviceCount; i++ {
		offset := i * int(unsafe.Sizeof(devicePtr))
		devices[i] = DeviceID(*(*C.cl_device_id)(unsafe.Pointer(&devicesInfo[offset])))
	}

	return devices, nil
}
