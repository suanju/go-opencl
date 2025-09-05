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
	"fmt"
	"unsafe"
)

// GetDeviceIDs 获取指定平台下的设备
func GetDeviceIDs(platform PlatformID, deviceType C.cl_device_type) ([]DeviceID, error) {
	var num C.cl_uint
	if err := C.clGetDeviceIDs(C.cl_platform_id(platform), deviceType, 0, nil, &num); err != C.CL_SUCCESS {
		return nil, OpenCLError{Int(err)}
	}
	if num == 0 {
		return nil, nil
	}
	ids := make([]C.cl_device_id, num)
	if err := C.clGetDeviceIDs(C.cl_platform_id(platform), deviceType, num, &ids[0], nil); err != C.CL_SUCCESS {
		return nil, OpenCLError{Int(err)}
	}
	result := make([]DeviceID, num)
	for i := range ids {
		result[i] = DeviceID(ids[i])
	}
	return result, nil
}

// GetDeviceInfo 获取设备信息
func GetDeviceInfo(device DeviceID, paramName UInt) (string, error) {
	var size Size
	errCode := Int(C.clGetDeviceInfo(
		C.cl_device_id(device),
		C.cl_device_info(paramName),
		0, nil, (*C.size_t)(&size)))
	if errCode != Success {
		return "", OpenCLError{Code: errCode}
	}

	if size == 0 {
		return "", nil
	}

	buffer := make([]byte, size)
	errCode = Int(C.clGetDeviceInfo(
		C.cl_device_id(device),
		C.cl_device_info(paramName),
		C.size_t(size), unsafe.Pointer(&buffer[0]), nil))
	if errCode != Success {
		return "", OpenCLError{Code: errCode}
	}

	return string(buffer[:size-1]), nil // 去掉末尾的null字符
}

// GetDeviceInfoUInt 获取设备UInt类型信息
func GetDeviceInfoUInt(device DeviceID, paramName UInt) (UInt, error) {
	var value UInt
	errCode := Int(C.clGetDeviceInfo(
		C.cl_device_id(device),
		C.cl_device_info(paramName),
		C.size_t(unsafe.Sizeof(value)),
		unsafe.Pointer(&value), nil))
	if errCode != Success {
		return UInt(0), OpenCLError{Code: errCode}
	}
	return value, nil
}

// GetDeviceInfoSize 获取设备Size类型信息
func GetDeviceInfoSize(device DeviceID, paramName UInt) (Size, error) {
	var value Size
	errCode := Int(C.clGetDeviceInfo(
		C.cl_device_id(device),
		C.cl_device_info(paramName),
		C.size_t(unsafe.Sizeof(value)),
		unsafe.Pointer(&value), nil))
	if errCode != Success {
		return 0, OpenCLError{Code: errCode}
	}
	return value, nil
}

// GetDeviceInfoULong 获取设备 cl_ulong 类型信息
func GetDeviceInfoULong(device DeviceID, paramName UInt) (uint64, error) {
	var value C.cl_ulong
	errCode := Int(C.clGetDeviceInfo(
		C.cl_device_id(device),
		C.cl_device_info(paramName),
		C.size_t(unsafe.Sizeof(value)),
		unsafe.Pointer(&value), nil))
	if errCode != Success {
		return 0, OpenCLError{Code: errCode}
	}
	return uint64(value), nil
}

// GetDeviceDetails 获取设备完整信息
func GetDeviceDetails(device DeviceID) (*DeviceInfo, error) {
	info := &DeviceInfo{ID: device}

	var err error
	info.Name, err = GetDeviceInfo(device, DeviceName)
	if err != nil {
		fmt.Println("GetDeviceInfo DeviceName err:", err)
		return nil, err
	}

	info.Vendor, err = GetDeviceInfo(device, DeviceVendor)
	if err != nil {
		fmt.Println("GetDeviceInfo DeviceVendor err:", err)
		return nil, err
	}

	info.Version, err = GetDeviceInfo(device, DeviceVersion)
	if err != nil {
		fmt.Println("GetDeviceInfo DeviceVersion err:", err)
		return nil, err
	}

	info.Type, err = GetDeviceInfoULong(device, DeviceType)
	if err != nil {
		fmt.Println("GetDeviceInfoULong err:", err)
		return nil, err
	}

	info.MaxMemAlloc, err = GetDeviceInfoULong(device, DeviceMaxMemAlloc)
	if err != nil {
		fmt.Println("GetDeviceInfoUInt err:", err)
		return nil, err
	}

	info.MaxWorkGroup, err = GetDeviceInfoSize(device, DeviceMaxWorkGroup)
	if err != nil {
		fmt.Println("GetDeviceInfoSize err:", err)
		return nil, err
	}

	return info, nil
}
