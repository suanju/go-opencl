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

// GetPlatformIDs 获取所有可用的 OpenCL 平台
func GetPlatformIDs() ([]PlatformID, error) {
	var num C.cl_uint
	if err := C.clGetPlatformIDs(0, nil, &num); err != C.CL_SUCCESS {
		return nil, OpenCLError{Int(err)}
	}
	if num == 0 {
		return nil, nil
	}
	ids := make([]C.cl_platform_id, num)
	if err := C.clGetPlatformIDs(num, &ids[0], nil); err != C.CL_SUCCESS {
		return nil, OpenCLError{Int(err)}
	}
	result := make([]PlatformID, num)
	for i := range ids {
		result[i] = PlatformID(ids[i])
	}
	return result, nil
}

// GetPlatformInfo 获取平台信息
func GetPlatformInfo(platform PlatformID, paramName UInt) (string, error) {
	var size Size
	errCode := Int(C.clGetPlatformInfo(
		C.cl_platform_id(platform),
		C.cl_platform_info(paramName),
		0, nil, (*C.size_t)(&size)))
	if errCode != Success {
		return "", OpenCLError{Code: errCode}
	}

	if size == 0 {
		return "", nil
	}

	buffer := make([]byte, size)
	errCode = Int(C.clGetPlatformInfo(
		C.cl_platform_id(platform),
		C.cl_platform_info(paramName),
		C.size_t(size), unsafe.Pointer(&buffer[0]), nil))
	if errCode != Success {
		return "", OpenCLError{Code: errCode}
	}

	return string(buffer[:size-1]), nil // 去掉末尾的null字符
}

func GetPlatformDetails(platform PlatformID) (*PlatformInfo, error) {
	info := &PlatformInfo{ID: platform}

	// 定义要获取的属性名和对应的结构字段
	type field struct {
		name   UInt
		assign func(value string)
	}

	fields := []field{
		{PlatformName, func(v string) { info.Name = v }},
		{PlatformVendor, func(v string) { info.Vendor = v }},
		{PlatformVersion, func(v string) { info.Version = v }},
		{PlatformProfile, func(v string) { info.Profile = v }},
		{PlatformExtensions, func(v string) { info.Extensions = v }},
	}

	for _, f := range fields {
		val, err := GetPlatformInfo(platform, f.name)
		if err != nil {
			return nil, err
		}
		f.assign(val)
	}

	return info, nil
}
