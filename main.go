package main

/*
#cgo LDFLAGS: -lOpenCL
#include <CL/cl.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func main() {
	var numPlatforms C.cl_uint
	status := C.clGetPlatformIDs(0, nil, &numPlatforms)
	if status != C.CL_SUCCESS {
		fmt.Println("clGetPlatformIDs failed:", status)
		return
	}
	fmt.Printf("找到 %d 个 OpenCL 平台:\n", numPlatforms)

	platforms := make([]C.cl_platform_id, numPlatforms)
	C.clGetPlatformIDs(numPlatforms, &platforms[0], nil)

	for i := C.cl_uint(0); i < numPlatforms; i++ {
		var name [128]C.char
		C.clGetPlatformInfo(platforms[i], C.CL_PLATFORM_NAME, 128, unsafe.Pointer(&name[0]), nil)
		fmt.Printf("平台 %d: %s\n", i, C.GoString(&name[0]))

		// 获取设备
		var numDevices C.cl_uint
		C.clGetDeviceIDs(platforms[i], C.CL_DEVICE_TYPE_ALL, 0, nil, &numDevices)
		devices := make([]C.cl_device_id, numDevices)
		C.clGetDeviceIDs(platforms[i], C.CL_DEVICE_TYPE_ALL, numDevices, &devices[0], nil)

		for j := C.cl_uint(0); j < numDevices; j++ {
			var dname [128]C.char
			C.clGetDeviceInfo(devices[j], C.CL_DEVICE_NAME, 128, unsafe.Pointer(&dname[0]), nil)
			fmt.Printf("  设备 %d: %s\n", j, C.GoString(&dname[0]))
		}
	}
}
