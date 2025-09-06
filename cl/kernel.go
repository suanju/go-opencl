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

func CreateKernel(program Program, kernelName string) (Kernel, error) {
	var err C.cl_int

	kernelNamePtr := C.CString(kernelName)
	defer C.free(unsafe.Pointer(kernelNamePtr))

	kernel := C.clCreateKernel(
		C.cl_program(program),
		kernelNamePtr,
		&err,
	)

	if err != C.CL_SUCCESS {
		return Kernel(nil), OpenCLError{Code: Int(err)}
	}

	return Kernel(kernel), nil
}
func CreateKernelsInProgram(program Program) ([]Kernel, error) {
	var err C.cl_int
	var numKernels C.cl_uint

	// 第一次调用，获取内核数量
	err = C.clCreateKernelsInProgram(
		C.cl_program(program),
		0,
		nil,
		&numKernels,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	if numKernels == 0 {
		return []Kernel{}, nil
	}

	// 分配内核数组
	kernels := make([]C.cl_kernel, numKernels)

	// 第二次调用，创建所有内核
	err = C.clCreateKernelsInProgram(
		C.cl_program(program),
		numKernels,
		&kernels[0],
		nil,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	// 转换为Go类型
	result := make([]Kernel, numKernels)
	for i, kernel := range kernels {
		result[i] = Kernel(kernel)
	}

	return result, nil
}
func ReleaseKernel(kernel Kernel) error {
	err := C.clReleaseKernel(C.cl_kernel(kernel))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}
func RetainKernel(kernel Kernel) error {
	err := C.clRetainKernel(C.cl_kernel(kernel))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}
func SetKernelArg(kernel Kernel, argIndex UInt, argSize Size, argValue unsafe.Pointer) error {
	err := C.clSetKernelArg(
		C.cl_kernel(kernel),
		C.cl_uint(argIndex),
		C.size_t(argSize),
		argValue,
	)

	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}

	return nil
}
func GetKernelInfo(kernel Kernel, paramName UInt) ([]byte, error) {
	var paramValueSizeRet C.size_t

	// 第一次调用，获取需要的大小
	err := C.clGetKernelInfo(
		C.cl_kernel(kernel),
		C.cl_kernel_info(paramName),
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
	err = C.clGetKernelInfo(
		C.cl_kernel(kernel),
		C.cl_kernel_info(paramName),
		paramValueSizeRet,
		unsafe.Pointer(&paramValue[0]),
		nil,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	return paramValue, nil
}

func GetKernelWorkGroupInfo(kernel Kernel, device DeviceID, paramName UInt) ([]byte, error) {
	var paramValueSizeRet C.size_t

	// 第一次调用，获取需要的大小
	err := C.clGetKernelWorkGroupInfo(
		C.cl_kernel(kernel),
		C.cl_device_id(device),
		C.cl_kernel_work_group_info(paramName),
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
	err = C.clGetKernelWorkGroupInfo(
		C.cl_kernel(kernel),
		C.cl_device_id(device),
		C.cl_kernel_work_group_info(paramName),
		paramValueSizeRet,
		unsafe.Pointer(&paramValue[0]),
		nil,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	return paramValue, nil
}
func GetKernelFunctionName(kernel Kernel) (string, error) {
	info, err := GetKernelInfo(kernel, C.CL_KERNEL_FUNCTION_NAME)
	if err != nil {
		return "", err
	}

	// 移除末尾的null字符
	if len(info) > 0 && info[len(info)-1] == 0 {
		info = info[:len(info)-1]
	}

	return string(info), nil
}
func GetKernelNumArgs(kernel Kernel) (UInt, error) {
	info, err := GetKernelInfo(kernel, C.CL_KERNEL_NUM_ARGS)
	if err != nil {
		return 0, err
	}

	numArgs := *(*C.cl_uint)(unsafe.Pointer(&info[0]))
	return UInt(numArgs), nil
}
func GetKernelWorkGroupSize(kernel Kernel, device DeviceID) (Size, error) {
	info, err := GetKernelWorkGroupInfo(kernel, device, C.CL_KERNEL_WORK_GROUP_SIZE)
	if err != nil {
		return 0, err
	}

	workGroupSize := *(*C.size_t)(unsafe.Pointer(&info[0]))
	return Size(workGroupSize), nil
}

func GetKernelLocalMemSize(kernel Kernel, device DeviceID) (UInt, error) {
	info, err := GetKernelWorkGroupInfo(kernel, device, C.CL_KERNEL_LOCAL_MEM_SIZE)
	if err != nil {
		return 0, err
	}

	localMemSize := *(*C.cl_ulong)(unsafe.Pointer(&info[0]))
	return UInt(localMemSize), nil
}
func GetKernelPreferredWorkGroupSizeMultiple(kernel Kernel, device DeviceID) (Size, error) {
	info, err := GetKernelWorkGroupInfo(kernel, device, C.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)
	if err != nil {
		return 0, err
	}

	preferredSize := *(*C.size_t)(unsafe.Pointer(&info[0]))
	return Size(preferredSize), nil
}

func GetKernelContext(kernel Kernel) (Context, error) {
	info, err := GetKernelInfo(kernel, C.CL_KERNEL_CONTEXT)
	if err != nil {
		return Context(nil), err
	}

	context := *(*C.cl_context)(unsafe.Pointer(&info[0]))
	return Context(context), nil
}

func GetKernelProgram(kernel Kernel) (Program, error) {
	info, err := GetKernelInfo(kernel, C.CL_KERNEL_PROGRAM)
	if err != nil {
		return Program(nil), err
	}

	program := *(*C.cl_program)(unsafe.Pointer(&info[0]))
	return Program(program), nil
}
