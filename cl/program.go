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

func CreateProgramWithSource(context Context, count UInt, strings []string, lengths []Size) (Program, error) {
	var err C.cl_int

	// 准备C字符串数组
	cStrings := make([]*C.char, count)
	cLengths := make([]C.size_t, count)

	for i := 0; i < int(count); i++ {
		cStrings[i] = C.CString(strings[i])
		if i < len(lengths) {
			cLengths[i] = C.size_t(lengths[i])
		} else {
			cLengths[i] = C.size_t(len(strings[i]))
		}
	}

	var lengthsPtr *C.size_t
	if len(lengths) > 0 {
		lengthsPtr = &cLengths[0]
	}

	program := C.clCreateProgramWithSource(
		C.cl_context(context),
		C.cl_uint(count),
		&cStrings[0],
		lengthsPtr,
		&err,
	)

	// 释放C字符串
	for i := 0; i < int(count); i++ {
		C.free(unsafe.Pointer(cStrings[i]))
	}

	if err != C.CL_SUCCESS {
		return Program(nil), OpenCLError{Code: Int(err)}
	}

	return Program(program), nil
}

func CreateProgramWithBinary(context Context, devices []DeviceID, lengths []Size, binaries [][]byte, binaryStatus []Int) (Program, error) {
	var err C.cl_int

	deviceCount := C.cl_uint(len(devices))
	if deviceCount == 0 {
		return Program(nil), OpenCLError{Code: Int(C.CL_INVALID_VALUE)}
	}

	// 准备设备数组
	deviceArray := make([]C.cl_device_id, deviceCount)
	for i, device := range devices {
		deviceArray[i] = C.cl_device_id(device)
	}

	// 准备长度数组
	lengthArray := make([]C.size_t, deviceCount)
	for i, length := range lengths {
		lengthArray[i] = C.size_t(length)
	}

	// 准备二进制指针数组
	binaryPointers := make([]unsafe.Pointer, deviceCount)
	for i, binary := range binaries {
		if len(binary) > 0 {
			binaryPointers[i] = unsafe.Pointer(&binary[0])
		} else {
			binaryPointers[i] = nil
		}
	}

	// 准备状态数组
	var statusArray []C.cl_int
	var statusPtr *C.cl_int
	if len(binaryStatus) > 0 {
		statusArray = make([]C.cl_int, deviceCount)
		for i, status := range binaryStatus {
			statusArray[i] = C.cl_int(status)
		}
		statusPtr = &statusArray[0]
	}

	program := C.clCreateProgramWithBinary(
		C.cl_context(context),
		deviceCount,
		&deviceArray[0],
		&lengthArray[0],
		(**C.uchar)(unsafe.Pointer(&binaryPointers[0])),
		statusPtr,
		&err,
	)

	if err != C.CL_SUCCESS {
		return Program(nil), OpenCLError{Code: Int(err)}
	}

	return Program(program), nil
}

func BuildProgram(program Program, devices []DeviceID, options string, notify unsafe.Pointer, userData unsafe.Pointer) error {
	var err C.cl_int

	var deviceCount C.cl_uint
	var deviceArray *C.cl_device_id
	if len(devices) > 0 {
		deviceCount = C.cl_uint(len(devices))
		deviceArray = (*C.cl_device_id)(unsafe.Pointer(&devices[0]))
	}

	var optionsPtr *C.char
	if options != "" {
		optionsPtr = C.CString(options)
		defer C.free(unsafe.Pointer(optionsPtr))
	}

	err = C.clBuildProgram(
		C.cl_program(program),
		deviceCount,
		deviceArray,
		optionsPtr,
		nil, // 通知回调函数
		userData,
	)

	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}

	return nil
}
func ReleaseProgram(program Program) error {
	err := C.clReleaseProgram(C.cl_program(program))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

func RetainProgram(program Program) error {
	err := C.clRetainProgram(C.cl_program(program))
	if err != C.CL_SUCCESS {
		return OpenCLError{Code: Int(err)}
	}
	return nil
}

func GetProgramInfo(program Program, paramName UInt) ([]byte, error) {
	var paramValueSizeRet C.size_t

	// 第一次调用，获取需要的大小
	err := C.clGetProgramInfo(
		C.cl_program(program),
		C.cl_program_info(paramName),
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
	err = C.clGetProgramInfo(
		C.cl_program(program),
		C.cl_program_info(paramName),
		paramValueSizeRet,
		unsafe.Pointer(&paramValue[0]),
		nil,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	return paramValue, nil
}

func GetProgramBuildInfo(program Program, device DeviceID, paramName UInt) ([]byte, error) {
	var paramValueSizeRet C.size_t

	// 第一次调用，获取需要的大小
	err := C.clGetProgramBuildInfo(
		C.cl_program(program),
		C.cl_device_id(device),
		C.cl_program_build_info(paramName),
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
	err = C.clGetProgramBuildInfo(
		C.cl_program(program),
		C.cl_device_id(device),
		C.cl_program_build_info(paramName),
		paramValueSizeRet,
		unsafe.Pointer(&paramValue[0]),
		nil,
	)
	if err != C.CL_SUCCESS {
		return nil, OpenCLError{Code: Int(err)}
	}

	return paramValue, nil
}

func GetProgramSource(program Program) (string, error) {
	info, err := GetProgramInfo(program, C.CL_PROGRAM_SOURCE)
	if err != nil {
		return "", err
	}

	// 移除末尾的null字符
	if len(info) > 0 && info[len(info)-1] == 0 {
		info = info[:len(info)-1]
	}

	return string(info), nil
}

func GetProgramBuildStatus(program Program, device DeviceID) (UInt, error) {
	info, err := GetProgramBuildInfo(program, device, C.CL_PROGRAM_BUILD_STATUS)
	if err != nil {
		return 0, err
	}

	status := *(*C.cl_build_status)(unsafe.Pointer(&info[0]))
	return UInt(status), nil
}

func GetProgramBuildLog(program Program, device DeviceID) (string, error) {
	info, err := GetProgramBuildInfo(program, device, C.CL_PROGRAM_BUILD_LOG)
	if err != nil {
		return "", err
	}

	// 移除末尾的null字符
	if len(info) > 0 && info[len(info)-1] == 0 {
		info = info[:len(info)-1]
	}

	return string(info), nil
}

func GetProgramNumKernels(program Program) (UInt, error) {
	info, err := GetProgramInfo(program, C.CL_PROGRAM_NUM_KERNELS)
	if err != nil {
		return 0, err
	}

	numKernels := *(*C.cl_uint)(unsafe.Pointer(&info[0]))
	return UInt(numKernels), nil
}
func GetProgramKernelNames(program Program) (string, error) {
	info, err := GetProgramInfo(program, C.CL_PROGRAM_KERNEL_NAMES)
	if err != nil {
		return "", err
	}

	// 移除末尾的null字符
	if len(info) > 0 && info[len(info)-1] == 0 {
		info = info[:len(info)-1]
	}

	return string(info), nil
}

// GetProgramBuildOptions 获取程序构建选项
func GetProgramBuildOptions(program Program, device DeviceID) (string, error) {
	info, err := GetProgramBuildInfo(program, device, C.CL_PROGRAM_BUILD_OPTIONS)
	if err != nil {
		return "", err
	}

	// 移除末尾的null字符
	if len(info) > 0 && info[len(info)-1] == 0 {
		info = info[:len(info)-1]
	}

	return string(info), nil
}

// GetProgramBuildBinaryType 获取程序构建二进制类型
func GetProgramBuildBinaryType(program Program, device DeviceID) (UInt, error) {
	info, err := GetProgramBuildInfo(program, device, C.CL_PROGRAM_BINARY_TYPE)
	if err != nil {
		return 0, err
	}

	binaryType := *(*C.cl_program_binary_type)(unsafe.Pointer(&info[0]))
	return UInt(binaryType), nil
}

// GetProgramBuildGlobalVariableTotalSize 获取程序构建全局变量总大小
func GetProgramBuildGlobalVariableTotalSize(program Program, device DeviceID) (Size, error) {
	info, err := GetProgramBuildInfo(program, device, C.CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE)
	if err != nil {
		return 0, err
	}

	totalSize := *(*C.size_t)(unsafe.Pointer(&info[0]))
	return Size(totalSize), nil
}

// BuildStatusString 将构建状态转换为字符串
func BuildStatusString(status UInt) string {
	switch status {
	case 0: // CL_BUILD_SUCCESS
		return "Build Success"
	case 1: // CL_BUILD_NONE
		return "Build None"
	case 2: // CL_BUILD_ERROR
		return "Build Error"
	case 3: // CL_BUILD_IN_PROGRESS
		return "Build In Progress"
	default:
		return fmt.Sprintf("Unknown build status (code: %d)", status)
	}
}

// BinaryTypeString 将二进制类型转换为字符串
func BinaryTypeString(binaryType UInt) string {
	switch binaryType {
	case 0: // CL_PROGRAM_BINARY_TYPE_NONE
		return "None"
	case 1: // CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT
		return "Compiled Object"
	case 2: // CL_PROGRAM_BINARY_TYPE_LIBRARY
		return "Library"
	case 3: // CL_PROGRAM_BINARY_TYPE_EXECUTABLE
		return "Executable"
	default:
		return fmt.Sprintf("Unknown binary type (code: %d)", binaryType)
	}
}

// GetDetailedBuildInfo 获取详细的构建信息
func GetDetailedBuildInfo(program Program, device DeviceID) (*BuildInfo, error) {
	status, err := GetProgramBuildStatus(program, device)
	if err != nil {
		return nil, err
	}

	log, err := GetProgramBuildLog(program, device)
	if err != nil {
		return nil, err
	}

	options, err := GetProgramBuildOptions(program, device)
	if err != nil {
		return nil, err
	}

	binaryType, err := GetProgramBuildBinaryType(program, device)
	if err != nil {
		return nil, err
	}

	return &BuildInfo{
		Status:     status,
		Log:        log,
		Options:    options,
		BinaryType: binaryType,
	}, nil
}

// BuildInfo 构建信息结构
type BuildInfo struct {
	Status     UInt   // 构建状态
	Log        string // 构建日志
	Options    string // 构建选项
	BinaryType UInt   // 二进制类型
}

// IsBuildSuccessful 检查构建是否成功
func (bi *BuildInfo) IsBuildSuccessful() bool {
	return bi.Status == 0 // CL_BUILD_SUCCESS
}

// HasBuildErrors 检查是否有构建错误
func (bi *BuildInfo) HasBuildErrors() bool {
	return bi.Status == 2 // CL_BUILD_ERROR
}

// IsBuilding 检查是否正在构建
func (bi *BuildInfo) IsBuilding() bool {
	return bi.Status == 3 // CL_BUILD_IN_PROGRESS
}

// String 返回构建信息的字符串表示
func (bi *BuildInfo) String() string {
	return fmt.Sprintf("BuildInfo{Status: %s, BinaryType: %s, HasLog: %t, HasOptions: %t}",
		BuildStatusString(bi.Status),
		BinaryTypeString(bi.BinaryType),
		len(bi.Log) > 0,
		len(bi.Options) > 0)
}
