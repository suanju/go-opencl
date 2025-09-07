package main

import (
	"fmt"
	"math"
	"time"
	"unsafe"

	"github.com/suanju/go-opencl/cl"
)

func main() {
	fmt.Println("=== OpenCL GPU 矩阵乘法性能测试 ===")

	// 初始化OpenCL
	platform, devices, err := initOpenCL()
	if err != nil {
		fmt.Printf("OpenCL初始化失败: %v\n", err)
		return
	}

	ctx, err := createContext(platform, devices)
	if err != nil {
		fmt.Printf("创建上下文失败: %v\n", err)
		return
	}
	defer cl.ReleaseContext(ctx)

	device := devices[0]
	fmt.Printf("使用设备: %s\n", getDeviceName(device))

	// 测试不同矩阵大小
	sizes := []int{256, 512, 1024, 2048}

	for _, size := range sizes {
		fmt.Printf("\n--- 矩阵大小: %dx%d ---\n", size, size)

		// CPU计算时间
		cpuTime := benchmarkCPU(size)
		fmt.Printf("CPU计算时间: %.3f秒\n", cpuTime)

		// GPU计算时间
		gpuTime, err := benchmarkGPU(ctx, device, size)
		if err != nil {
			fmt.Printf("GPU计算失败: %v\n", err)
			continue
		}
		fmt.Printf("GPU计算时间: %.3f秒\n", gpuTime)
		fmt.Printf("加速比: %.2fx\n", cpuTime/gpuTime)
	}
}

func initOpenCL() (cl.PlatformID, []cl.DeviceID, error) {
	platforms, err := cl.GetPlatformIDs()
	if err != nil {
		return nil, nil, err
	}
	if len(platforms) == 0 {
		return nil, nil, fmt.Errorf("未找到OpenCL平台")
	}

	// 选择第一个有GPU设备的平台
	for _, p := range platforms {
		devices, err := cl.GetDeviceIDs(p, cl.DeviceTypeGPU)
		if err == nil && len(devices) > 0 {
			return p, devices, nil
		}
	}

	// 如果没有GPU，使用CPU
	for _, p := range platforms {
		devices, err := cl.GetDeviceIDs(p, cl.DeviceTypeCPU)
		if err == nil && len(devices) > 0 {
			return p, devices, nil
		}
	}

	return nil, nil, fmt.Errorf("未找到可用的OpenCL设备")
}

func createContext(platform cl.PlatformID, devices []cl.DeviceID) (cl.Context, error) {
	props := map[cl.UInt]interface{}{cl.ContextPlatform: platform}
	return cl.CreateContext(platform, devices, props)
}

func getDeviceName(device cl.DeviceID) string {
	if info, err := cl.GetDeviceDetails(device); err == nil {
		return info.Name
	}
	return "未知设备"
}

func benchmarkCPU(size int) float64 {
	// 创建测试矩阵
	a := make([][]float32, size)
	b := make([][]float32, size)
	c := make([][]float32, size)

	for i := 0; i < size; i++ {
		a[i] = make([]float32, size)
		b[i] = make([]float32, size)
		c[i] = make([]float32, size)
		for j := 0; j < size; j++ {
			a[i][j] = float32(i + j)
			b[i][j] = float32(i - j)
		}
	}

	start := time.Now()

	// CPU矩阵乘法
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			sum := float32(0)
			for k := 0; k < size; k++ {
				sum += a[i][k] * b[k][j]
			}
			c[i][j] = sum
		}
	}

	return time.Since(start).Seconds()
}

func benchmarkGPU(ctx cl.Context, device cl.DeviceID, size int) (float64, error) {
	queue, err := cl.CreateCommandQueue(ctx, device, 0)
	if err != nil {
		return 0, err
	}
	defer cl.ReleaseCommandQueue(queue)

	// 创建一维数组
	a := make([]float32, size*size)
	b := make([]float32, size*size)
	c := make([]float32, size*size)

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			a[i*size+j] = float32(i + j)
			b[i*size+j] = float32(i - j)
		}
	}

	bufSize := cl.Size(size * size * 4)

	// 创建缓冲区
	bufA, err := cl.CreateBuffer(ctx, cl.MemReadOnly, bufSize, nil)
	if err != nil {
		return 0, err
	}
	defer cl.ReleaseMemObject(bufA)

	bufB, err := cl.CreateBuffer(ctx, cl.MemReadOnly, bufSize, nil)
	if err != nil {
		return 0, err
	}
	defer cl.ReleaseMemObject(bufB)

	bufC, err := cl.CreateBuffer(ctx, cl.MemWriteOnly, bufSize, nil)
	if err != nil {
		return 0, err
	}
	defer cl.ReleaseMemObject(bufC)

	// OpenCL程序源码
	src := `
__kernel void matrix_multiply(__global const float* a, __global const float* b, __global float* c, int size) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row >= size || col >= size) return;
    
    float sum = 0.0f;
    for (int k = 0; k < size; k++) {
        sum += a[row * size + k] * b[k * size + col];
    }
    c[row * size + col] = sum;
}
`

	// 创建程序
	prog, err := cl.CreateProgramWithSource(ctx, 1, []string{src}, nil)
	if err != nil {
		return 0, err
	}
	defer cl.ReleaseProgram(prog)

	// 构建程序
	if err := cl.BuildProgram(prog, []cl.DeviceID{device}, "", nil, nil); err != nil {
		return 0, err
	}

	// 创建内核
	kernel, err := cl.CreateKernel(prog, "matrix_multiply")
	if err != nil {
		return 0, err
	}
	defer cl.ReleaseKernel(kernel)

	// 写入数据
	if _, err := cl.EnqueueWriteBuffer(queue, bufA, cl.Bool(1), 0, bufSize, unsafe.Pointer(&a[0]), nil); err != nil {
		return 0, err
	}
	if _, err := cl.EnqueueWriteBuffer(queue, bufB, cl.Bool(1), 0, bufSize, unsafe.Pointer(&b[0]), nil); err != nil {
		return 0, err
	}
	cl.Finish(queue)

	// 设置内核参数
	cl.SetKernelArg(kernel, 0, cl.Size(unsafe.Sizeof(bufA)), unsafe.Pointer(&bufA))
	cl.SetKernelArg(kernel, 1, cl.Size(unsafe.Sizeof(bufB)), unsafe.Pointer(&bufB))
	cl.SetKernelArg(kernel, 2, cl.Size(unsafe.Sizeof(bufC)), unsafe.Pointer(&bufC))
	cl.SetKernelArg(kernel, 3, cl.Size(4), unsafe.Pointer(&size))

	// 执行内核
	start := time.Now()
	if err := cl.EnqueueNDRangeKernel(queue, kernel, cl.UInt(2), nil, []cl.Size{cl.Size(size), cl.Size(size)}, nil, nil, nil); err != nil {
		return 0, err
	}
	cl.Finish(queue)
	gpuTime := time.Since(start).Seconds()

	// 读取结果
	if _, err := cl.EnqueueReadBuffer(queue, bufC, cl.Bool(1), 0, bufSize, unsafe.Pointer(&c[0]), nil); err != nil {
		return 0, err
	}
	cl.Finish(queue)

	// 验证结果（检查几个随机位置）
	for i := 0; i < 3; i++ {
		row := int(math.Floor(float64(len(c)) * float64(i+1) / 4))
		col := int(math.Floor(float64(len(c)) * float64(i+2) / 4))
		if row >= size || col >= size {
			continue
		}

		expected := float32(0)
		for k := 0; k < size; k++ {
			expected += a[row*size+k] * b[k*size+col]
		}

		if math.Abs(float64(c[row*size+col]-expected)) > 1e-5 {
			fmt.Printf("警告: 结果验证失败 at [%d,%d], got %f, expected %f\n", row, col, c[row*size+col], expected)
		}
	}

	return gpuTime, nil
}
