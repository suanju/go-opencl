package main

import (
	"bufio"
	"fmt"
	"math"
	"opencl/cl"
	"os"
	"unsafe"
)

func main() {
	fmt.Println("=== OpenCL 示例（简洁版） ===")

	platform, devices, err := listPlatformsAndDevices()
	if err != nil {
		fmt.Println("初始化失败:", err)
		return
	}

	ctx, err := createContext(platform, devices)
	if err != nil {
		fmt.Println("创建上下文失败:", err)
		return
	}
	// 主上下文最后释放
	defer func() {
		_ = cl.ReleaseContext(ctx)
		fmt.Println("✓ 主上下文已释放")
	}()

	// buffer demo
	if err := bufferDemo(ctx, devices[0]); err != nil {
		fmt.Println("bufferDemo 错误:", err)
	}

	// program & kernel demo
	if err := programKernelDemo(ctx, devices[0]); err != nil {
		fmt.Println("programKernelDemo 错误:", err)
	}

	// task demo (示例使用 NDRange 替代 clEnqueueTask)
	if err := taskDemo(ctx, devices[0]); err != nil {
		fmt.Println("taskDemo 错误:", err)
	}

	// debug / error demo
	if err := debugDemo(ctx, devices[0]); err != nil {
		fmt.Println("debugDemo 错误:", err)
	}

	fmt.Println("\n=== 示例完成 ===")
	fmt.Println("按回车键退出...")
	bufio.NewReader(os.Stdin).ReadBytes('\n')
}

// listPlatformsAndDevices: 返回第一个有设备的平台和该平台设备列表
func listPlatformsAndDevices() (cl.PlatformID, []cl.DeviceID, error) {
	platforms, err := cl.GetPlatformIDs()
	if err != nil {
		return nil, nil, fmt.Errorf("GetPlatformIDs: %w", err)
	}
	if len(platforms) == 0 {
		return nil, nil, fmt.Errorf("no OpenCL platforms found")
	}

	var selP cl.PlatformID
	var selDevices []cl.DeviceID

	for i, p := range platforms {
		info, _ := cl.GetPlatformDetails(p)
		fmt.Printf("--- 平台 %d ---\n名称: %s\n厂商: %s\n版本: %s\n", i+1, info.Name, info.Vendor, info.Version)

		devs, err := cl.GetDeviceIDs(p, cl.DeviceTypeAll)
		if err != nil {
			fmt.Printf("  获取设备失败: %v\n", err)
			continue
		}
		for j, d := range devs {
			di, _ := cl.GetDeviceDetails(d)
			fmt.Printf("  设备 %d: %s (%s)\n", j+1, di.Name, di.Version)
		}
		if selP == nil && len(devs) > 0 {
			selP = p
			selDevices = devs
		}
	}
	if selP == nil {
		return nil, nil, fmt.Errorf("no usable device found")
	}
	return selP, selDevices, nil
}

// createContext: 根据平台与设备创建 context
func createContext(platform cl.PlatformID, devices []cl.DeviceID) (cl.Context, error) {
	props := map[cl.UInt]interface{}{cl.ContextPlatform: platform}
	ctx, err := cl.CreateContext(platform, devices, props)
	if err != nil {
		return nil, err
	}
	fmt.Println("✓ 上下文创建成功")
	return ctx, nil
}

// createQueueAuto: 尝试使用带属性的新 API，失败则退回旧 API
func createQueueAuto(ctx cl.Context, device cl.DeviceID) (cl.CommandQueue, error) {
	// 先尝试新 API（启用 profiling 作为示例属性）
	props := map[cl.UInt]interface{}{cl.QueueProperties: cl.QueueProfilingEnable}
	q, err := cl.CreateCommandQueueWithProperties(ctx, device, props)
	if err == nil {
		return q, nil
	}
	// 回退到旧 API
	q2, err2 := cl.CreateCommandQueue(ctx, device, 0)
	if err2 != nil {
		// 返回第一个错误（或合并更详细信息）
		return nil, fmt.Errorf("CreateCommandQueueWithProperties failed: %v; fallback CreateCommandQueue failed: %v", err, err2)
	}
	return q2, nil
}

// bufferDemo: 简短的缓冲区写读验证
func bufferDemo(ctx cl.Context, device cl.DeviceID) error {
	fmt.Println("\n== bufferDemo ==")
	queue, err := createQueueAuto(ctx, device)
	if err != nil {
		return err
	}
	defer cl.ReleaseCommandQueue(queue)

	n := 256
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i)
	}
	size := cl.Size(n * 4)

	buf, err := cl.CreateBuffer(ctx, cl.MemReadWrite, size, nil)
	if err != nil {
		return fmt.Errorf("CreateBuffer: %w", err)
	}
	defer cl.ReleaseMemObject(buf)
	fmt.Println("  ✓ 缓冲区创建")

	if _, err := cl.EnqueueWriteBuffer(queue, buf, cl.Bool(1), 0, size, unsafe.Pointer(&data[0]), nil); err != nil {
		return fmt.Errorf("EnqueueWriteBuffer: %w", err)
	}
	cl.Finish(queue)

	out := make([]float32, n)
	if _, err := cl.EnqueueReadBuffer(queue, buf, cl.Bool(1), 0, size, unsafe.Pointer(&out[0]), nil); err != nil {
		return fmt.Errorf("EnqueueReadBuffer: %w", err)
	}
	cl.Finish(queue)

	for i := 0; i < n; i++ {
		if data[i] != out[i] {
			return fmt.Errorf("data mismatch at %d: got %v want %v", i, out[i], data[i])
		}
	}
	fmt.Println("  ✓ 缓冲区读写验证通过")
	return nil
}

// programKernelDemo: 创建 program、kernel 并显示一些 kernel 信息（不执行重负载）
func programKernelDemo(ctx cl.Context, device cl.DeviceID) error {
	fmt.Println("\n== programKernelDemo ==")
	queue, err := createQueueAuto(ctx, device)
	if err != nil {
		return err
	}
	defer cl.ReleaseCommandQueue(queue)

	src := `
__kernel void vector_add(__global const float* a, __global const float* b, __global float* c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}
`
	prog, err := cl.CreateProgramWithSource(ctx, 1, []string{src}, nil)
	if err != nil {
		return fmt.Errorf("CreateProgramWithSource: %w", err)
	}
	defer cl.ReleaseProgram(prog)

	if err := cl.BuildProgram(prog, []cl.DeviceID{device}, "", nil, nil); err != nil {
		// 尝试获取构建日志
		if log, le := cl.GetProgramBuildLog(prog, device); le == nil && log != "" {
			return fmt.Errorf("BuildProgram failed: %v\nBuildLog:\n%s", err, log)
		}
		return fmt.Errorf("BuildProgram: %w", err)
	}
	fmt.Println("  ✓ Program 构建成功")

	kernel, err := cl.CreateKernel(prog, "vector_add")
	if err != nil {
		return fmt.Errorf("CreateKernel: %w", err)
	}
	defer cl.ReleaseKernel(kernel)

	name, _ := cl.GetKernelFunctionName(kernel)
	numArgs, _ := cl.GetKernelNumArgs(kernel)
	wg, _ := cl.GetKernelWorkGroupSize(kernel, device)
	localMem, _ := cl.GetKernelLocalMemSize(kernel, device)

	fmt.Printf("  kernel: %s, args=%d, wg=%d, localMem=%d\n", name, numArgs, wg, localMem)
	return nil
}

// taskDemo: 提交一个小的 NDRange 内核并验证结果（使用 vector_add）
func taskDemo(ctx cl.Context, device cl.DeviceID) error {
	fmt.Println("\n== taskDemo ==")
	queue, err := createQueueAuto(ctx, device)
	if err != nil {
		return err
	}
	defer cl.ReleaseCommandQueue(queue)

	// small test
	n := 1024
	a := make([]float32, n)
	b := make([]float32, n)
	c := make([]float32, n)
	for i := 0; i < n; i++ {
		a[i] = float32(i)
		b[i] = float32(i * 2)
	}
	size := cl.Size(n * 4)

	// create buffers
	bufA, err := cl.CreateBuffer(ctx, cl.MemReadOnly, size, nil)
	if err != nil {
		return fmt.Errorf("CreateBuffer A: %w", err)
	}
	defer cl.ReleaseMemObject(bufA)

	bufB, err := cl.CreateBuffer(ctx, cl.MemReadOnly, size, nil)
	if err != nil {
		return fmt.Errorf("CreateBuffer B: %w", err)
	}
	defer cl.ReleaseMemObject(bufB)

	bufC, err := cl.CreateBuffer(ctx, cl.MemWriteOnly, size, nil)
	if err != nil {
		return fmt.Errorf("CreateBuffer C: %w", err)
	}
	defer cl.ReleaseMemObject(bufC)

	// simple program
	src := `
__kernel void vector_add(__global const float* a, __global const float* b, __global float* c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}
`
	prog, err := cl.CreateProgramWithSource(ctx, 1, []string{src}, nil)
	if err != nil {
		return fmt.Errorf("CreateProgram: %w", err)
	}
	defer cl.ReleaseProgram(prog)

	if err := cl.BuildProgram(prog, []cl.DeviceID{device}, "", nil, nil); err != nil {
		return fmt.Errorf("BuildProgram: %w", err)
	}

	kernel, err := cl.CreateKernel(prog, "vector_add")
	if err != nil {
		return fmt.Errorf("CreateKernel: %w", err)
	}
	defer cl.ReleaseKernel(kernel)

	// write host->device (synchronous)
	if _, err := cl.EnqueueWriteBuffer(queue, bufA, cl.Bool(1), 0, size, unsafe.Pointer(&a[0]), nil); err != nil {
		return fmt.Errorf("EnqueueWriteBuffer A: %w", err)
	}
	if _, err := cl.EnqueueWriteBuffer(queue, bufB, cl.Bool(1), 0, size, unsafe.Pointer(&b[0]), nil); err != nil {
		return fmt.Errorf("EnqueueWriteBuffer B: %w", err)
	}
	cl.Finish(queue)

	// set args (pass cl_mem by pointer as package expects)
	cl.SetKernelArg(kernel, 0, cl.Size(unsafe.Sizeof(bufA)), unsafe.Pointer(&bufA))
	cl.SetKernelArg(kernel, 1, cl.Size(unsafe.Sizeof(bufB)), unsafe.Pointer(&bufB))
	cl.SetKernelArg(kernel, 2, cl.Size(unsafe.Sizeof(bufC)), unsafe.Pointer(&bufC))

	// enqueue kernel
	if err := cl.EnqueueNDRangeKernel(queue, kernel, cl.UInt(1), nil, []cl.Size{cl.Size(n)}, nil, nil, nil); err != nil {
		return fmt.Errorf("EnqueueNDRangeKernel: %w", err)
	}
	if err := cl.Finish(queue); err != nil {
		return fmt.Errorf("finish: %w", err)
	}

	// read back
	if _, err := cl.EnqueueReadBuffer(queue, bufC, cl.Bool(1), 0, size, unsafe.Pointer(&c[0]), nil); err != nil {
		return fmt.Errorf("EnqueueReadBuffer: %w", err)
	}
	cl.Finish(queue)

	// verify (浮点比较使用 eps)
	eps := 1e-6
	for i := 0; i < n; i++ {
		expected := float64(a[i] + b[i])
		if math.Abs(float64(c[i])-expected) > eps {
			return fmt.Errorf("result mismatch at %d: got %v want %v", i, c[i], expected)
		}
	}
	fmt.Println("  ✓ taskDemo: kernel result verified")
	return nil
}

// debugDemo: 演示错误日志/构建错误捕获/Validate 等（简洁）
func debugDemo(ctx cl.Context, device cl.DeviceID) error {
	fmt.Println("\n== debugDemo ==")
	cl.EnableDebugLogging()
	defer cl.DisableDebugLogging()

	// 构造有语法错误的程序以示范获取构建日志
	invalid := `
__kernel void bad(__global const float* a) {
    int gid = get_global_id(0)
    // missing semicolon above
    a[gid] = a[gid];
}
`
	prog, err := cl.CreateProgramWithSource(ctx, 1, []string{invalid}, nil)
	if err != nil {
		return fmt.Errorf("CreateProgramWithSource: %w", err)
	}
	defer cl.ReleaseProgram(prog)

	if err := cl.BuildProgram(prog, []cl.DeviceID{device}, "", nil, nil); err == nil {
		// unexpectedly succeeded
		return fmt.Errorf("expected build to fail but it succeeded")
	} else {
		// try get detailed build info
		if bi, e := cl.GetDetailedBuildInfo(prog, device); e == nil {
			fmt.Printf("  build status: %s\n", cl.BuildStatusString(bi.Status))
			if bi.HasBuildErrors() {
				fmt.Println("  build log:\n", bi.Log)
			}
		}
	}

	// ValidateContext / ValidateDevice 示例（轻量）
	if err := cl.ValidateContext(ctx); err != nil {
		fmt.Println("  ValidateContext:", err)
	} else {
		fmt.Println("  ✓ ValidateContext ok")
	}
	if err := cl.ValidateDevice(device); err != nil {
		fmt.Println("  ValidateDevice:", err)
	} else {
		fmt.Println("  ✓ ValidateDevice ok")
	}

	// 打印并清空日志摘要
	fmt.Println("  ErrorSummary:\n", cl.GetErrorSummary())
	cl.ClearErrorLog()
	return nil
}
