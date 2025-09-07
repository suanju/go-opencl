# go-opencl

一个功能完整的Go语言OpenCL绑定库，提供高性能GPU计算能力。支持OpenCL 3.0+标准

## ✨ 特性

- 🚀 **完整的OpenCL 3.0+支持** - 支持最新的OpenCL标准
- 🛠️ **丰富的API** - 涵盖平台、设备、上下文、程序、内核等完整功能
- 📊 **调试支持** - 内置错误处理和调试工具
- 🖼️ **图像处理** - 支持OpenCL图像和缓冲区操作
- ⚡ **高性能** - 直接C绑定，最小化性能开销
- 📚 **详细示例** - 包含矩阵乘法、向量运算等实用示例

## 🚀 快速开始

### 安装

```bash
go get github.com/suanju/go-opencl
```

### 环境要求

#### Windows

- NVIDIA GPU驱动（支持CUDA）
- AMD GPU驱动（支持OpenCL）
- Intel GPU驱动（支持OpenCL）

#### Linux

- 相应的GPU驱动
- OpenCL运行时库

#### macOS

- 系统自带OpenCL支持

### 基础示例

```go
package main

import (
    "fmt"
    "unsafe"

    "github.com/suanju/go-opencl/cl"
)

func main() {
    // 获取平台和设备
    platforms, err := cl.GetPlatformIDs()
    if err != nil {
        panic(err)
    }

    devices, err := cl.GetDeviceIDs(platforms[0], cl.DeviceTypeGPU)
    if err != nil {
        panic(err)
    }

    // 创建上下文
    ctx, err := cl.CreateContext(platforms[0], devices, nil)
    if err != nil {
        panic(err)
    }
    defer cl.ReleaseContext(ctx)

    // 创建命令队列
    queue, err := cl.CreateCommandQueue(ctx, devices[0], 0)
    if err != nil {
        panic(err)
    }
    defer cl.ReleaseCommandQueue(queue)

    // 输入数据
    data := []float32{1, 2, 3, 4, 5}

    // 创建缓冲区（不直接传 host pointer）
    buffer, err := cl.CreateBuffer(ctx, cl.MemReadWrite, cl.Size(len(data)*4), nil)
    if err != nil {
        panic(err)
    }
    defer cl.ReleaseMemObject(buffer)

    // 把数据写到 GPU buffer
    _, err = cl.EnqueueWriteBuffer(queue, buffer, cl.Bool(1), 0,
        cl.Size(len(data)*4), unsafe.Pointer(&data[0]), nil)
    if err != nil {
        panic(err)
    }

    // 创建程序
    source := `
    __kernel void square(__global float* data) {
        int gid = get_global_id(0);
        data[gid] = data[gid] * data[gid];
    }`

    program, err := cl.CreateProgramWithSource(ctx, 1, []string{source}, nil)
    if err != nil {
        panic(err)
    }
    defer cl.ReleaseProgram(program)

    // 构建程序
    err = cl.BuildProgram(program, devices, "", nil, nil)
    if err != nil {
        // 打印编译日志
        log, _ := cl.GetProgramBuildInfo(program, devices[0], 0x1183)
        fmt.Println("Build Log:", log)
        panic(err)
    }

    // 创建内核
    kernel, err := cl.CreateKernel(program, "square")
    if err != nil {
        panic(err)
    }
    defer cl.ReleaseKernel(kernel)

    // 设置内核参数
    err = cl.SetKernelArg(kernel, 0, cl.Size(unsafe.Sizeof(buffer)), unsafe.Pointer(&buffer))
    if err != nil {
        panic(err)
    }

    // 执行内核
    err = cl.EnqueueNDRangeKernel(queue, kernel, 1, nil, []cl.Size{cl.Size(len(data))}, nil, nil, nil)
    if err != nil {
        panic(err)
    }

    // 确保执行完成
    cl.Finish(queue)

    // 读取结果
    result := make([]float32, len(data))
    _, err = cl.EnqueueReadBuffer(queue, buffer, cl.Bool(1), 0,
        cl.Size(len(result)*4), unsafe.Pointer(&result[0]), nil)
    if err != nil {
        panic(err)
    }
    
    fmt.Println("结果:", result) // 预期: [1 4 9 16 25]
}
```

## 📖 API 文档

### 平台和设备管理

```go
// 获取所有OpenCL平台
platforms, err := cl.GetPlatformIDs()

// 获取平台详细信息
info, err := cl.GetPlatformDetails(platform)

// 获取设备列表
devices, err := cl.GetDeviceIDs(platform, cl.DeviceTypeGPU)

// 获取设备详细信息
deviceInfo, err := cl.GetDeviceDetails(device)
```

### 上下文和命令队列

```go
// 创建上下文
ctx, err := cl.CreateContext(platform, devices, properties)

// 创建命令队列
queue, err := cl.CreateCommandQueue(ctx, device, properties)

// 创建带属性的命令队列
props := map[cl.UInt]interface{}{
    cl.QueueProperties: cl.QueueProfilingEnable,
}
queue, err := cl.CreateCommandQueueWithProperties(ctx, device, props)
```

### 内存管理

```go
// 创建缓冲区
buffer, err := cl.CreateBuffer(ctx, cl.MemReadWrite, size, hostPtr)

// 创建子缓冲区
subBuffer, err := cl.CreateSubBuffer(buffer, flags, createType, createInfo)

// 读写缓冲区
_, err = cl.EnqueueWriteBuffer(queue, buffer, cl.Bool(1), 0, size, ptr, nil)
_, err = cl.EnqueueReadBuffer(queue, buffer, cl.Bool(1), 0, size, ptr, nil)

// 内存映射
ptr, event, err := cl.EnqueueMapBuffer(queue, buffer, cl.Bool(1), flags, 0, size, nil)
```

### 程序构建

```go
// 从源码创建程序
program, err := cl.CreateProgramWithSource(ctx, 1, []string{source}, nil)

// 从二进制创建程序
program, err := cl.CreateProgramWithBinary(ctx, devices, binaries, lengths, nil)

// 构建程序
err = cl.BuildProgram(program, devices, options, nil, nil)

// 获取构建日志
log, err := cl.GetProgramBuildLog(program, device)
```

### 内核执行

```go
// 创建内核
kernel, err := cl.CreateKernel(program, "kernel_name")

// 设置内核参数
cl.SetKernelArg(kernel, 0, cl.Size(unsafe.Sizeof(buffer)), unsafe.Pointer(&buffer))

// 执行NDRange内核
err = cl.EnqueueNDRangeKernel(queue, kernel, workDim, globalOffset, 
    globalSize, localSize, nil, nil)

// 执行任务
err = cl.EnqueueTask(queue, kernel, nil, nil)
```

### 图像处理

```go
// 创建图像
image, err := cl.CreateImage(ctx, flags, format, desc, hostPtr)

// 读写图像
_, err = cl.EnqueueWriteImage(queue, image, blocking, origin, region, 
    rowPitch, slicePitch, ptr, nil)
_, err = cl.EnqueueReadImage(queue, image, blocking, origin, region, 
    rowPitch, slicePitch, ptr, nil)
```

### 事件管理

```go
// 等待事件完成
err = cl.WaitForEvents([]cl.Event{event})

// 获取事件状态
status, err := cl.GetEventInfo(event, cl.EventCommandExecutionStatus)

// 释放事件
err = cl.ReleaseEvent(event)
```

### 调试和错误处理

```go
// 启用调试日志
cl.EnableDebugLogging()

// 验证对象
err = cl.ValidateContext(ctx)
err = cl.ValidateDevice(device)
err = cl.ValidateProgram(program, device)
err = cl.ValidateKernel(kernel)

// 获取错误摘要
summary := cl.GetErrorSummary()

// 安全执行
err = cl.SafeExecute("operation", func() error {
    return cl.SomeOperation()
})
```

## 🎯 示例项目

项目包含多个实用示例：

### 1. 基础示例

```bash
cd example
go run example.go
```

展示OpenCL基础功能：平台查询、设备管理、缓冲区操作、程序构建。

### 2. 矩阵乘法性能测试

```bash
cd example/matrix_multiply
go run main.go
```

比较CPU和GPU的矩阵乘法性能，测试不同矩阵大小（256x256 到 2048x2048）。

### 3. 向量运算性能测试

```bash
cd example/vector_operations
go run main.go
```

比较CPU和GPU的向量运算性能，执行复杂运算：`c = a * b + sin(a) + cos(b)`。

## 🔧 高级用法

### 多设备并行计算

```go
// 为每个设备创建独立的上下文和队列
for _, device := range devices {
    ctx, _ := cl.CreateContext(platform, []cl.DeviceID{device}, nil)
    queue, _ := cl.CreateCommandQueue(ctx, device, 0)
    
    // 并行执行任务
    go func(ctx cl.Context, queue cl.CommandQueue) {
        // 执行GPU计算任务
    }(ctx, queue)
}
```

### 内存优化

```go
// 使用内存映射减少数据传输
ptr, event, err := cl.EnqueueMapBuffer(queue, buffer, cl.Bool(1), 
    cl.MapWrite, 0, size, nil)
if err == nil {
    // 直接操作映射内存
    cl.EnqueueUnmapMemObject(queue, buffer, ptr, nil)
}
```

### 异步执行

```go
// 异步执行内核
var event cl.Event
err = cl.EnqueueNDRangeKernel(queue, kernel, cl.UInt(1), nil, 
    []cl.Size{cl.Size(workSize)}, nil, nil, &event)

// 执行其他任务...

// 等待内核完成
cl.WaitForEvents([]cl.Event{event})
```

## 🐛 故障排除

### 常见问题

1. **"未找到OpenCL平台"**
   - 确保已安装GPU驱动
   - 检查OpenCL运行时是否正确安装

2. **"未找到可用的OpenCL设备"**
   - 检查GPU是否支持OpenCL
   - 尝试更新GPU驱动

3. **构建错误**
   - 确保Go版本 >= 1.16
   - 检查CGO是否启用
   - Windows上可能需要安装MinGW或Visual Studio

### 调试技巧

```go
// 启用详细调试
cl.EnableDebugLogging()

// 检查程序构建状态
buildInfo, err := cl.GetDetailedBuildInfo(program, device)
if buildInfo.HasBuildErrors() {
    fmt.Println("构建错误:", buildInfo.Log)
}

// 获取错误摘要
fmt.Println(cl.GetErrorSummary())
```

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

⭐ 如果这个项目对您有帮助，请给它一个星标！
