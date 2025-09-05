package main

import (
	"bufio"
	"fmt"
	"opencl/cl"
	"os"
)

func main() {
	fmt.Println("=== OpenCL 平台、设备、上下文与命令队列管理示例 ===")

	// 1. 获取所有平台
	fmt.Println("\n1. 获取所有平台:")
	platforms, err := cl.GetPlatformIDs()
	if err != nil {
		fmt.Printf("获取平台失败: %s\n", err.Error())
		return
	}

	if len(platforms) == 0 {
		fmt.Println("未找到任何 OpenCL 平台")
		return
	}

	var selectedPlatform cl.PlatformID
	var selectedDevices []cl.DeviceID

	// 2. 遍历每个平台
	for i, platform := range platforms {
		fmt.Printf("\n--- 平台 %d ---\n", i+1)
		// 获取平台详细信息
		platformInfo, err := cl.GetPlatformDetails(platform)
		if err != nil {
			fmt.Printf("获取平台信息失败: %s\n", err.Error())
			continue
		}

		fmt.Printf("名称: %s\n", platformInfo.Name)
		fmt.Printf("厂商: %s\n", platformInfo.Vendor)
		fmt.Printf("版本: %s\n", platformInfo.Version)
		fmt.Printf("配置文件: %s\n", platformInfo.Profile)

		// 3. 获取该平台下的所有设备
		fmt.Println("\n设备列表:")
		devices, err := cl.GetDeviceIDs(platform, cl.DeviceTypeAll)
		if err != nil {
			fmt.Printf("获取设备失败: %s\n", err.Error())
			continue
		}

		if len(devices) == 0 {
			fmt.Println("  未找到任何设备")
			continue
		}

		// 保存第一个有设备的平台和设备列表用于后续演示
		if selectedPlatform == nil && len(devices) > 0 {
			selectedPlatform = platform
			selectedDevices = devices
		}

		// 4. 遍历每个设备
		for j, device := range devices {
			fmt.Printf("\n  设备 %d:\n", j+1)

			deviceInfo, err := cl.GetDeviceDetails(device)
			if err != nil {
				fmt.Printf("    获取设备信息失败: %s\n", err.Error())
				continue
			}

			fmt.Printf("    名称: %s\n", deviceInfo.Name)
			fmt.Printf("    厂商: %s\n", deviceInfo.Vendor)
			fmt.Printf("    版本: %s\n", deviceInfo.Version)

			// 设备类型
			deviceType := "未知"
			switch deviceInfo.Type {
			case cl.DeviceTypeCPU:
				deviceType = "CPU"
			case cl.DeviceTypeGPU:
				deviceType = "GPU"
			case cl.DeviceTypeACC:
				deviceType = "加速器"
			}
			fmt.Printf("    类型: %s\n", deviceType)
			fmt.Printf("    最大内存分配: %d 字节\n", deviceInfo.MaxMemAlloc)
			fmt.Printf("    最大工作组大小: %d\n", deviceInfo.MaxWorkGroup)
		}
	}

	// 5. 演示上下文和命令队列创建
	if selectedPlatform != nil && len(selectedDevices) > 0 {
		fmt.Println("\n=== 上下文与命令队列演示 ===")

		// 5.1 创建上下文
		fmt.Println("\n5.1 创建上下文:")

		// 方法1: 使用指定设备创建上下文
		fmt.Println("  方法1: 使用指定设备创建上下文")
		properties := map[cl.UInt]interface{}{
			cl.ContextPlatform: selectedPlatform,
		}

		context, err := cl.CreateContext(selectedPlatform, selectedDevices, properties)
		if err != nil {
			fmt.Printf("  创建上下文失败: %s\n", err.Error())
		} else {
			fmt.Println("  ✓ 上下文创建成功")

			// 获取上下文中的设备
			contextDevices, err := cl.GetContextDevices(context)
			if err != nil {
				fmt.Printf("  获取上下文设备失败: %s\n", err.Error())
			} else {
				fmt.Printf("  上下文包含设备数量: %d\n", len(contextDevices))
			}

			// 5.2 创建命令队列
			fmt.Println("\n5.2 创建命令队列:")

			// 方法1: 使用旧版API创建命令队列
			fmt.Println("  方法1: 使用 CreateCommandQueue (OpenCL 1.x)")
			queue1, err := cl.CreateCommandQueue(context, selectedDevices[0], 0)
			if err != nil {
				fmt.Printf("  创建命令队列失败: %s\n", err.Error())
			} else {
				fmt.Println("  ✓ 命令队列创建成功")

				// 获取命令队列信息
				queueContext, err := cl.GetCommandQueueContext(queue1)
				if err != nil {
					fmt.Printf("  获取命令队列上下文失败: %s\n", err.Error())
				} else {
					fmt.Printf("  命令队列关联的上下文: %p\n", queueContext)
				}

				queueDevice, err := cl.GetCommandQueueDevice(queue1)
				if err != nil {
					fmt.Printf("  获取命令队列设备失败: %s\n", err.Error())
				} else {
					fmt.Printf("  命令队列关联的设备: %p\n", queueDevice)
				}

				// 演示队列操作
				fmt.Println("  演示队列操作:")
				fmt.Println("    - 执行 Flush 操作...")
				err = cl.Flush(queue1)
				if err != nil {
					fmt.Printf("    Flush 失败: %s\n", err.Error())
				} else {
					fmt.Println("    ✓ Flush 成功")
				}

				fmt.Println("    - 执行 Finish 操作...")
				err = cl.Finish(queue1)
				if err != nil {
					fmt.Printf("    Finish 失败: %s\n", err.Error())
				} else {
					fmt.Println("    ✓ Finish 成功")
				}

				// 释放命令队列
				err = cl.ReleaseCommandQueue(queue1)
				if err != nil {
					fmt.Printf("  释放命令队列失败: %s\n", err.Error())
				} else {
					fmt.Println("  ✓ 命令队列已释放")
				}
			}

			// 方法2: 使用新版API创建命令队列（带属性）
			fmt.Println("\n  方法2: 使用 CreateCommandQueueWithProperties (OpenCL 2.x+)")
			queueProperties := map[cl.UInt]interface{}{
				cl.QueueProperties: cl.QueueProfilingEnable,
			}

			queue2, err := cl.CreateCommandQueueWithProperties(context, selectedDevices[0], queueProperties)
			if err != nil {
				fmt.Printf("  创建命令队列失败: %s\n", err.Error())
			} else {
				fmt.Println("  ✓ 带属性的命令队列创建成功")

				// 获取队列属性
				properties, err := cl.GetCommandQueueProperties(queue2)
				if err != nil {
					fmt.Printf("  获取命令队列属性失败: %s\n", err.Error())
				} else {
					fmt.Printf("  命令队列属性: 0x%x\n", properties)
					if properties&cl.QueueProfilingEnable != 0 {
						fmt.Println("  ✓ 性能分析已启用")
					}
				}

				// 释放命令队列
				err = cl.ReleaseCommandQueue(queue2)
				if err != nil {
					fmt.Printf("  释放命令队列失败: %s\n", err.Error())
				} else {
					fmt.Println("  ✓ 命令队列已释放")
				}
			}

			// 方法3: 根据设备类型创建上下文
			fmt.Println("\n  方法3: 根据设备类型创建上下文")
			contextFromType, err := cl.CreateContextFromType(selectedPlatform, cl.DeviceTypeAll, properties)
			if err != nil {
				fmt.Printf("  创建上下文失败: %s\n", err.Error())
			} else {
				fmt.Println("  ✓ 根据设备类型的上下文创建成功")

				// 释放上下文
				err = cl.ReleaseContext(contextFromType)
				if err != nil {
					fmt.Printf("  释放上下文失败: %s\n", err.Error())
				} else {
					fmt.Println("  ✓ 上下文已释放")
				}
			}

			// 释放主上下文
			err = cl.ReleaseContext(context)
			if err != nil {
				fmt.Printf("释放上下文失败: %s\n", err.Error())
			} else {
				fmt.Println("✓ 主上下文已释放")
			}
		}
	}

	fmt.Println("\n=== 示例完成 ===")
	fmt.Println("按回车键退出...")
	bufio.NewReader(os.Stdin).ReadBytes('\n')
}
