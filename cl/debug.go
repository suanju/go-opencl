package cl

import (
	"fmt"
	"opencl/internal"
)

var dbg = internal.NewDebugHelper()

func wrapErr(op string, err error) error {
	if err == nil {
		return nil
	}
	dbg.CheckError(op, err)
	fmt.Printf("[ERROR] %s: %v\n", op, err)
	return fmt.Errorf("%s failed: %w", op, err)
}
func MustSucceed(err error, op string) {
	if err != nil {
		panic(fmt.Sprintf("%s failed: %v", op, err))
	}
}

func LogWarning(op, msg string)    { fmt.Printf("[WARNING] %s: %s\n", op, msg) }
func LogError(s string, err error) { fmt.Printf("[ERROR] %s: %s\n", s, err) }
func LogInfo(op, msg string)       { fmt.Printf("[INFO] %s: %s\n", op, msg) }

func EnableDebugLogging()  { LogInfo("Debug", "enabled") }
func DisableDebugLogging() { LogInfo("Debug", "disabled") }

func GetErrorSummary() string { return dbg.GetLogger().Summary() }
func ClearErrorLog() {
	dbg.GetLogger().ClearErrors()
	LogInfo("ErrorLog", "cleared")
}
func PrintErrorLog() {
	le := dbg.GetLogger().GetErrors()
	if len(le) == 0 {
		fmt.Println("No errors in log")
		return
	}
	fmt.Printf("=== Error Log (%d errors) ===\n", len(le))
	for i, e := range le {
		fmt.Printf("%d. %s\n", i+1, e.String())
	}
	fmt.Println("==========================")
}
func PrintStackTrace() { internal.PrintStackTrace() }

func SafeExecute(op string, fn func() error) error {
	if err := fn(); err != nil {
		return wrapErr(op, err)
	}
	return nil
}

func SafeExecuteWithRecover(op string, fn func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%s panicked: %v", op, r)
			fmt.Printf("[ERROR] %s: %v\n", op, err)
			PrintStackTrace()
		}
	}()
	return SafeExecute(op, fn)
}

func ValidateProgram(program Program, device DeviceID) error {
	buildInfo, err := GetDetailedBuildInfo(program, device)
	if err != nil {
		return wrapErr("GetDetailedBuildInfo", err)
	}
	if buildInfo.HasBuildErrors() {
		return fmt.Errorf("program build failed: %s\nBuild log:\n%s",
			BuildStatusString(buildInfo.Status), buildInfo.Log)
	}
	if buildInfo.IsBuilding() {
		return fmt.Errorf("program is still building")
	}
	if !buildInfo.IsBuildSuccessful() {
		return fmt.Errorf("program build status unknown: %s",
			BuildStatusString(buildInfo.Status))
	}
	return nil
}

func ValidateKernel(kernel Kernel) error {
	numArgs, err := GetKernelNumArgs(kernel)
	if err != nil {
		return wrapErr("GetKernelNumArgs", err)
	}
	if numArgs == 0 {
		LogWarning("ValidateKernel", "no arguments")
	}
	name, err := GetKernelFunctionName(kernel)
	if err != nil {
		return wrapErr("GetKernelFunctionName", err)
	}
	LogInfo("ValidateKernel", fmt.Sprintf("kernel '%s' has %d arguments", name, numArgs))
	return nil
}

func ValidateContext(context Context) error {
	if context == nil {
		return fmt.Errorf("context is nil")
	}
	LogInfo("ValidateContext", "ok")
	return nil
}

func ValidateDevice(device DeviceID) error {
	if device == nil {
		return fmt.Errorf("device is nil")
	}
	LogInfo("ValidateDevice", "ok")
	return nil
}

func (e OpenCLError) GetErrorCode() int { return int(e.Code) }

func IsSuccess(err Int) bool { return err == 0 }
func IsError(err Int) bool   { return err != 0 }

func GetErrorSeverity(err Int) string {
	switch err {
	case 0:
		return "SUCCESS"
	case -1, -2:
		return "WARNING"
	case -5, -6:
		return "CRITICAL"
	case -30, -31, -32:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}
