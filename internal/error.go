package internal

import (
	"fmt"
	"runtime"
	"strings"
	"time"
)

// ErrorContext 错误上下文信息
type ErrorContext struct {
	Function    string
	File        string
	Line        int
	Timestamp   time.Time
	Description string
	ErrorCode   int
}

// NewErrorContext 创建新的错误上下文
func NewErrorContext(function, description string, errorCode int) *ErrorContext {
	_, file, line, _ := runtime.Caller(1)
	return &ErrorContext{
		Function:    function,
		File:        file,
		Line:        line,
		Timestamp:   time.Now(),
		Description: description,
		ErrorCode:   errorCode,
	}
}

// String 返回错误上下文的字符串表示
func (ec *ErrorContext) String() string {
	return fmt.Sprintf("[%s] %s:%d - %s (Error: %d)",
		ec.Timestamp.Format("2006-01-02 15:04:05"),
		ec.File,
		ec.Line,
		ec.Description,
		ec.ErrorCode)
}

// DetailedString 返回详细的错误上下文信息
func (ec *ErrorContext) DetailedString() string {
	var sb strings.Builder
	sb.WriteString("=== OpenCL Error Context ===\n")
	sb.WriteString(fmt.Sprintf("Function: %s\n", ec.Function))
	sb.WriteString(fmt.Sprintf("File: %s\n", ec.File))
	sb.WriteString(fmt.Sprintf("Line: %d\n", ec.Line))
	sb.WriteString(fmt.Sprintf("Timestamp: %s\n", ec.Timestamp.Format("2006-01-02 15:04:05.000")))
	sb.WriteString(fmt.Sprintf("Description: %s\n", ec.Description))
	sb.WriteString(fmt.Sprintf("Error Code: %d\n", ec.ErrorCode))
	sb.WriteString("===========================")
	return sb.String()
}

// ErrorLogger 错误日志记录器
type ErrorLogger struct {
	errors []*ErrorContext
	maxLog int // 最大日志数量
}

// NewErrorLogger 创建新的错误日志记录器
func NewErrorLogger(maxLog int) *ErrorLogger {
	if maxLog <= 0 {
		maxLog = 100
	}
	return &ErrorLogger{
		errors: make([]*ErrorContext, 0),
		maxLog: maxLog,
	}
}

// LogError 记录错误
func (el *ErrorLogger) LogError(function, description string, errorCode int) {
	context := NewErrorContext(function, description, errorCode)
	if len(el.errors) >= el.maxLog {
		el.errors = el.errors[1:]
	}
	el.errors = append(el.errors, context)
}

// GetErrors 获取所有错误日志
func (el *ErrorLogger) GetErrors() []*ErrorContext {
	return el.errors
}

// GetLastError 获取最后一个错误
func (el *ErrorLogger) GetLastError() *ErrorContext {
	if len(el.errors) == 0 {
		return nil
	}
	return el.errors[len(el.errors)-1]
}

// ClearErrors 清空错误日志
func (el *ErrorLogger) ClearErrors() {
	el.errors = el.errors[:0]
}

// ErrorCount 获取错误数量
func (el *ErrorLogger) ErrorCount() int {
	return len(el.errors)
}

// GetErrorsByCode 根据错误码获取错误
func (el *ErrorLogger) GetErrorsByCode(errorCode int) []*ErrorContext {
	var result []*ErrorContext
	for _, err := range el.errors {
		if err.ErrorCode == errorCode {
			result = append(result, err)
		}
	}
	return result
}

// GetErrorsByFunction 根据函数名获取错误
func (el *ErrorLogger) GetErrorsByFunction(function string) []*ErrorContext {
	var result []*ErrorContext
	for _, err := range el.errors {
		if err.Function == function {
			result = append(result, err)
		}
	}
	return result
}

// Summary 获取错误摘要
func (el *ErrorLogger) Summary() string {
	if len(el.errors) == 0 {
		return "No errors recorded"
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Error Summary: %d total errors\n", len(el.errors)))

	// 统计错误码分布
	errorCodeCount := make(map[int]int)
	for _, err := range el.errors {
		errorCodeCount[err.ErrorCode]++
	}

	sb.WriteString("Error Code Distribution:\n")
	for code, count := range errorCodeCount {
		sb.WriteString(fmt.Sprintf("  Code %d: %d occurrences\n", code, count))
	}

	// 显示最近的几个错误
	recentCount := min(3, len(el.errors))
	sb.WriteString("Recent Errors:\n")
	for i := len(el.errors) - recentCount; i < len(el.errors); i++ {
		sb.WriteString(fmt.Sprintf("  %s\n", el.errors[i].String()))
	}

	return sb.String()
}

// DebugHelper 调试辅助工具
type DebugHelper struct {
	logger *ErrorLogger
}

// NewDebugHelper 创建新的调试辅助工具
func NewDebugHelper() *DebugHelper {
	return &DebugHelper{
		logger: NewErrorLogger(50),
	}
}

// CheckError 检查错误并记录
func (dh *DebugHelper) CheckError(function string, err error) error {
	if err != nil {
		// 尝试从错误中提取错误码
		errorCode := -1
		if openCLErr, ok := err.(interface{ Code() int }); ok {
			errorCode = openCLErr.Code()
		}

		dh.logger.LogError(function, err.Error(), errorCode)
	}
	return err
}

// LogInfo 记录信息
func (dh *DebugHelper) LogInfo(function, message string) {
	fmt.Printf("[INFO] %s: %s\n", function, message)
}

// LogWarning 记录警告
func (dh *DebugHelper) LogWarning(function, message string) {
	fmt.Printf("[WARNING] %s: %s\n", function, message)
}

// GetLogger 获取错误日志记录器
func (dh *DebugHelper) GetLogger() *ErrorLogger {
	return dh.logger
}

// PrintStackTrace 打印堆栈跟踪
func PrintStackTrace() {
	fmt.Println("=== Stack Trace ===")
	buf := make([]byte, 1024)
	for {
		n := runtime.Stack(buf, false)
		if n < len(buf) {
			fmt.Println(string(buf[:n]))
			break
		}
		buf = make([]byte, 2*len(buf))
	}
	fmt.Println("===================")
}

// GetCallerInfo 获取调用者信息
func GetCallerInfo(skip int) (function, file string, line int) {
	pc, file, line, ok := runtime.Caller(skip + 1)
	if !ok {
		return "unknown", "unknown", 0
	}

	fn := runtime.FuncForPC(pc)
	if fn != nil {
		function = fn.Name()
	}

	return function, file, line
}

// FormatErrorWithContext 格式化错误和上下文信息
func FormatErrorWithContext(err error, context string) string {
	if err == nil {
		return ""
	}

	function, file, line := GetCallerInfo(1)
	return fmt.Sprintf("[%s] %s:%d - %s: %v",
		context,
		file,
		line,
		function,
		err)
}
