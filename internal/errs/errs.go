// Package errs defines the typed, structured errors returned by the service.
// Every validation or runtime failure carries a stable machine-readable Code
// so callers can branch on the failure class instead of parsing messages.
package errs

import "fmt"

// Stable error codes exposed through the HTTP API.
const (
	CodeInvalidJSON         = "INVALID_JSON"
	CodeInvalidVGN          = "INVALID_VG_N"
	CodeInvalidVGAlpha      = "INVALID_VG_ALPHA"
	CodeInvalidThetaRange   = "INVALID_THETA_RANGE"
	CodeInvalidKs           = "INVALID_KS"
	CodeInvalidColumnLength = "INVALID_COLUMN_LENGTH"
	CodeInvalidCells        = "INVALID_CELLS"
	CodeInvalidDt           = "INVALID_DT"
	CodeInvalidDuration     = "INVALID_DURATION"
	CodeInvalidBoundary     = "INVALID_BOUNDARY"
	CodeInvalidProfile      = "INVALID_THETA_PROFILE"
	CodeNotConverged        = "STEP_NOT_CONVERGED"
	CodeThetaOutOfRange     = "THETA_OUT_OF_RANGE"
	CodeJobNotFound         = "JOB_NOT_FOUND"
)

// Error is a typed service error.
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *Error) Error() string { return fmt.Sprintf("%s: %s", e.Code, e.Message) }

// New builds a typed error.
func New(code, message string) *Error { return &Error{Code: code, Message: message} }

// Newf builds a typed error with a formatted message.
func Newf(code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}
