// Package api wires the HTTP routes (Gin) onto the job orchestration
// layer. It owns no numerics: validation errors from vg/solver/job are
// translated into typed JSON error bodies here.
package api

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/example/unsatflow/internal/errs"
	"github.com/example/unsatflow/internal/job"
	"github.com/example/unsatflow/internal/solver"
	"github.com/example/unsatflow/internal/vg"
)

// Server bundles the router and the job manager.
type Server struct {
	Router  *gin.Engine
	manager *job.Manager
}

// New builds the HTTP server with all routes.
func New() *Server {
	gin.SetMode(gin.ReleaseMode)
	s := &Server{manager: job.NewManager()}
	r := gin.New()
	r.Use(gin.Recovery())

	v1 := r.Group("/api/v1")
	v1.POST("/jobs", s.postJob)
	v1.GET("/jobs/:id", s.getJob)
	v1.POST("/steps", s.postStep)
	v1.GET("/constitutive", s.getConstitutive)
	v1.GET("/health", s.getHealth)
	v1.GET("/examples/sand-ponding", s.getSandPondingExample)

	s.Router = r
	return s
}

// writeErr maps typed errors to HTTP status codes and a structured body.
func writeErr(c *gin.Context, e *errs.Error) {
	status := http.StatusBadRequest
	switch e.Code {
	case errs.CodeJobNotFound:
		status = http.StatusNotFound
	case errs.CodeNotConverged, errs.CodeThetaOutOfRange:
		status = http.StatusUnprocessableEntity
	}
	c.JSON(status, gin.H{"error": e})
}

func (s *Server) postJob(c *gin.Context) {
	var spec job.Spec
	if err := c.ShouldBindJSON(&spec); err != nil {
		writeErr(c, errs.Newf(errs.CodeInvalidJSON, "cannot parse job spec: %v", err))
		return
	}
	rec := s.manager.Submit(spec)
	if rec.Error != nil {
		writeErr(c, rec.Error)
		return
	}
	c.JSON(http.StatusCreated, rec)
}

func (s *Server) getJob(c *gin.Context) {
	rec, ok := s.manager.Get(c.Param("id"))
	if !ok {
		writeErr(c, errs.Newf(errs.CodeJobNotFound, "no job with id %q", c.Param("id")))
		return
	}
	c.JSON(http.StatusOK, rec)
}

func (s *Server) postStep(c *gin.Context) {
	var spec job.StepSpec
	if err := c.ShouldBindJSON(&spec); err != nil {
		writeErr(c, errs.Newf(errs.CodeInvalidJSON, "cannot parse step spec: %v", err))
		return
	}
	out, e := job.RunStep(spec)
	if e != nil {
		writeErr(c, e)
		return
	}
	c.JSON(http.StatusOK, out)
}

// getConstitutive echoes the constitutive model form and its fixed
// constants so callers can verify exactly what is being solved.
func (s *Server) getConstitutive(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"model": "van Genuchten / Mualem",
		"formulas": gin.H{
			"effectiveSaturation": "Se(h) = [1 + (alpha*|h|)^n]^(-m) for h < 0, Se = 1 for h >= 0",
			"waterContent":        "theta(h) = thetaR + (thetaS - thetaR) * Se",
			"conductivity":        "K(h) = Ks * Se^(1/2) * [1 - (1 - Se^(1/m))^m]^2",
			"capacity":            "C(h) = d(theta)/dh (analytic)",
		},
		"constants": gin.H{
			"mBinding":               "m = 1 - 1/n",
			"conductivityReductionExponent": 0.5,
			"timeScheme":                  "backward Euler (fully implicit)",
			"intercellConductivityAverage":  "harmonic",
			"nonlinearSolver": gin.H{
				"method":        "damped Newton, analytic tridiagonal Jacobian",
				"residualTol":   1e-11,
				"maxIterations": 60,
			},
		},
	})
}

func (s *Server) getHealth(c *gin.Context) {
	stats := s.manager.Stats()
	c.JSON(http.StatusOK, gin.H{
		"status":        "ok",
		"uptimeSeconds": stats.UptimeSeconds,
		"jobsCompleted": stats.JobsCompleted,
		"jobsFailed":    stats.JobsFailed,
		"jobsStored":    stats.JobsStored,
	})
}

// SandPondingSpec is the preset verifiable example: ponded infiltration
// into a 1 m sand column. The wetting front moves downward and total
// storage grows monotonically.
func SandPondingSpec() job.Spec {
	thetaInit := 0.10
	return job.Spec{
		Column: job.ColumnSpec{Length: 1.0, Cells: 100},
		Soil: vg.Params{
			ThetaR: 0.045,
			ThetaS: 0.43,
			Alpha:  14.5,
			N:      2.68,
			Ks:     8.25e-5,
		},
		Initial: job.ProfileSpec{Uniform: &thetaInit},
		Boundary: job.BoundarySpec{
			Top:    solver.Boundary{Kind: solver.BCDirichlet, Head: 0.10},
			Bottom: solver.Boundary{Kind: solver.BCFreeDrainage},
		},
		Time: job.TimeSpec{Dt: 10.0, Total: 3600.0},
	}
}

func (s *Server) getSandPondingExample(c *gin.Context) {
	c.JSON(http.StatusOK, SandPondingSpec())
}
