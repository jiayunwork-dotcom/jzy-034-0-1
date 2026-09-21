// Package job orchestrates infiltration runs. A job is a self-contained
// description of a column, its soil, an initial water-content profile,
// boundary conditions and a time-marching request. Jobs share no mutable
// state: every run builds its own solver and state, so concurrent jobs
// cannot contaminate each other.
package job

import (
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/example/unsatflow/internal/errs"
	"github.com/example/unsatflow/internal/solver"
	"github.com/example/unsatflow/internal/vg"
)

// ColumnSpec describes the soil column geometry and grid.
type ColumnSpec struct {
	Length float64 `json:"length"` // column thickness [m]
	Cells  int     `json:"cells"`  // number of finite-difference cells
}

// ThetaPoint is one node of a piecewise-linear initial profile.
type ThetaPoint struct {
	Depth float64 `json:"depth"`
	Theta float64 `json:"theta"`
}

// ProfileSpec defines a water-content profile: uniform, per-cell values,
// or piecewise linear through points covering [0, length].
type ProfileSpec struct {
	Uniform *float64     `json:"theta,omitempty"`
	Points  []ThetaPoint `json:"points,omitempty"`
	Values  []float64    `json:"values,omitempty"`
}

// BoundarySpec groups the top and bottom boundary conditions.
type BoundarySpec struct {
	Top    solver.Boundary `json:"top"`
	Bottom solver.Boundary `json:"bottom"`
}

// TimeSpec controls the march: step size and total duration [s].
type TimeSpec struct {
	Dt    float64 `json:"dt"`
	Total float64 `json:"total"`
}

// Spec is a complete infiltration job description.
type Spec struct {
	Column   ColumnSpec   `json:"column"`
	Soil     vg.Params    `json:"soil"`
	Initial  ProfileSpec  `json:"initial"`
	Boundary BoundarySpec `json:"boundary"`
	Time     TimeSpec     `json:"time"`
}

// StepSpec describes a single-step request: same column/soil/boundary,
// plus the current profile to advance from and the step size.
type StepSpec struct {
	Column   ColumnSpec   `json:"column"`
	Soil     vg.Params    `json:"soil"`
	State    ProfileSpec  `json:"state"`
	Boundary BoundarySpec `json:"boundary"`
	Dt       float64      `json:"dt"`
}

// Snapshot is the column state at one saved time level.
type Snapshot struct {
	Time          float64   `json:"time"`
	Theta         []float64 `json:"theta"`
	Head          []float64 `json:"head"`
	Storage       float64   `json:"storage"`
	CumTopFlux    float64   `json:"cumTopFlux"`
	CumBottomFlux float64   `json:"cumBottomFlux"`
	MassResidual  float64   `json:"massResidual"`
	Iterations    int       `json:"iterations"`
}

// Result is the full time series produced by a job.
type Result struct {
	Snapshots []Snapshot `json:"snapshots"`
	Steps     int        `json:"steps"`
}

// maxSteps guards against accidental runaway requests.
const maxSteps = 100000

func buildSolver(col ColumnSpec, soil vg.Params, b BoundarySpec) (*solver.Solver, *errs.Error) {
	return solver.New(solver.Config{
		Soil:   soil,
		Length: col.Length,
		Cells:  col.Cells,
		Top:    b.Top,
		Bottom: b.Bottom,
		// Harmonic intercell averaging — the zero value — is deliberate.
	})
}

// profileToCells expands a profile specification onto the grid.
func profileToCells(p ProfileSpec, col ColumnSpec, soil vg.Params) ([]float64, *errs.Error) {
	n := col.Cells
	switch {
	case p.Uniform != nil:
		out := make([]float64, n)
		for i := range out {
			out[i] = *p.Uniform
		}
		return out, nil
	case len(p.Values) > 0:
		if len(p.Values) != n {
			return nil, errs.Newf(errs.CodeInvalidProfile,
				"values profile has %d entries, grid has %d cells", len(p.Values), n)
		}
		return append([]float64(nil), p.Values...), nil
	case len(p.Points) >= 2:
		pts := p.Points
		if pts[0].Depth > 0 || pts[len(pts)-1].Depth < col.Length {
			return nil, errs.Newf(errs.CodeInvalidProfile,
				"points must cover the full column [0, %g]", col.Length)
		}
		for i := 1; i < len(pts); i++ {
			if pts[i].Depth <= pts[i-1].Depth {
				return nil, errs.New(errs.CodeInvalidProfile, "points must be ordered by increasing depth")
			}
		}
		out := make([]float64, n)
		dz := col.Length / float64(n)
		seg := 0
		for i := 0; i < n; i++ {
			z := (float64(i) + 0.5) * dz
			for seg+1 < len(pts)-1 && pts[seg+1].Depth < z {
				seg++
			}
			a, b := pts[seg], pts[seg+1]
			out[i] = a.Theta + (b.Theta-a.Theta)*(z-a.Depth)/(b.Depth-a.Depth)
		}
		return out, nil
	}
	return nil, errs.New(errs.CodeInvalidProfile,
		"profile needs exactly one of: theta (uniform), values (per-cell), points (piecewise linear)")
}

func makeSnapshot(s *solver.Solver, st *solver.State, massResidual float64, iterations int) Snapshot {
	return Snapshot{
		Time:          st.Time,
		Theta:         append([]float64(nil), st.Theta...),
		Head:          append([]float64(nil), st.Head...),
		Storage:       s.Storage(st),
		CumTopFlux:    st.CumTop,
		CumBottomFlux: st.CumBottom,
		MassResidual:  massResidual,
		Iterations:    iterations,
	}
}

// RunFull executes a complete infiltration job and returns the snapshot
// series, including the initial state at t=0.
func RunFull(spec Spec) (*Result, *errs.Error) {
	s, e := buildSolver(spec.Column, spec.Soil, spec.Boundary)
	if e != nil {
		return nil, e
	}
	if !(spec.Time.Dt > 0) {
		return nil, errs.Newf(errs.CodeInvalidDt, "dt must be > 0, got %g", spec.Time.Dt)
	}
	if !(spec.Time.Total > 0) {
		return nil, errs.Newf(errs.CodeInvalidDuration, "total duration must be > 0, got %g", spec.Time.Total)
	}
	stepsF := spec.Time.Total / spec.Time.Dt
	steps := int(stepsF + 0.5)
	if steps < 1 || math.Abs(float64(steps)-stepsF) > 1e-9 {
		return nil, errs.Newf(errs.CodeInvalidDuration,
			"total (%g) must be a positive integer multiple of dt (%g)", spec.Time.Total, spec.Time.Dt)
	}
	if steps > maxSteps {
		return nil, errs.Newf(errs.CodeInvalidDuration, "%d steps exceeds the %d step limit", steps, maxSteps)
	}
	theta, e := profileToCells(spec.Initial, spec.Column, spec.Soil)
	if e != nil {
		return nil, e
	}
	state, e := s.NewState(theta)
	if e != nil {
		return nil, e
	}

	res := &Result{Snapshots: make([]Snapshot, 0, steps+1), Steps: steps}
	res.Snapshots = append(res.Snapshots, makeSnapshot(s, state, 0, 0))
	for k := 0; k < steps; k++ {
		out, e := s.Step(state, spec.Time.Dt)
		if e != nil {
			return nil, e
		}
		state = &out.After
		res.Snapshots = append(res.Snapshots, makeSnapshot(s, state, out.MassResidual, out.Iterations))
	}
	return res, nil
}

// RunStep advances exactly one time step from the supplied profile and
// reports the before/after states plus the mass-closure residual. It uses
// the same solver.Step as RunFull, so identical inputs give identical
// profiles.
func RunStep(spec StepSpec) (*solver.StepOutcome, *errs.Error) {
	s, e := buildSolver(spec.Column, spec.Soil, spec.Boundary)
	if e != nil {
		return nil, e
	}
	theta, e := profileToCells(spec.State, spec.Column, spec.Soil)
	if e != nil {
		return nil, e
	}
	state, e := s.NewState(theta)
	if e != nil {
		return nil, e
	}
	return s.Step(state, spec.Dt)
}

// Record is a completed (or failed) job kept by the Manager.
type Record struct {
	ID         string      `json:"id"`
	Status     string      `json:"status"` // "completed" | "failed"
	Spec       Spec        `json:"spec"`
	Result     *Result     `json:"result,omitempty"`
	Error      *errs.Error `json:"error,omitempty"`
	CreatedAt  time.Time   `json:"createdAt"`
	FinishedAt time.Time   `json:"finishedAt"`
}

// Stats backs the health endpoint.
type Stats struct {
	UptimeSeconds float64 `json:"uptimeSeconds"`
	JobsCompleted int64   `json:"jobsCompleted"`
	JobsFailed    int64   `json:"jobsFailed"`
	JobsStored    int     `json:"jobsStored"`
}

// Manager runs jobs and keeps their records. The store is the only shared
// state and is mutex-guarded; job execution itself touches no shared
// memory.
type Manager struct {
	mu        sync.RWMutex
	jobs      map[string]*Record
	started   time.Time
	seq       atomic.Int64
	completed atomic.Int64
	failed    atomic.Int64
}

// NewManager creates an empty manager.
func NewManager() *Manager {
	return &Manager{jobs: make(map[string]*Record), started: time.Now()}
}

// Submit executes a job synchronously and stores its record.
func (m *Manager) Submit(spec Spec) *Record {
	rec := &Record{Spec: spec, CreatedAt: time.Now()}
	rec.ID = rec.CreatedAt.UTC().Format("20060102T150405.000000000") + "-" + itoa(m.seq.Add(1))
	res, e := RunFull(spec)
	rec.FinishedAt = time.Now()
	if e != nil {
		rec.Status = "failed"
		rec.Error = e
		m.failed.Add(1)
	} else {
		rec.Status = "completed"
		rec.Result = res
		m.completed.Add(1)
	}
	m.mu.Lock()
	m.jobs[rec.ID] = rec
	m.mu.Unlock()
	return rec
}

// Get fetches a stored job record.
func (m *Manager) Get(id string) (*Record, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	r, ok := m.jobs[id]
	return r, ok
}

// Stats reports liveness and job counters.
func (m *Manager) Stats() Stats {
	m.mu.RLock()
	stored := len(m.jobs)
	m.mu.RUnlock()
	return Stats{
		UptimeSeconds: time.Since(m.started).Seconds(),
		JobsCompleted: m.completed.Load(),
		JobsFailed:    m.failed.Load(),
		JobsStored:    stored,
	}
}

func itoa(v int64) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
