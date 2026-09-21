// Package solver advances the 1-D Richards equation on a vertical soil
// column with a fully implicit (backward Euler) finite-difference scheme.
//
// Coordinates: z is depth, positive downward, z in [0, L]. The downward
// Darcy flux is q = K(h) * (1 - dh/dz), so hydrostatic conditions
// (dh/dz = 1) give zero flux. Mass conservation reads
//
//	d(theta)/dt = -dq/dz.
//
// Cell-centred grid: N cells of width dz = L/N, centres at (i+1/2)*dz.
// Face fluxes F[j] (j = 0..N) are positive downward. The semi-discrete
// conservative form is
//
//	(theta_i^{n+1} - theta_i^n)/dt + (F[i+1] - F[i])/dz = 0,
//
// which telescopes exactly, so the column storage change per step equals
// dt*(F[0] - F[N]) to nonlinear-solver precision. Intercell conductivity
// is the harmonic mean of the two adjacent cell conductivities — the
// arithmetic mean is known to misplace the wetting front when K spans
// orders of magnitude.
package solver

import (
	"math"

	"github.com/example/unsatflow/internal/errs"
	"github.com/example/unsatflow/internal/vg"
)

// BCKind enumerates the supported boundary-condition types.
type BCKind string

const (
	// BCDirichlet prescribes the pressure head (ponding) at the surface.
	BCDirichlet BCKind = "dirichlet"
	// BCNeumann prescribes the downward flux at the surface.
	BCNeumann BCKind = "neumann"
	// BCFreeDrainage is a unit-gradient bottom boundary (q = K(h)).
	BCFreeDrainage BCKind = "freeDrainage"
	// BCNoFlow imposes zero flux.
	BCNoFlow BCKind = "noFlow"
)

// Boundary describes one column boundary. Head is used by BCDirichlet,
// Flux (positive downward, into the column at the top) by BCNeumann.
type Boundary struct {
	Kind BCKind `json:"type"`
	Head float64 `json:"head,omitempty"`
	Flux float64 `json:"flux,omitempty"`
}

// Averaging selects the intercell conductivity average. Harmonic is the
// physically correct default; Arithmetic exists only so tests can
// demonstrate that the choice matters.
type Averaging int

const (
	// Harmonic averaging: 2*Ka*Kb/(Ka+Kb).
	Harmonic Averaging = iota
	// Arithmetic averaging: (Ka+Kb)/2. Not used by the service paths.
	Arithmetic
)

// Config bundles everything needed to build a Solver.
type Config struct {
	Soil      vg.Params
	Length    float64 // column thickness [m]
	Cells     int     // number of finite-difference cells
	Top       Boundary
	Bottom    Boundary
	Averaging Averaging // zero value = Harmonic
}

// Solver owns the grid and constitutive model. It is stateless with
// respect to time stepping: all evolving quantities live in State, so
// concurrent jobs never share mutable memory.
type Solver struct {
	cfg Config
	dz  float64
}

// Newton solver tolerances.
const (
	resTol    = 1e-11 // max-abs residual tolerance [1/s]
	maxIter   = 60    // max Newton iterations per step
	headFloor = -1e6  // [m] below this the iteration is declared diverged
)

// New validates the configuration and builds a solver.
func New(cfg Config) (*Solver, *errs.Error) {
	if e := cfg.Soil.Validate(); e != nil {
		return nil, e
	}
	if !(cfg.Length > 0) {
		return nil, errs.Newf(errs.CodeInvalidColumnLength, "column length must be > 0, got %g", cfg.Length)
	}
	if cfg.Cells < 2 {
		return nil, errs.Newf(errs.CodeInvalidCells, "need at least 2 cells, got %d", cfg.Cells)
	}
	switch cfg.Top.Kind {
	case BCDirichlet, BCNeumann:
	default:
		return nil, errs.Newf(errs.CodeInvalidBoundary,
			"top boundary must be %q or %q, got %q", BCDirichlet, BCNeumann, cfg.Top.Kind)
	}
	switch cfg.Bottom.Kind {
	case BCFreeDrainage, BCNoFlow:
	default:
		return nil, errs.Newf(errs.CodeInvalidBoundary,
			"bottom boundary must be %q or %q, got %q", BCFreeDrainage, BCNoFlow, cfg.Bottom.Kind)
	}
	return &Solver{cfg: cfg, dz: cfg.Length / float64(cfg.Cells)}, nil
}

// Dz returns the cell width.
func (s *Solver) Dz() float64 { return s.dz }

// Cells returns the number of grid cells.
func (s *Solver) Cells() int { return s.cfg.Cells }

// State is the full evolving column state. Cumulative fluxes are positive
// into the column at the top and positive out of the column at the bottom.
type State struct {
	Head      []float64 `json:"head"`  // pressure head at cell centres [m]
	Theta     []float64 `json:"theta"` // water content at cell centres [-]
	Time      float64   `json:"time"`
	CumTop    float64   `json:"cumTopFlux"`    // cumulative infiltration [m]
	CumBottom float64   `json:"cumBottomFlux"` // cumulative drainage [m]
}

// Storage returns the total water stored in the column [m].
func (s *Solver) Storage(st *State) float64 {
	sum := 0.0
	for _, th := range st.Theta {
		sum += th
	}
	return sum * s.dz
}

// NewState builds a State from a water-content profile defined at cell
// centres. Values outside [thetaR, thetaS] are rejected.
func (s *Solver) NewState(theta []float64) (*State, *errs.Error) {
	if len(theta) != s.cfg.Cells {
		return nil, errs.Newf(errs.CodeInvalidProfile,
			"theta profile has %d values, grid has %d cells", len(theta), s.cfg.Cells)
	}
	p := s.cfg.Soil
	h := make([]float64, len(theta))
	th := make([]float64, len(theta))
	for i, v := range theta {
		if math.IsNaN(v) || v < p.ThetaR || v > p.ThetaS {
			return nil, errs.Newf(errs.CodeInvalidProfile,
				"initial theta[%d]=%g outside [thetaR=%g, thetaS=%g]", i, v, p.ThetaR, p.ThetaS)
		}
		th[i] = v
		h[i] = p.HeadFromTheta(v)
	}
	return &State{Head: h, Theta: th}, nil
}

// StepOutcome reports one fully implicit time step.
type StepOutcome struct {
	Before        State   `json:"before"`
	After         State   `json:"after"`
	TopFlux       float64 `json:"topFlux"`       // mean downward flux at top during the step [m/s]
	BottomFlux    float64 `json:"bottomFlux"`    // mean downward flux at bottom during the step [m/s]
	StorageBefore float64 `json:"storageBefore"` // [m]
	StorageAfter  float64 `json:"storageAfter"`  // [m]
	// MassResidual = dStorage - dt*(topFlux-bottomFlux); identically ~0
	// for the conservative scheme once Newton has converged.
	MassResidual float64 `json:"massResidual"`
	Iterations   int     `json:"iterations"`
}

// Step advances the state by one implicit step of size dt. The input
// state is not modified; the outcome carries deep copies.
func (s *Solver) Step(st *State, dt float64) (*StepOutcome, *errs.Error) {
	if !(dt > 0) {
		return nil, errs.Newf(errs.CodeInvalidDt, "time step must be > 0, got %g", dt)
	}
	n := s.cfg.Cells
	p := s.cfg.Soil

	h := make([]float64, n)
	copy(h, st.Head)
	thetaOld := make([]float64, n)
	copy(thetaOld, st.Theta)

	// Scratch arrays.
	theta := make([]float64, n)
	capC := make([]float64, n)
	kCell := make([]float64, n)
	dkCell := make([]float64, n)
	flux := make([]float64, n+1)
	res := make([]float64, n)
	lower := make([]float64, n)
	diag := make([]float64, n)
	upper := make([]float64, n)
	dh := make([]float64, n)

	// assemble computes theta, conductivities, face fluxes, the residual
	// and (when jacobian=true) the tridiagonal Jacobian for the current h.
	assemble := func(jacobian bool) {
		for i := 0; i < n; i++ {
			theta[i] = p.Theta(h[i])
			capC[i] = p.Capacity(h[i])
			kCell[i] = p.K(h[i])
			dkCell[i] = p.DKDh(h[i])
		}
		s.faceFluxes(h, kCell, flux)

		for i := 0; i < n; i++ {
			res[i] = (theta[i]-thetaOld[i])/dt + (flux[i+1]-flux[i])/s.dz
		}
		if !jacobian {
			return
		}
		for i := 0; i < n; i++ {
			lower[i], diag[i], upper[i] = 0, 0, 0
		}
		// Interior faces j=1..n-1 between cells j-1 and j.
		for j := 1; j < n; j++ {
			kf, dkfA, dkfB := s.faceK(kCell[j-1], kCell[j])
			grad := 1.0 - (h[j]-h[j-1])/s.dz
			dFa := dkfA*dkCell[j-1]*grad + kf/s.dz // dF_j/dh_{j-1}
			dFb := dkfB*dkCell[j]*grad - kf/s.dz   // dF_j/dh_j
			// R_{j-1} contains +F_j/dz; R_j contains -F_j/dz.
			upper[j-1] += dFb / s.dz
			diag[j-1] += dFa / s.dz
			lower[j] -= dFa / s.dz
			diag[j] -= dFb / s.dz
		}
		// Top boundary face 0 (enters R_0 as -F_0/dz).
		if s.cfg.Top.Kind == BCDirichlet {
			kTop := p.K(s.cfg.Top.Head)
			kf, _, dkfB := s.faceK(kTop, kCell[0])
			grad := 1.0 - (h[0]-s.cfg.Top.Head)/(s.dz/2.0)
			diag[0] -= (dkfB*dkCell[0]*grad - kf/(s.dz/2.0)) / s.dz
		}
		// Bottom boundary face n.
		if s.cfg.Bottom.Kind == BCFreeDrainage {
			diag[n-1] += dkCell[n-1] / s.dz
		}
		// Capacity term.
		for i := 0; i < n; i++ {
			diag[i] += capC[i] / dt
		}
	}

	converged := false
	iters := 0
	for iter := 0; iter < maxIter; iter++ {
		assemble(true)
		if maxAbs(res) < resTol {
			converged = true
			iters = iter
			break
		}
		iters = iter + 1
		// Solve J*dh = -res.
		copy(dh, res)
		for i := range dh {
			dh[i] = -dh[i]
		}
		if !thomas(lower, diag, upper, dh) {
			return nil, errs.New(errs.CodeNotConverged, "singular Jacobian in Newton iteration")
		}
		// Damped update: shrink the step until the residual decreases.
		base := maxAbs(res)
		lambda := 1.0
		accepted := false
		for halve := 0; halve < 25; halve++ {
			ok := true
			for i := 0; i < n; i++ {
				hNew := h[i] + lambda*dh[i]
				if math.IsNaN(hNew) || hNew < headFloor {
					ok = false
					break
				}
			}
			if !ok {
				lambda *= 0.5
				continue
			}
			trial := make([]float64, n)
			for i := 0; i < n; i++ {
				trial[i] = h[i] + lambda*dh[i]
			}
			saved := h
			h = trial
			assemble(false)
			r := maxAbs(res)
			h = saved
			if r < base {
				h = trial
				accepted = true
				break
			}
			lambda *= 0.5
		}
		if !accepted {
			return nil, errs.New(errs.CodeNotConverged,
				"Newton line search failed to reduce the residual")
		}
	}
	if !converged {
		return nil, errs.Newf(errs.CodeNotConverged,
			"Newton iteration did not converge in %d iterations (residual %.3e)", maxIter, maxAbs(res))
	}

	// Final fluxes and state at the converged head field.
	assemble(false)
	after := &State{
		Head:      make([]float64, n),
		Theta:     make([]float64, n),
		Time:      st.Time + dt,
		CumTop:    st.CumTop + dt*flux[0],
		CumBottom: st.CumBottom + dt*flux[n],
	}
	copy(after.Head, h)
	copy(after.Theta, theta)

	// Hard invariant: water content must stay inside [thetaR, thetaS].
	// Violations are reported, never silently clipped.
	for i, th := range after.Theta {
		if math.IsNaN(th) || th < p.ThetaR-1e-12 || th > p.ThetaS+1e-12 {
			return nil, errs.Newf(errs.CodeThetaOutOfRange,
				"theta[%d]=%g left [thetaR=%g, thetaS=%g] after step", i, th, p.ThetaR, p.ThetaS)
		}
	}

	before := State{
		Head:      append([]float64(nil), st.Head...),
		Theta:     append([]float64(nil), st.Theta...),
		Time:      st.Time,
		CumTop:    st.CumTop,
		CumBottom: st.CumBottom,
	}
	out := &StepOutcome{
		Before:        before,
		After:         *after,
		TopFlux:       flux[0],
		BottomFlux:    flux[n],
		StorageBefore: s.Storage(st),
		Iterations:    iters,
	}
	out.StorageAfter = s.Storage(after)
	out.MassResidual = (out.StorageAfter - out.StorageBefore) - dt*(flux[0]-flux[n])
	return out, nil
}

// faceFluxes fills F[0..n] for the current head field.
func (s *Solver) faceFluxes(h, kCell, flux []float64) {
	n := s.cfg.Cells
	for j := 1; j < n; j++ {
		kf, _, _ := s.faceK(kCell[j-1], kCell[j])
		flux[j] = kf * (1.0 - (h[j]-h[j-1])/s.dz)
	}
	// Top face.
	switch s.cfg.Top.Kind {
	case BCDirichlet:
		kTop := s.cfg.Soil.K(s.cfg.Top.Head)
		kf, _, _ := s.faceK(kTop, kCell[0])
		flux[0] = kf * (1.0 - (h[0]-s.cfg.Top.Head)/(s.dz/2.0))
	case BCNeumann:
		flux[0] = s.cfg.Top.Flux
	}
	// Bottom face.
	switch s.cfg.Bottom.Kind {
	case BCFreeDrainage:
		flux[n] = kCell[n-1]
	case BCNoFlow:
		flux[n] = 0
	}
}

// faceK returns the intercell conductivity and its derivatives w.r.t.
// each side's cell conductivity.
func (s *Solver) faceK(ka, kb float64) (k, dKa, dKb float64) {
	if s.cfg.Averaging == Arithmetic {
		return 0.5 * (ka + kb), 0.5, 0.5
	}
	sum := ka + kb
	if sum <= 0 {
		return 0, 0, 0
	}
	return 2 * ka * kb / sum, 2 * kb * kb / (sum * sum), 2 * ka * ka / (sum * sum)
}

func maxAbs(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		if a := math.Abs(x); a > m {
			m = a
		}
	}
	return m
}

// thomas solves a tridiagonal system in place; b is overwritten with the
// solution. Returns false on a (near-)singular pivot.
func thomas(lower, diag, upper, b []float64) bool {
	n := len(b)
	for i := 1; i < n; i++ {
		if diag[i-1] == 0 {
			return false
		}
		w := lower[i] / diag[i-1]
		diag[i] -= w * upper[i-1]
		b[i] -= w * b[i-1]
	}
	if diag[n-1] == 0 {
		return false
	}
	b[n-1] /= diag[n-1]
	for i := n - 2; i >= 0; i-- {
		if diag[i] == 0 {
			return false
		}
		b[i] = (b[i] - upper[i]*b[i+1]) / diag[i]
	}
	return true
}
