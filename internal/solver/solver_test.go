package solver

import (
	"math"
	"testing"

	"github.com/example/unsatflow/internal/vg"
)

var sand = vg.Params{ThetaR: 0.045, ThetaS: 0.43, Alpha: 14.5, N: 2.68, Ks: 8.25e-5}

// pondingConfig returns the standard ponded-infiltration configuration.
func pondingConfig(head, ks float64) Config {
	p := sand
	p.Ks = ks
	return Config{
		Soil:   p,
		Length: 1.0,
		Cells:  100,
		Top:    Boundary{Kind: BCDirichlet, Head: head},
		Bottom: Boundary{Kind: BCFreeDrainage},
	}
}

func uniformState(t *testing.T, s *Solver, theta float64) *State {
	t.Helper()
	th := make([]float64, s.Cells())
	for i := range th {
		th[i] = theta
	}
	st, e := s.NewState(th)
	if e != nil {
		t.Fatalf("NewState: %v", e)
	}
	return st
}

// runSteps marches a state forward, failing the test on any error.
func runSteps(t *testing.T, s *Solver, st *State, dt float64, steps int) (*State, []StepOutcome) {
	t.Helper()
	outs := make([]StepOutcome, 0, steps)
	for k := 0; k < steps; k++ {
		out, e := s.Step(st, dt)
		if e != nil {
			t.Fatalf("step %d: %v", k, e)
		}
		outs = append(outs, *out)
		st = &out.After
	}
	return st, outs
}

// frontDepth estimates the wetting-front position: depth of the deepest
// cell whose water content exceeds the initial value by 10% of the
// available range.
func frontDepth(s *Solver, theta []float64, thetaInit float64) float64 {
	threshold := thetaInit + 0.1*(s.cfg.Soil.ThetaS-thetaInit)
	depth := 0.0
	for i, th := range theta {
		if th > threshold {
			depth = (float64(i) + 0.5) * s.Dz()
		}
	}
	return depth
}

func TestMassClosureSingleStep(t *testing.T) {
	s, e := New(pondingConfig(0.10, sand.Ks))
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	st := uniformState(t, s, 0.10)
	out, e2 := s.Step(st, 10.0)
	if e2 != nil {
		t.Fatalf("Step: %v", e2)
	}
	if math.Abs(out.MassResidual) > 1e-10 {
		t.Fatalf("mass residual %g exceeds 1e-10", out.MassResidual)
	}
	// Storage change must equal dt*(in-out) through the fluxes.
	got := out.StorageAfter - out.StorageBefore
	want := 10.0 * (out.TopFlux - out.BottomFlux)
	if math.Abs(got-want) > 1e-10 {
		t.Fatalf("dStorage %g != dt*(Ftop-Fbot) %g", got, want)
	}
	if got <= 0 {
		t.Fatalf("ponding infiltration should increase storage, got dStorage %g", got)
	}
}

func TestMassClosureFullRun(t *testing.T) {
	s, e := New(pondingConfig(0.10, sand.Ks))
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	st := uniformState(t, s, 0.10)
	st, outs := runSteps(t, s, st, 10.0, 360)
	for k, o := range outs {
		if math.Abs(o.MassResidual) > 1e-10 {
			t.Fatalf("step %d mass residual %g exceeds 1e-10", k, o.MassResidual)
		}
	}
	// Cumulative closure: storage(t) - storage(0) == cumTop - cumBottom.
	s0 := 0.10 * 1.0 // uniform theta * length
	dS := s.Storage(st) - s0
	dF := st.CumTop - st.CumBottom
	if math.Abs(dS-dF) > 1e-9 {
		t.Fatalf("cumulative closure violated: dStorage %g vs net flux %g", dS, dF)
	}
	if st.CumTop <= 0 {
		t.Fatalf("expected positive cumulative infiltration, got %g", st.CumTop)
	}
}

func TestPondingHeadDeepensFront(t *testing.T) {
	shallow, e := New(pondingConfig(0.05, sand.Ks))
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	deep, e := New(pondingConfig(0.50, sand.Ks))
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	stA := uniformState(t, shallow, 0.10)
	stB := uniformState(t, deep, 0.10)
	stA, _ = runSteps(t, shallow, stA, 10.0, 360)
	stB, _ = runSteps(t, deep, stB, 10.0, 360)

	fA := frontDepth(shallow, stA.Theta, 0.10)
	fB := frontDepth(deep, stB.Theta, 0.10)
	if fB <= fA {
		t.Fatalf("larger ponding head should deepen front: head 0.05 -> %g m, head 0.50 -> %g m", fA, fB)
	}
	sA, sB := shallow.Storage(stA), deep.Storage(stB)
	if sB <= sA {
		t.Fatalf("larger ponding head should store more water: %g vs %g", sB, sA)
	}
}

func TestLowerKsSlowsFront(t *testing.T) {
	fast, e := New(pondingConfig(0.10, sand.Ks))
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	slow, e := New(pondingConfig(0.10, sand.Ks/10))
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	stF := uniformState(t, fast, 0.10)
	stS := uniformState(t, slow, 0.10)
	stF, _ = runSteps(t, fast, stF, 10.0, 360)
	stS, _ = runSteps(t, slow, stS, 10.0, 360)

	fF := frontDepth(fast, stF.Theta, 0.10)
	fS := frontDepth(slow, stS.Theta, 0.10)
	if fS >= fF {
		t.Fatalf("10x lower Ks should slow the front: Ks -> %g m, Ks/10 -> %g m", fF, fS)
	}
	if fS > 0.6*fF {
		t.Fatalf("front slowdown too weak for 10x Ks reduction: %g vs %g", fS, fF)
	}
}

func TestHydrostaticNoFlowStaysPut(t *testing.T) {
	cfg := Config{
		Soil:   sand,
		Length: 1.0,
		Cells:  100,
		Top:    Boundary{Kind: BCNeumann, Flux: 0},
		Bottom: Boundary{Kind: BCNoFlow},
	}
	s, e := New(cfg)
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	// Hydrostatic: dh/dz = 1 (z downward), h(z) = -1.5 + z.
	n := s.Cells()
	theta := make([]float64, n)
	for i := 0; i < n; i++ {
		z := (float64(i) + 0.5) * s.Dz()
		theta[i] = sand.Theta(-1.5 + z)
	}
	st, e := s.NewState(theta)
	if e != nil {
		t.Fatalf("NewState: %v", e)
	}
	s0 := s.Storage(st)
	st, _ = runSteps(t, s, st, 60.0, 200) // 3.3 hours
	for i, th := range st.Theta {
		if math.Abs(th-theta[i]) > 1e-10 {
			t.Fatalf("cell %d drifted: theta %g -> %g", i, theta[i], th)
		}
	}
	if math.Abs(s.Storage(st)-s0) > 1e-10 {
		t.Fatalf("storage drifted: %g -> %g", s0, s.Storage(st))
	}
}

func TestThetaStaysWithinBounds(t *testing.T) {
	s, e := New(pondingConfig(0.10, sand.Ks))
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	st := uniformState(t, s, 0.10)
	st, outs := runSteps(t, s, st, 10.0, 360)
	for k, o := range outs {
		for i, th := range o.After.Theta {
			if th < sand.ThetaR || th > sand.ThetaS {
				t.Fatalf("step %d cell %d: theta %g outside [%g, %g]", k, i, th, sand.ThetaR, sand.ThetaS)
			}
		}
	}
}

func TestHarmonicVsArithmeticDiffer(t *testing.T) {
	cfgH := pondingConfig(0.10, sand.Ks)
	cfgA := pondingConfig(0.10, sand.Ks)
	cfgA.Averaging = Arithmetic
	sh, e := New(cfgH)
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	sa, e := New(cfgA)
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	stH := uniformState(t, sh, 0.10)
	stA := uniformState(t, sa, 0.10)
	stH, _ = runSteps(t, sh, stH, 10.0, 120)
	stA, _ = runSteps(t, sa, stA, 10.0, 120)

	maxDiff := 0.0
	for i := range stH.Theta {
		if d := math.Abs(stH.Theta[i] - stA.Theta[i]); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff < 1e-4 {
		t.Fatalf("harmonic and arithmetic averaging indistinguishable: max |dTheta| = %g", maxDiff)
	}
	// The harmonic mean throttles flow into dry cells; the arithmetic
	// mean must overestimate infiltration into the dry column.
	if sa.Storage(stA) <= sh.Storage(stH) {
		t.Fatalf("arithmetic averaging should infiltrate faster here: %g vs %g",
			sa.Storage(stA), sh.Storage(stH))
	}
}

func TestFreeDrainageBottomDrains(t *testing.T) {
	// Sanity on the bottom boundary: with free drainage and no inflow,
	// a wet column must lose water.
	cfg := Config{
		Soil:   sand,
		Length: 1.0,
		Cells:  100,
		Top:    Boundary{Kind: BCNeumann, Flux: 0},
		Bottom: Boundary{Kind: BCFreeDrainage},
	}
	s, e := New(cfg)
	if e != nil {
		t.Fatalf("New: %v", e)
	}
	st := uniformState(t, s, 0.30)
	s0 := s.Storage(st)
	st, _ = runSteps(t, s, st, 60.0, 60)
	if s.Storage(st) >= s0 {
		t.Fatalf("free drainage should reduce storage: %g -> %g", s0, s.Storage(st))
	}
	if st.CumBottom <= 0 {
		t.Fatalf("expected positive cumulative drainage, got %g", st.CumBottom)
	}
}
