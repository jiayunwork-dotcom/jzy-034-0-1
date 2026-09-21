package job

import (
	"testing"

	"github.com/example/unsatflow/internal/solver"
	"github.com/example/unsatflow/internal/vg"
)

func testSpec(steps int) Spec {
	thetaInit := 0.10
	return Spec{
		Column:  ColumnSpec{Length: 1.0, Cells: 100},
		Soil:    vg.Params{ThetaR: 0.045, ThetaS: 0.43, Alpha: 14.5, N: 2.68, Ks: 8.25e-5},
		Initial: ProfileSpec{Uniform: &thetaInit},
		Boundary: BoundarySpec{
			Top:    solver.Boundary{Kind: solver.BCDirichlet, Head: 0.10},
			Bottom: solver.Boundary{Kind: solver.BCFreeDrainage},
		},
		Time: TimeSpec{Dt: 10.0, Total: 10.0 * float64(steps)},
	}
}

// The single-step endpoint and the full-run endpoint share one solver:
// identical initial states must produce identical profiles.
func TestSingleStepMatchesFullRun(t *testing.T) {
	full, e := RunFull(testSpec(1))
	if e != nil {
		t.Fatalf("RunFull: %v", e)
	}
	thetaInit := 0.10
	step, e := RunStep(StepSpec{
		Column:  ColumnSpec{Length: 1.0, Cells: 100},
		Soil:    vg.Params{ThetaR: 0.045, ThetaS: 0.43, Alpha: 14.5, N: 2.68, Ks: 8.25e-5},
		State:   ProfileSpec{Uniform: &thetaInit},
		Boundary: BoundarySpec{
			Top:    solver.Boundary{Kind: solver.BCDirichlet, Head: 0.10},
			Bottom: solver.Boundary{Kind: solver.BCFreeDrainage},
		},
		Dt: 10.0,
	})
	if e != nil {
		t.Fatalf("RunStep: %v", e)
	}
	snap := full.Snapshots[1]
	if len(snap.Theta) != len(step.After.Theta) {
		t.Fatalf("profile length mismatch: %d vs %d", len(snap.Theta), len(step.After.Theta))
	}
	for i := range snap.Theta {
		if snap.Theta[i] != step.After.Theta[i] {
			t.Fatalf("cell %d: full-run theta %g != single-step theta %g", i, snap.Theta[i], step.After.Theta[i])
		}
		if snap.Head[i] != step.After.Head[i] {
			t.Fatalf("cell %d: full-run head %g != single-step head %g", i, snap.Head[i], step.After.Head[i])
		}
	}
	if snap.Storage != step.StorageAfter {
		t.Fatalf("storage mismatch: %g vs %g", snap.Storage, step.StorageAfter)
	}
}

// Chaining single steps must reproduce the full run step by step.
func TestChainedStepsMatchFullRun(t *testing.T) {
	full, e := RunFull(testSpec(5))
	if e != nil {
		t.Fatalf("RunFull: %v", e)
	}
	theta := make([]float64, 100)
	for i := range theta {
		theta[i] = 0.10
	}
	for k := 0; k < 5; k++ {
		out, e := RunStep(StepSpec{
			Column:  ColumnSpec{Length: 1.0, Cells: 100},
			Soil:    vg.Params{ThetaR: 0.045, ThetaS: 0.43, Alpha: 14.5, N: 2.68, Ks: 8.25e-5},
			State:   ProfileSpec{Values: theta},
			Boundary: BoundarySpec{
				Top:    solver.Boundary{Kind: solver.BCDirichlet, Head: 0.10},
				Bottom: solver.Boundary{Kind: solver.BCFreeDrainage},
			},
			Dt: 10.0,
		})
		if e != nil {
			t.Fatalf("RunStep %d: %v", k, e)
		}
		copy(theta, out.After.Theta)
		snap := full.Snapshots[k+1]
		for i := range theta {
			if theta[i] != snap.Theta[i] {
				t.Fatalf("step %d cell %d: chained %g != full %g", k, i, theta[i], snap.Theta[i])
			}
		}
	}
}

func TestRunFullMassClosureAcrossSnapshots(t *testing.T) {
	res, e := RunFull(testSpec(60))
	if e != nil {
		t.Fatalf("RunFull: %v", e)
	}
	first, last := res.Snapshots[0], res.Snapshots[len(res.Snapshots)-1]
	dS := last.Storage - first.Storage
	dF := last.CumTopFlux - last.CumBottomFlux
	if d := dS - dF; d < 0 {
		d = -d
		if d > 1e-9 {
			t.Fatalf("closure violated: dStorage %g vs net flux %g", dS, dF)
		}
	} else if d > 1e-9 {
		t.Fatalf("closure violated: dStorage %g vs net flux %g", dS, dF)
	}
	for k, snap := range res.Snapshots[1:] {
		if snap.MassResidual > 1e-10 || snap.MassResidual < -1e-10 {
			t.Fatalf("snapshot %d mass residual %g", k+1, snap.MassResidual)
		}
		if snap.Storage < res.Snapshots[k].Storage {
			t.Fatalf("ponded infiltration: storage decreased at snapshot %d", k+1)
		}
	}
}
