package vg

import (
	"math"
	"testing"
)

var sand = Params{ThetaR: 0.045, ThetaS: 0.43, Alpha: 14.5, N: 2.68, Ks: 8.25e-5}

func TestMualemBinding(t *testing.T) {
	if got, want := sand.M(), 1.0-1.0/sand.N; got != want {
		t.Fatalf("m = %g, want 1 - 1/n = %g", got, want)
	}
}

func TestSaturationEndpoints(t *testing.T) {
	if se := sand.Se(0); se != 1 {
		t.Fatalf("Se(0) = %g, want 1", se)
	}
	if se := sand.Se(0.3); se != 1 {
		t.Fatalf("Se(positive head) = %g, want 1", se)
	}
	if th := sand.Theta(0); th != sand.ThetaS {
		t.Fatalf("Theta(0) = %g, want thetaS = %g", th, sand.ThetaS)
	}
	if k := sand.K(0); k != sand.Ks {
		t.Fatalf("K(0) = %g, want Ks = %g", k, sand.Ks)
	}
	if se := sand.Se(-100); se >= 1 || se <= 0 {
		t.Fatalf("Se(-100) = %g, want in (0,1)", se)
	}
}

func TestThetaBoundedEverywhere(t *testing.T) {
	for _, h := range []float64{-1e-4, -0.01, -0.1, -1, -10, -100, -1e4} {
		th := sand.Theta(h)
		if th < sand.ThetaR || th > sand.ThetaS {
			t.Fatalf("Theta(%g) = %g outside [thetaR, thetaS]", h, th)
		}
		if k := sand.K(h); k <= 0 || k > sand.Ks {
			t.Fatalf("K(%g) = %g outside (0, Ks]", h, k)
		}
	}
}

func TestHeadFromThetaRoundTrip(t *testing.T) {
	for _, th := range []float64{0.05, 0.08, 0.15, 0.3, 0.429} {
		h := sand.HeadFromTheta(th)
		got := sand.Theta(h)
		if math.Abs(got-th) > 1e-12 {
			t.Fatalf("round trip theta=%g -> h=%g -> theta=%g", th, h, got)
		}
	}
}

func TestCapacityAndDKDhFiniteDifference(t *testing.T) {
	for _, h := range []float64{-0.05, -0.3, -1.0, -5.0} {
		d := 1e-7
		cNum := (sand.Theta(h+d) - sand.Theta(h-d)) / (2 * d)
		if c := sand.Capacity(h); math.Abs(c-cNum) > 1e-6*math.Max(1, math.Abs(cNum)) {
			t.Fatalf("Capacity(%g) = %g, fd = %g", h, c, cNum)
		}
		kNum := (sand.K(h+d) - sand.K(h-d)) / (2 * d)
		if k := sand.DKDh(h); math.Abs(k-kNum) > 1e-5*math.Max(1e-12, math.Abs(kNum)) {
			t.Fatalf("DKDh(%g) = %g, fd = %g", h, k, kNum)
		}
	}
}

func TestValidate(t *testing.T) {
	ok := sand
	if e := ok.Validate(); e != nil {
		t.Fatalf("valid params rejected: %v", e)
	}
	cases := []struct {
		name string
		p    Params
		code string
	}{
		{"n<=1", Params{ThetaR: 0.05, ThetaS: 0.4, Alpha: 1, N: 1.0, Ks: 1e-5}, "INVALID_VG_N"},
		{"alpha<=0", Params{ThetaR: 0.05, ThetaS: 0.4, Alpha: 0, N: 2, Ks: 1e-5}, "INVALID_VG_ALPHA"},
		{"thetaR>=thetaS", Params{ThetaR: 0.4, ThetaS: 0.4, Alpha: 1, N: 2, Ks: 1e-5}, "INVALID_THETA_RANGE"},
		{"ks<=0", Params{ThetaR: 0.05, ThetaS: 0.4, Alpha: 1, N: 2, Ks: 0}, "INVALID_KS"},
	}
	for _, tc := range cases {
		if e := tc.p.Validate(); e == nil || e.Code != tc.code {
			t.Fatalf("%s: got %v, want code %s", tc.name, e, tc.code)
		}
	}
}
