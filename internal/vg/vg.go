// Package vg implements the van Genuchten water-retention curve and the
// Mualem-van Genuchten unsaturated hydraulic-conductivity model.
//
// Constitutive forms (h = pressure head, negative above the water table):
//
//	Se(h) = [1 + (alpha*|h|)^n]^(-m),   m = 1 - 1/n        (h < 0; Se = 1 for h >= 0)
//	theta(h) = thetaR + (thetaS - thetaR) * Se
//	K(h)  = Ks * Se^(1/2) * [1 - (1 - Se^(1/m))^m]^2
//
// The binding m = 1 - 1/n and the 1/2 reduction exponent are fixed model
// constants; they are echoed by the /api/v1/constitutive endpoint.
package vg

import (
	"math"

	"github.com/example/unsatflow/internal/errs"
)

// Params holds the van Genuchten / Mualem model parameters (SI units:
// heads in metres, Ks in m/s, water contents dimensionless).
type Params struct {
	ThetaR float64 `json:"thetaR"` // residual water content [-]
	ThetaS float64 `json:"thetaS"` // saturated water content [-]
	Alpha  float64 `json:"alpha"`  // air-entry parameter [1/m]
	N      float64 `json:"n"`      // pore-size distribution index [-]
	Ks     float64 `json:"ks"`     // saturated hydraulic conductivity [m/s]
}

// M returns the Mualem binding m = 1 - 1/n.
func (p Params) M() float64 { return 1.0 - 1.0/p.N }

// Validate rejects physically meaningless parameter sets before any
// time stepping starts.
func (p Params) Validate() *errs.Error {
	if !(p.N > 1.0) {
		return errs.Newf(errs.CodeInvalidVGN, "van Genuchten n must be > 1, got %g", p.N)
	}
	if !(p.Alpha > 0) {
		return errs.Newf(errs.CodeInvalidVGAlpha, "air-entry parameter alpha must be > 0, got %g", p.Alpha)
	}
	if !(p.ThetaR < p.ThetaS) {
		return errs.Newf(errs.CodeInvalidThetaRange,
			"residual water content thetaR (%g) must be smaller than saturated thetaS (%g)", p.ThetaR, p.ThetaS)
	}
	if !(p.Ks > 0) {
		return errs.Newf(errs.CodeInvalidKs, "saturated conductivity ks must be > 0, got %g", p.Ks)
	}
	return nil
}

// Se returns the effective saturation.
func (p Params) Se(h float64) float64 {
	if h >= 0 {
		return 1.0
	}
	a := p.Alpha * (-h)
	return math.Pow(1.0+math.Pow(a, p.N), -p.M())
}

// Theta returns the volumetric water content thetaR + (thetaS-thetaR)*Se.
func (p Params) Theta(h float64) float64 {
	return p.ThetaR + (p.ThetaS-p.ThetaR)*p.Se(h)
}

// dSeDh is the derivative of effective saturation w.r.t. pressure head.
func (p Params) dSeDh(h float64) float64 {
	if h >= 0 {
		return 0.0
	}
	m := p.M()
	a := p.Alpha * (-h)
	// Se = (1 + a^n)^(-m), da/dh = -alpha
	// dSe/dh = m*n*alpha*a^(n-1)*(1+a^n)^(-m-1)
	return m * p.N * p.Alpha * math.Pow(a, p.N-1.0) * math.Pow(1.0+math.Pow(a, p.N), -m-1.0)
}

// Capacity returns the specific moisture capacity C = d(theta)/dh [1/m].
func (p Params) Capacity(h float64) float64 {
	return (p.ThetaS - p.ThetaR) * p.dSeDh(h)
}

// K returns the Mualem-van Genuchten unsaturated conductivity [m/s].
func (p Params) K(h float64) float64 {
	se := p.Se(h)
	if se >= 1.0 {
		return p.Ks
	}
	m := p.M()
	term := 1.0 - math.Pow(1.0-math.Pow(se, 1.0/m), m)
	return p.Ks * math.Sqrt(se) * term * term
}

// DKDh returns dK/dh, needed by the Newton Jacobian.
func (p Params) DKDh(h float64) float64 {
	if h >= 0 {
		return 0.0
	}
	m := p.M()
	se := p.Se(h)
	root := math.Sqrt(se)
	u := 1.0 - math.Pow(se, 1.0/m) // 1 - Se^(1/m)
	g := 1.0 - math.Pow(u, m)      // 1 - (1 - Se^(1/m))^m
	// dg/dSe = u^(m-1) * Se^(1/m - 1)
	dg := math.Pow(u, m-1.0) * math.Pow(se, 1.0/m-1.0)
	// K = Ks * sqrt(Se) * g^2
	dKdSe := p.Ks * (g*g/(2.0*root) + 2.0*root*g*dg)
	return dKdSe * p.dSeDh(h)
}

// HeadFromTheta inverts the retention curve; used to build initial states
// (e.g. hydrostatic profiles) from water contents.
func (p Params) HeadFromTheta(theta float64) float64 {
	se := (theta - p.ThetaR) / (p.ThetaS - p.ThetaR)
	if se >= 1.0 {
		return 0.0
	}
	if se <= 0.0 {
		return math.Inf(-1)
	}
	m := p.M()
	return -math.Pow(math.Pow(se, -1.0/m)-1.0, 1.0/p.N) / p.Alpha
}
