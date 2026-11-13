# Pulse durations of the CaOCH3 preparation tree (Table tab:caoch3_pulses, Appendix app:caoch3_pulses).
# Inputs (stored outputs, nothing recomputed):
#   bend_dipole_b3lyp.out     -> mu8, the nu8 fundamental transition dipole (B3LYP, double harmonic)
#   check_tree_measured.out   -> couplings d/mu8 of P1-P7 ("Tree lines") and the Rabi-frequency bound of each
#                                pulse from its nearest spin component ("Hyperfine-resolved check")
# All seven tones share one intensity I, chosen so that P7, the pulse with the tightest bound, has
# Omega/2pi = 10 kHz (the value chosen in Appendix app:caoch3_pulses). Then Omega_n = |d_n| E / hbar,
# E = sqrt(2 I / (c eps0)), and a pulse of area Theta lasts Theta / Omega_n.
import re
h, c, eps0, debye = 6.62607015e-34, 2.99792458e8, 8.8541878128e-12, 3.33564095e-30
F_REF, P_REF = 10e3, 'P7'                       # Hz

mu8 = float(re.search(r'MU8_DEBYE = ([0-9.]+)', open('bend_dipole_b3lyp.out').read()).group(1))
out = open('check_tree_measured.out').read()
dmu = {p: float(x) for p, x in re.findall(r'(P\d): \S+\s*->\S+\s+q=[+-]\d dN=[+-]\d\s+d=([+-][0-9.]+) mu8', out)}
bound = {p: float(x)*1e3 for p, x in re.findall(r'(P\d): \|d\|=.*?Omega/2pi <=\s*([0-9.]+) kHz', out)}
assert len(dmu) == 7 and len(bound) == 7

E = h*F_REF/(abs(dmu[P_REF])*mu8*debye)        # V/m
I = 0.5*c*eps0*E**2                            # W/m^2
print(f"mu8 = {mu8:.4f} D; reference {P_REF} at Omega/2pi = {F_REF/1e3:.0f} kHz")
print(f"field amplitude E = {E:.1f} V/m, intensity I = c eps0 E^2/2 = {I:.2f} W/m^2 = {I/10:.3f} mW/cm^2")
f, tpi = {}, {}
for p in sorted(dmu):
    f[p] = abs(dmu[p])*mu8*debye*E/h
    tpi[p] = 1/(2*f[p])
    print(f"  {p}: |d| = {abs(dmu[p]):.4f} mu8  Omega/2pi = {f[p]/1e3:5.1f} kHz (bound {bound[p]/1e3:7.1f} kHz)  "
          f"pi pulse {tpi[p]*1e6:5.1f} us  leakage ~ {1e-3*(f[p]/bound[p])**2:.1e}")
    assert f[p] <= bound[p]
dmin = float(re.search(r'overall closest: P\d ([0-9.]+) cm-1', out).group(1))*c*100     # Hz, nearest competing rovibrational line
fmax = max(f.values())
print(f"largest Omega/2pi = {fmax/1e3:.1f} kHz; nearest competing rovibrational line {dmin/1e9:.2f} GHz -> "
      f"(Omega/Delta)^2 <= {(fmax/dmin)**2:.1e};  largest hyperfine leakage {max(1e-3*(f[p]/bound[p])**2 for p in f):.1e}")
us = lambda t: f"{t*1e6:.0f} us"
print("Totals (pulses in sequence):")
print(f"  (b) L30: P1 P2 P5 = {us(tpi['P1'] + tpi['P2'] + tpi['P5'])};  L03: P1 P2 P4 = {us(tpi['P1'] + tpi['P2'] + tpi['P4'])}")
print(f"  (c) L21: P1 P3 P7 = {us(tpi['P1'] + tpi['P3'] + tpi['P7'])};  L12: P1 P3 P6 = {us(tpi['P1'] + tpi['P3'] + tpi['P6'])}")
eq = {p: tpi[p]*(0.5 if p in ('P2', 'P4', 'P6') else 1.0) for p in tpi}   # every c = 1/2: P2, P4, P6 are pi/2 pulses
ser = sum(eq.values())
par = eq['P1'] + eq['P2'] + eq['P3'] + max(eq['P4'] + eq['P5'], eq['P6'] + eq['P7'])
print(f"  (a) every c = 1/2: one after another {us(ser)}; with P6, P7 overlapping P4, P5 {us(par)}")
print(f"  (a) any target, upper bound (all seven as pi pulses, in sequence): {us(sum(tpi.values()))}")
