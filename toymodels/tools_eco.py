import warnings
import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
from tools import Config

def foodweb(vars, cg, fiH, fiU):
    C, H, U = vars
    dCdt =  (-cg.muC + cg.eH * fiH(vars, cg) + cg.eU * fiU(vars, cg)) * C
    dHdt = (cg.rH * (1 - H/cg.K) - cg.muH)*H - fiH(vars, cg) * C
    dUdt = - cg.nu * U - fiU(vars, cg) * C
    return [dCdt, dHdt, dUdt]

def holling_ii_H(vars, cg):
    C, H, U = vars
    fiH = cg.aH * H/(1 + cg.aH * cg.ThH * H + cg.aU * cg.ThU * U + cg.b * C)
    return fiH
def holling_ii_U(vars, cg):
    C, H, U = vars
    fiU = cg.aU * U/(1 + cg.aU * cg.ThU * U + cg.aH * cg.ThH * H + cg.b * C)
    return fiU

def holling_iii_H(vars, cg):
    C, H, U = vars
    fiH = cg.aH * (H**cg.wH)/(1 + cg.aH * cg.ThH * (H**cg.wH) + cg.aU * cg.ThU * (U**cg.wU) + cg.b * C)
    return fiH
def holling_iii_U(vars, cg):
    C, H, U = vars
    fiU = cg.aU * (U**cg.wU)/(1 + cg.aU * cg.ThU * (U**cg.wU) + cg.aH * cg.ThH * (H**cg.wH) + cg.b * C)
    return fiU

def foodweb_jacobian_holling_ii(vars, cg):
    C, H, U = vars
    D = 1 + cg.aH * cg.ThH * H + cg.aU * cg.ThU * U + cg.b * C  # common denominator of fiH and fiU
    fiH = cg.aH * H/D
    fiU = cg.aU * U/D
    # derivatives of the functional responses
    dfiH_dC = -cg.aH * cg.b * H/D**2
    dfiH_dH = cg.aH * (1 + cg.aU * cg.ThU * U + cg.b * C)/D**2
    dfiH_dU = -cg.aH * cg.aU * cg.ThU * H/D**2
    dfiU_dC = -cg.aU * cg.b * U/D**2
    dfiU_dH = -cg.aU * cg.aH * cg.ThH * U/D**2
    dfiU_dU = cg.aU * (1 + cg.aH * cg.ThH * H + cg.b * C)/D**2
    dCdi = [-cg.muC + cg.eH * fiH + cg.eU * fiU + C * (cg.eH * dfiH_dC + cg.eU * dfiU_dC),
            C * (cg.eH * dfiH_dH + cg.eU * dfiU_dH),
            C * (cg.eH * dfiH_dU + cg.eU * dfiU_dU)]
    dHdi = [-fiH - C * dfiH_dC,
            cg.rH * (1 - 2 * H/cg.K) - cg.muH - C * dfiH_dH,
            -C * dfiH_dU]
    dUdi = [-fiU - C * dfiU_dC,
            -C * dfiU_dH,
            -cg.nu - C * dfiU_dU]
    return np.array([dCdi, dHdi, dUdi])

def foodweb_jacobian_holling_iii(vars, cg):
    C, H, U = vars
    Hw = H**cg.wH
    Uw = U**cg.wU
    dHw_dH = cg.wH * H**(cg.wH - 1)   # derivative of H**wH
    dUw_dU = cg.wU * U**(cg.wU - 1)   # derivative of U**wU
    D = 1 + cg.aH * cg.ThH * Hw + cg.aU * cg.ThU * Uw + cg.b * C  # common denominator of fiH and fiU
    fiH = cg.aH * Hw/D
    fiU = cg.aU * Uw/D
    # derivatives of the functional responses
    dfiH_dC = -cg.aH * cg.b * Hw/D**2
    dfiH_dH = cg.aH * dHw_dH * (1 + cg.aU * cg.ThU * Uw + cg.b * C)/D**2
    dfiH_dU = -cg.aH * cg.aU * cg.ThU * Hw * dUw_dU/D**2
    dfiU_dC = -cg.aU * cg.b * Uw/D**2
    dfiU_dH = -cg.aU * cg.aH * cg.ThH * Uw * dHw_dH/D**2
    dfiU_dU = cg.aU * dUw_dU * (1 + cg.aH * cg.ThH * Hw + cg.b * C)/D**2
    dCdi = [-cg.muC + cg.eH * fiH + cg.eU * fiU + C * (cg.eH * dfiH_dC + cg.eU * dfiU_dC),
            C * (cg.eH * dfiH_dH + cg.eU * dfiU_dH),
            C * (cg.eH * dfiH_dU + cg.eU * dfiU_dU)]
    dHdi = [-fiH - C * dfiH_dC,
            cg.rH * (1 - 2 * H/cg.K) - cg.muH - C * dfiH_dH,
            -C * dfiH_dU]
    dUdi = [-fiU - C * dfiU_dC,
            -C * dfiU_dH,
            -cg.nu - C * dfiU_dU]
    return np.array([dCdi, dHdi, dUdi])

def followpars(system,fiH, fiU, commonpars, varypars, varyvals, initeq):
    eqlist = ()
    ef_varypar = ()
    pars = commonpars.clone_me()
    for i in varyvals:
        pars[varypars] = i
        eq = root(lambda y: system(y, pars, fiH, fiU), initeq)
        if eq.success:
            eqlist += (eq.x,)
            ef_varypar += (i,)
            initeq = eq.x
        else:
            warnings.warn(f"Root finding failed for {varypars}={i}.")
    return ef_varypar, np.array(eqlist)


def followpars_with_eqnature(system, fiH, fiU, systemJ, commonpars, varypars,
                              varyvals, initeq):
    eqlist = ()
    ef_varypar = ()
    eqnaturelist = ()
    eqnatcollist = ()
    pars = commonpars.clone_me()
    for i in varyvals:
        pars[varypars] = i
        eq = root(lambda y: system(y, pars, fiH, fiU), initeq)
        if eq.success:
            eqlist += (eq.x,)
            ef_varypar += (i,)
            initeq = eq.x
            # Determine the nature of the equilibrium point
            J = systemJ(eq.x, pars)
            eigenvalues = np.linalg.eigvals(J)
            re, im = np.real(eigenvalues), np.imag(eigenvalues)
            if np.all(re < 0):
                label, col = ('StableNode', 'k') if np.all(im == 0) else ('StableFocus', 'lightgray')
            elif np.any(re > 0):
                if np.any((re > 0) & (np.abs(im) > 1e-12)):
                    label, col = ('UnstableFocus', 'C3')
                else:
                    label, col = ('Saddle', 'C1')
            else:
                label, col = ('unknown', 'C0')
            eqnaturelist += (label,)
            eqnatcollist += (col,)
        else:
            warnings.warn(f"Root finding failed for {varypars}={i}.")
    return ef_varypar, np.array(eqlist), eqnaturelist, eqnatcollist


def simulation_with_pulse(system, pulse_freq, pulse_amp, Tmax, y0, pars, fiH, fiU):
    pp = pulse_freq
    ts = 0
    sol_cb = []
    time = []
    while ts < Tmax:
        tspan = np.linspace(ts, pp, 100)
        sol = solve_ivp(lambda t, y: system(y, pars, fiH, fiU), [ts, pp], y0,
                            t_eval=tspan)
        sol_cb.append(sol.y[:, :-1])
        time.append(sol.t[:-1])
        ts = pp
        pp = pp + pulse_freq        
        s = 800
        y0 = sol.y[:, -1].copy()
        y0[-1] = y0[-1] + np.random.gamma(shape=pulse_amp**2/s**2, scale=s**2/pulse_amp)
    return Config(dict(y = np.hstack(sol_cb).T, t = np.concatenate(time)))
