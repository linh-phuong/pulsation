import warnings
import numpy as np
from scipy.optimize import root

def foodweb(vars, cg):
    C, H, U = vars
    fiH = cg.aH * cg.omegaH * H/(1 + cg.omegaH * cg.ThH * H + cg.omegaU * cg.ThU * U )
    fiU = cg.aU * cg.omegaU * U/(1 + cg.omegaU * cg.ThU * U + cg.omegaH * cg.ThH * H)
    dCdt =  (-cg.muC + cg.eH * cg.pCH**(-1/4) * fiH + cg.eU * cg.pCU**(-1/4) * fiU) * C
    dHdt = (cg.rH * (1 - H/cg.K) - cg.muH)*H - cg.pCH**(-1/4) * fiH * C
    dUdt = - cg.nu * U - cg.pCU**(-1/4) * fiU * C
    return [dCdt, dHdt, dUdt]


def foodweb_jacobian(vars, cg):
    C, H, U = vars
    D = 1 + cg.omegaH * cg.ThH * H + cg.omegaU * cg.ThU * U  # common denominator of fiH and fiU
    dCdi = [-cg.muC + cg.aH * cg.eH * H * cg.omegaH/(cg.pCH**(1/4) * D)
                    + cg.aU * cg.eU * U * cg.omegaU/(cg.pCU**(1/4) * D),
            C * (-cg.aH * cg.eH * H * cg.ThH * cg.omegaH**2/(cg.pCH**(1/4) * D**2)
                 - cg.aU * cg.eU * cg.ThH * U * cg.omegaH * cg.omegaU/(cg.pCU**(1/4) * D**2)
                 + cg.aH * cg.eH * cg.omegaH/(cg.pCH**(1/4) * D)),
            C * (-cg.aH * cg.eH * H * cg.ThU * cg.omegaH * cg.omegaU/(cg.pCH**(1/4) * D**2)
                 - cg.aU * cg.eU * cg.ThU * U * cg.omegaU**2/(cg.pCU**(1/4) * D**2)
                 + cg.aU * cg.eU * cg.omegaU/(cg.pCU**(1/4) * D))]
    dHdi = [-cg.aH * H * cg.omegaH/(cg.pCH**(1/4) * D),
            cg.rH * (1 - H/cg.K) - cg.rH * H/cg.K - cg.muH
                + cg.aH * H * cg.ThH * C * cg.omegaH**2/(cg.pCH**(1/4) * D**2)
                - cg.aH * C * cg.omegaH/(cg.pCH**(1/4) * D),
            cg.aH * H * cg.ThU * C * cg.omegaH * cg.omegaU/(cg.pCH**(1/4) * D**2)]
    dUdi = [-cg.aU * U * cg.omegaU/(cg.pCU**(1/4) * D),
            cg.aU * cg.ThH * U * C * cg.omegaH * cg.omegaU/(cg.pCU**(1/4) * D**2),
            -cg.nu + cg.aU * cg.ThU * U * C * cg.omegaU**2/(cg.pCU**(1/4) * D**2)
                - cg.aU * C * cg.omegaU/(cg.pCU**(1/4) * D)]
    return np.array([dCdi, dHdi, dUdi])

def followpars(system, commonpars, varypars, varyvals, initeq):
    eqlist = ()
    ef_varypar = ()
    pars = commonpars.clone_me()
    for i in varyvals:
        pars[varypars] = i
        eq = root(system, initeq, args=(pars,))
        if eq.success:
            eqlist += (eq.x,)
            ef_varypar += (i,)
            initeq = eq.x
        else:
            warnings.warn(f"Root finding failed for {varypars}={i}.")
    return ef_varypar, np.array(eqlist)


def followpars_with_eqnature(system, systemJ, commonpars, varypars,
                              varyvals, initeq):
    eqlist = ()
    ef_varypar = ()
    eqnaturelist = ()
    eqnatcollist = ()
    pars = commonpars.clone_me()
    for i in varyvals:
        pars[varypars] = i
        eq = root(system, initeq, args=(pars,))
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
                label, col = ('Unstable', 'C3')
            else:
                label, col = ('unknown', 'C0')
            eqnaturelist += (label,)
            eqnatcollist += (col,)
        else:
            warnings.warn(f"Root finding failed for {varypars}={i}.")
    return ef_varypar, np.array(eqlist), eqnaturelist, eqnatcollist
