"""A finite interior pocket: an inverse, spherical GR experiment.

This is a prescribed metric, NOT a collapse simulation or a matter theory.
Units: G=c=1; all lengths are measured in an arbitrary reference length L0.
The default ADM mass parameter is m=1 L0. No physical cutoff is fitted.

Run: python src/baseline/pocket_model.py --output results
Dependencies: numpy, scipy, matplotlib. No network access is used.

Reference lapse: Hayward, https://arxiv.org/abs/gr-qc/0506126
The nonmonotonic areal-radius profile below is a trial choice for this study.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Parameters:
    m: float = 1.0
    ell: float = 2.0 / 3.0
    b: float = 1.0 / 12.0
    amplitude: float = 5.0

    def validate(self):
        if min(self.m, self.ell, self.b) <= 0 or self.amplitude < 0:
            raise ValueError("m, ell, b must be positive; amplitude >= 0")


def critical_amplitude():
    z = (11.0 + np.sqrt(41.0)) / 4.0
    return float(np.exp(z) / (z * z * (2.0 * z - 5.0)))


def geometry(x, p=Parameters()):
    """R, R', R'', f, f', f'', plus cancellation-safe central quantities.

    ds^2 = -f(x) dt^2 + dx^2/f(x) + R(x)^2 dOmega^2.
    R=x[1+A(x/b)^4 exp(-(x/b)^2)], x >= 0.
    f=F(R)=1-2m R^2/(R^3+2m ell^2).
    Complex inputs are supported for independent derivative checks.
    """
    x = np.asarray(x)
    y = x / p.b
    z = y * y
    e = np.exp(-z)
    u = p.amplitude * z * z * e
    delta_rp = u * (5.0 - 2.0 * z)
    radius = x * (1.0 + u)
    rp = 1.0 + delta_rp
    rpp = p.amplitude / p.b * e * y**3 * (20.0 - 22.0 * z + 4.0 * z*z)
    s = p.amplitude / p.b**2 * e * z * (20.0 - 22.0*z + 4.0*z*z) / (1.0+u)
    delta_rp_over_r2 = p.amplitude / p.b**2 * e * z * (5.0-2.0*z) / (1.0+u)**2
    c0 = 2.0 * p.m * p.ell**2
    den = radius**3 + c0
    f = 1.0 - 2.0 * p.m * radius**2 / den
    fr = 2.0 * p.m * radius * (radius**3 - 2.0*c0) / den**2
    frr = -4.0 * p.m * (radius**6 - 7.0*c0*radius**3 + c0*c0) / den**3
    fp = fr * rp
    fpp = frr * rp**2 + fr*rpp
    # B=f'R'/(2R), D=(1-f R'^2)/R^2; these versions include x=0.
    B = p.m * (radius**3 - 2.0*c0) * rp**2 / den**2
    D = 2.0 * p.m / den - f * (2.0+delta_rp) * delta_rp_over_r2
    return dict(R=radius, Rp=rp, Rpp=rpp, f=f, fp=fp, fpp=fpp, S=s, B=B, D=D)


def tensors(x, p=Parameters()):
    """Einstein eigenvalues, polynomial invariants, and radial null contraction.

    G=8 pi T. Energy density and pressure are local eigenframe quantities.
    The t/x timelike interpretation switches at f=0; the limiting values at
    a horizon are used in the arrays, not a static observer at the horizon.
    N=8 pi T(n,n) for n=-partial_x in ingoing EF coordinates, where
    l=partial_v+(f/2)partial_x and g(l,n)=-1. N is regular at f=0.
    """
    g = geometry(x, p)
    a = 0.5*g['fpp']
    B, D, S, f = g['B'], g['D'], g['S'], g['f']
    C = f*S+B
    gt = 2.0*f*S+2.0*B-D
    gx = 2.0*B-D
    ga = f*S+2.0*B+a
    trace = gt+gx+2.0*ga
    ricci = -trace
    ricci2 = (gt-0.5*trace)**2 + (gx-0.5*trace)**2 + 2.0*(ga-0.5*trace)**2
    K = 4.0*(a*a+2.0*B*B+2.0*C*C+D*D)
    epsilon8pi = np.where(np.real(f) >= 0, -gt, -gx)
    radial_pressure8pi = np.where(np.real(f) >= 0, gx, gt)
    return dict(**g, Gt=gt, Gx=gx, Gangle=ga, Ricci=ricci, Ricci2=ricci2,
                K=K, NEC8pi=-2.0*S, epsilon8pi=epsilon8pi,
                radial_pressure8pi=radial_pressure8pi,
                transverse_pressure8pi=ga)


def extrema(p=Parameters()):
    """Return (bulge x, neck x), or None when no strict extrema exist."""
    if p.amplitude <= critical_amplitude():
        return None
    z = (11.0+np.sqrt(41.0))/4.0
    derivative = lambda y: float(geometry(p.b*y, p)['Rp'])
    peak = brentq(derivative, np.sqrt(2.5), np.sqrt(z), xtol=1e-13)
    neck = brentq(derivative, np.sqrt(z), 12.0, xtol=1e-13)
    return p.b*peak, p.b*neck


def areal_horizons(p=Parameters()):
    roots = np.roots([1.0, -2.0*p.m, 0.0, 2.0*p.m*p.ell**2])
    return sorted(float(r.real) for r in roots if abs(r.imag) < 1e-8 and r.real > 0)


def x_for_outer_branch(radius, p=Parameters()):
    ex = extrema(p)
    start = 0.0 if ex is None else ex[1]
    end = max(10.0*p.m, 20.0*p.b, 2.0*radius)
    return brentq(lambda x: float(geometry(x,p)['R'])-radius, start, end)


def coordinate_horizons(p=Parameters()):
    """All zeros of f, including extra intersections for a very large bulge.

    An extremum of R is not automatically a Killing or event horizon.
    """
    hs=areal_horizons(p)
    if not hs:
        return []
    ex=extrema(p)
    end=max(10*p.m,20*p.b,2*max(hs))
    edges=[0.0]+([] if ex is None else list(ex))+[end]
    roots=[]
    for target in hs:
        for lo,hi in zip(edges[:-1],edges[1:]):
            func=lambda x:float(geometry(x,p)['R'])-target
            a,b=func(lo),func(hi)
            if abs(a)<1e-11: roots.append(lo)
            if a*b<0: roots.append(brentq(func,lo,hi,xtol=1e-13))
            if abs(b)<1e-11: roots.append(hi)
    unique=[]
    for root in sorted(roots):
        if not unique or abs(root-unique[-1])>1e-9:
            unique.append(root)
    return unique


def profile(p=Parameters()):
    # Resolve the nonmonotonic core, all horizons, and the far-field tail.
    return np.unique(np.r_[0.0, np.geomspace(p.b*1e-5,p.b*0.02,100),
                          np.linspace(p.b*0.02,6.0*p.b,5001),
                          np.linspace(6.0*p.b,10.0*p.m,2000)])


def summary(p=Parameters()):
    p.validate()
    ex = extrema(p)
    hs = areal_horizons(p)
    x = profile(p)
    t = tensors(x,p)
    i = int(np.argmax(t['K']))
    if 0 < i < len(x)-1:
        fit = minimize_scalar(lambda q:-float(tensors(q,p)['K']),
                              bounds=(x[i-1],x[i+1]), method='bounded',
                              options={'xatol':1e-14})
        kmax, xmax = -float(fit.fun), float(fit.x)
    else:
        kmax, xmax = float(t['K'][i]), float(x[i])
    result = dict(parameters=asdict(p), amplitude_critical=critical_amplitude(),
                  K_center=float(t['K'][0]), K_peak_in_scan=kmax, x_K_peak=xmax,
                  areal_horizons=hs,
                  epsilon8pi_min_in_scan=float(np.min(t['epsilon8pi'])),
                  NEC8pi_min_in_scan=float(np.min(t['NEC8pi'])))
    if ex:
        xp,xn = ex
        tp,tn = tensors(xp,p),tensors(xn,p)
        result.update(x_bulge=xp,R_bulge=float(tp['R']),x_neck=xn,R_neck=float(tn['R']),
                      f_neck=float(tn['f']),
                      epsilon8pi_neck=float(tn['epsilon8pi']),
                      radial_pressure8pi_neck=float(tn['radial_pressure8pi']),
                      transverse_pressure8pi_neck=float(tn['transverse_pressure8pi']),
                      NEC8pi_neck=float(tn['NEC8pi']),K_neck=float(tn['K']))
        core_grid=np.linspace(0,xn,2001)
        result['pocket_static_throughout']=bool(np.min(geometry(core_grid,p)['f'])>0)
        if result['pocket_static_throughout']:
            integrand=lambda q:float(geometry(q,p)['R']**2/np.sqrt(geometry(q,p)['f']))
            value,error=quad(integrand,0,xn,epsabs=1e-12,epsrel=1e-11)
            volume=4.0*np.pi*value
            result.update(pocket_volume=volume,volume_quadrature_error=4*np.pi*error,
                          volume_over_euclidean_ball=volume/(4*np.pi*float(tn['R'])**3/3))
    if hs:
        xh=coordinate_horizons(p)
        result['x_killing_horizons']=xh
        result['inner_surface_gravity_abs']=abs(float(geometry(xh[0],p)['fp']))/2
        result['outer_surface_gravity']=float(geometry(xh[-1],p)['fp'])/2
        result['all_killing_horizons_on_outer_branch']=bool(ex is None or result['R_bulge'] < hs[0])
        # Photon sphere: R F_R - 2 F=0; no R(x) shape parameter occurs here.
        q=replace(p,amplitude=0.0)
        photon=brentq(lambda r:float(r*geometry(r,q)['fp']-2*geometry(r,q)['f']),
                      hs[-1]*(1+1e-9),10*p.m)
        result['photon_areal_radius']=photon
        result['critical_photon_impact_parameter']=photon/np.sqrt(float(geometry(photon,q)['f']))
    return result


def ef_metric(coordinates, p):
    """Metric only; independent coordinate tensor check starts from here."""
    _,x,theta,_ = coordinates
    g = geometry(x,p)
    metric=np.zeros((4,4),dtype=np.result_type(coordinates,float))
    metric[0,0]=-g['f']
    metric[0,1]=metric[1,0]=1.0
    metric[2,2]=g['R']**2
    metric[3,3]=g['R']**2*np.sin(theta)**2
    return metric


def coordinate_curvature(x,p):
    """Independent Christoffel/Riemann contraction in a horizon-regular chart.

    First metric derivatives: complex step. Second derivatives: fourth-order
    central differences of the first derivatives. No reduced curvature or
    stress formula is used by the tensor contraction.
    """
    q=np.array([0.,x,1.1,0.])
    def first(at):
        derivative=np.zeros((4,4,4))
        for k in (1,2):
            z=at.astype(complex)
            z[k]+=1e-25j
            derivative[k]=np.imag(ef_metric(z,p))/1e-25
        return derivative
    metric=ef_metric(q,p)
    inv=np.linalg.inv(metric)
    dg=first(q)
    ddg=np.zeros((4,4,4,4))
    for k in (1,2):
        h=2e-4*(max(p.b,abs(x)) if k==1 else 1.0)
        offset=np.zeros(4);offset[k]=h
        ddg[k]=(-first(q+2*offset)+8*first(q+offset)-8*first(q-offset)+first(q-2*offset))/(12*h)
    dinv=np.array([-inv@dg[k]@inv for k in range(4)])
    gamma=np.zeros((4,4,4));dGamma=np.zeros((4,4,4,4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                term=np.array([dg[b,d,c]+dg[c,d,b]-dg[d,b,c] for d in range(4)])
                gamma[a,b,c]=0.5*np.dot(inv[a],term)
                for e in range(4):
                    dterm=np.array([ddg[e,b,d,c]+ddg[e,c,d,b]-ddg[e,d,b,c] for d in range(4)])
                    dGamma[e,a,b,c]=0.5*(np.dot(dinv[e,a],term)+np.dot(inv[a],dterm))
    riemann=np.zeros((4,4,4,4))
    for a,b,c,d in np.ndindex(4,4,4,4):
        riemann[a,b,c,d]=(dGamma[c,a,d,b]-dGamma[d,a,c,b]
            +np.dot(gamma[a,c,:],gamma[:,d,b])-np.dot(gamma[a,d,:],gamma[:,c,b]))
    ricci=np.einsum('abad->bd',riemann)
    scalar=float(np.einsum('ab,ab',inv,ricci))
    einstein=inv@(ricci-0.5*scalar*metric)
    lower=np.einsum('ae,ebcd->abcd',metric,riemann)
    K=float(np.einsum('abcd,efgh,ae,bf,cg,dh->',lower,lower,inv,inv,inv,inv,optimize=True))
    return einstein,scalar,K


def verify(p=Parameters()):
    """Scientific validation: independent contractions and known limits."""
    checks=[]
    def record(name,error,tolerance):
        checks.append(dict(check=name,error=float(error),tolerance=tolerance,
                           passed=bool(error<tolerance)))
    q=replace(p,amplitude=0.0)
    x=np.geomspace(1e-3,10,300)*p.m
    t=tensors(x,q)
    c0=2*p.m*p.ell**2
    gt_expected=-12*p.ell**2*p.m**2/(x**3+c0)**2
    ga_expected=24*(x**3-p.ell**2*p.m)*p.ell**2*p.m**2/(x**3+c0)**3
    record('Hayward Einstein tensor against published expressions',
           max(np.max(np.abs(t['Gt']-gt_expected)),np.max(np.abs(t['Gangle']-ga_expected))),1e-9)
    record('Regular-centre K=24/ell^4',abs(float(tensors(0,p)['K'])-24/p.ell**4),1e-9)
    # Independent coordinate calculation includes both horizons, where t,x
    # Schwarzschild-like coordinates themselves would be ill-conditioned.
    points=[0.025*p.m,0.08*p.m,0.14*p.m,0.24*p.m,0.4*p.m,3*p.m]
    points+=summary(p).get('x_killing_horizons',[])
    for point in points:
        gt,rs,k=coordinate_curvature(point,p)
        expected=tensors(point,p)
        g=np.diag([expected['Gt'],expected['Gx'],expected['Gangle'],expected['Gangle']])
        g[0,1]=expected['NEC8pi']
        scale=max(1.,np.max(np.abs(g)))
        record(f'EF Einstein tensor x={point:.9g}',np.max(np.abs(gt-g))/scale,2e-6)
        record(f'EF Riemann contraction K x={point:.9g}',abs(k-expected['K'])/max(1.,abs(k)),2e-6)
        record(f'EF Ricci scalar x={point:.9g}',abs(rs-expected['Ricci'])/max(1.,abs(rs)),2e-6)
    # Schwarzschild asymptotic coefficient, with no subtraction at the centre.
    far=1000*p.m
    record('Schwarzschild far-field K coefficient',
           abs(float(tensors(far,p)['K'])/(48*p.m**2/far**6)-1),1e-7)
    # Bianchi identity with the f denominator cancelled analytically.
    x=np.linspace(0.01*p.m,2.5*p.m,801)
    h=1e-25
    dG=np.imag(tensors(x+1j*h,p)['Gx'])/h
    t=tensors(x,p)
    residual=dG-t['fp']*t['S']+2*t['Rp']/t['R']*(t['Gx']-t['Gangle'])
    record('Covariant stress conservation',np.max(np.abs(residual))/max(1.,np.max(np.abs(dG))),1e-9)
    # NEC at a strict neck follows from the actual computed R'' > 0.
    ex=extrema(p)
    if ex:
        record('Strict neck R prime = 0',abs(float(geometry(ex[1],p)['Rp'])),1e-9)
        if not geometry(ex[1],p)['Rpp']>0:
            raise AssertionError('Candidate is not a strict neck')
    if not all(item['passed'] for item in checks):
        raise AssertionError(json.dumps(checks,indent=2))
    return checks


def write_csv(path,columns):
    names=list(columns)
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(names)
        writer.writerows(zip(*(np.atleast_1d(columns[n]) for n in names)))


def make_plots(out,p,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':0.16,'figure.facecolor':'#ffffff'})
    q=replace(p,amplitude=0.)
    orange,blue,red='#d97818','#3265aa','#a92648'
    if not (s.get('pocket_static_throughout') and len(s.get('x_killing_horizons',[]))==2
            and s.get('all_killing_horizons_on_outer_branch')):
        raise ValueError('This overview layout requires a static pocket below two horizons; use tensors() to explore other regimes.')
    xp,xn=s['x_bulge'],s['x_neck']
    fig,axs=plt.subplots(2,2,figsize=(12.5,8.4),layout='constrained')
    x=np.linspace(0,0.55*p.m,2200); t=tensors(x,p)
    ax=axs[0,0]
    ax.plot(x,t['R'],color=orange,lw=2.5,label='Trial pocket')
    ax.plot(x,x,'--',color=blue,lw=1.5,label='Control without pocket')
    ax.axvspan(0,xn,color=orange,alpha=.08)
    ax.scatter([xp,xn],[s['R_bulge'],s['R_neck']],color=orange,zorder=4)
    ax.annotate('Bulge',xy=(xp,s['R_bulge']),xytext=(0.19,0.49),fontsize=10)
    ax.annotate('Neck',xy=(xn,s['R_neck']),xytext=(0.29,0.24),fontsize=10,
                arrowprops={'arrowstyle':'-','color':orange})
    ax.set(title='1. Shape of the finite pocket',xlabel=r'$x/L_0$',ylabel=r'$R/L_0$')
    ax.legend(fontsize=9,loc='lower right')
    ax=axs[0,1]
    xx=np.unique(np.r_[np.geomspace(1e-5,0.01,150),np.linspace(.01,3,3500)])*p.m
    tt=tensors(xx,p); tq=tensors(xx,q)
    ax.loglog(xx,48*p.m**2/xx**6,color='#777777',ls=':',label='Schwarzschild')
    ax.loglog(xx,tq['K'],color=blue,ls='--',label='Hayward: control')
    ax.loglog(xx,tt['K'],color=orange,label='Trial pocket')
    ax.set(title='2. Finite curvature with a maximum in the layer',xlabel=r'$x/L_0$',ylabel=r'$K L_0^4$')
    ax.set_ylim(1e-1,1e11);ax.set_xlim(1e-4,3)
    ax.legend(fontsize=9,loc='upper right')
    ax=axs[1,0]
    ax.plot(x,t['epsilon8pi'],color=red,label=r'$8\pi\varepsilon$')
    ax.plot(x,t['radial_pressure8pi'],color=blue,label=r'$8\pi p_{\parallel}$')
    ax.plot(x,t['epsilon8pi']+t['radial_pressure8pi'],color=orange,ls='--',label=r'$8\pi(\varepsilon+p_{\parallel})$')
    ax.axhline(0,color='#333333',lw=.7)
    ax.axvline(xn,color='#777777',ls=':',lw=1)
    ax.set(title='3. Energy density and radial pressure',xlabel=r'$x/L_0$',ylabel=r'Value $\times L_0^2$')
    ax.legend(fontsize=9,loc='lower right')
    ax=axs[1,1]
    xx=np.linspace(0,2.2*p.m,2500); ff=geometry(xx,p)['f']
    ax.plot(xx,ff,color=blue,lw=2)
    ax.axhline(0,color='#333333',lw=.7)
    hi,ho=s['x_killing_horizons']
    ax.axvspan(hi,ho,color=red,alpha=.10,label='Layer with f < 0')
    ax.axvspan(0,xn,color=orange,alpha=.13,label='Pocket')
    ax.scatter([hi,ho],[0,0],color=red,zorder=4)
    ax.set(title='4. The pocket lies deeper than the inner horizon',xlabel=r'$x/L_0$',ylabel=r'$f(x)$')
    ax.legend(fontsize=9,loc='upper right')
    fig.suptitle('First calculation: prescribed geometry, computed consequences',fontsize=15)
    fig.savefig(out/'overview.png',dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.8,5.4),layout='constrained')
    v=np.linspace(0,8*p.m,1000)
    ax.axhspan(hi,ho,color=red,alpha=.08)
    ax.axhspan(0,xn,color=orange,alpha=.10)
    for start in [xp,.35*p.m,.65*p.m,1.15*p.m,1.95*p.m]:
        sol=solve_ivp(lambda vv,yy:[float(geometry(yy[0],p)['f'])/2],
                      (v[0],v[-1]),[start],t_eval=v,rtol=1e-10,atol=1e-12,max_step=.025*p.m)
        ax.plot(sol.t,sol.y[0],color=blue,lw=1.7)
        # Arrow tangent shows future direction in this EF patch.
        j=150
        ax.annotate('',xy=(sol.t[j+12],sol.y[0,j+12]),xytext=(sol.t[j],sol.y[0,j]),
                    arrowprops={'arrowstyle':'->','color':blue,'lw':1.5})
    # Other future radial null family: v=constant, decreasing x.
    for vv in [.6,2.4,4.2,6.0]:
        ax.annotate('',xy=(vv,.05*p.m),xytext=(vv,.8*p.m),
                    arrowprops={'arrowstyle':'->','color':orange,'lw':1.4})
    for h,label in [(xn,'neck'),(hi,'inner horizon'),(ho,'outer horizon')]:
        ax.axhline(h,color='#777777',lw=.8,ls=':')
        ax.text(7.9*p.m,h+.025*p.m,label,ha='right',fontsize=9)
    ax.set(xlim=(0,8*p.m),ylim=(0,2.35*p.m),xlabel=r'$v/L_0$',ylabel=r'$x/L_0$',
           title='Light paths: exchange at the neck is possible; no escape outward follows')
    fig.savefig(out/'causal_paths.png',dpi=160);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for amp in [0,critical_amplitude(),2,5]:
        qq=replace(p,amplitude=amp)
        xx=np.linspace(0,.45*p.m,1200)
        axs[0].plot(xx,geometry(xx,qq)['R'],label=f'α = {amp:.3g}')
    axs[0].set(title='How the neck appears',xlabel=r'$x/L_0$',ylabel=r'$R/L_0$')
    axs[0].legend(fontsize=9)
    amps=np.linspace(critical_amplitude()+1e-5,5,160)
    ratios=[]
    for amp in amps:
        qq=replace(p,amplitude=float(amp)); ex=extrema(qq)
        ratios.append(float(geometry(ex[0],qq)['R']/geometry(ex[1],qq)['R']))
    axs[1].plot(amps,ratios,color=orange,lw=2)
    axs[1].axvline(critical_amplitude(),ls=':',color=blue)
    axs[1].set(title='Shape threshold and pocket size',xlabel='Shape parameter α',ylabel=r'$R_b/R_n$')
    fig.savefig(out/'shape_scan.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    p=Parameters();s=summary(p);checks=verify(p)
    x=profile(p);t=tensors(x,p)
    write_csv(out/'radial_profile.csv',dict(x=x,**t))
    scan=[]
    for amp in np.unique(np.r_[np.linspace(0,5,61),critical_amplitude()]):
        qq=replace(p,amplitude=float(amp));ex=extrema(qq)
        grid=profile(qq);field=tensors(grid,qq)
        peak_K=float(np.max(field['K']))
        min_epsilon=float(np.min(field['epsilon8pi']))
        if ex:
            tn=tensors(ex[1],qq)
            row=[amp,*ex,float(geometry(ex[0],qq)['R']),float(tn['R']),
                 float(tn['NEC8pi']),float(tn['epsilon8pi']),peak_K,min_epsilon]
        else:row=[amp,*([np.nan]*6),peak_K,min_epsilon]
        scan.append(row)
    write_csv(out/'shape_scan.csv',dict(zip(['amplitude','x_bulge','x_neck','R_bulge','R_neck',
                                           'NEC8pi_neck','epsilon8pi_neck','K_peak_in_scan',
                                           'epsilon8pi_min_in_scan'],np.array(scan).T)))
    (out/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    make_plots(out,p,s)
    print(json.dumps(dict(summary=s,validation_count=len(checks),all_passed=True),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
