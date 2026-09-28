"""Certificates from dependent, fully observed finite-state Markov trajectories.
Row-exit confidence sequences are classical Hoeffding/union-bound constructions.
No IID trajectory assumption, scheduler optimization, or noisy-mode inference.
"""
import math
import numpy as np
from scipy.optimize import linprog
from coupling import binary_data,binary_interval,flow_system

def overlap_interval(p,q,lo,hi):
    if np.any(np.asarray(lo)>np.asarray(hi)+1e-12):
        raise ValueError('empty confidence consistency set: row interval')
    w=np.array([1-p-q,q,p,0.]);z=np.array([1,-1,-1,1.])
    left=max(0.,p+q-1);right=min(p,q)
    for A,B in [(z@lo-1,-w@lo),(1-z@hi,w@hi)]:
        if A>1e-13:right=min(right,B/A)
        elif A< -1e-13:left=max(left,B/A)
        elif B< -1e-12:raise ValueError('empty confidence consistency set')
    if left>right+1e-10:raise ValueError('empty confidence consistency set')
    return float(left),float(right)

def cost_interval(p,q,lo,hi,cost):
    ends=overlap_interval(p,q,lo,hi)
    vals=[np.array([1-p-q+t,q-t,p-t,t])@cost for t in ends]
    return float(min(vals)),float(max(vals)),ends

def refined_witness(args,lo,hi,t):
    from coupling import binary_kernel
    _,_,p,q,*_=binary_data(*args)
    pi=np.array([1-p-q+t,q-t,p-t,t])
    den=pi@(hi-lo);alpha=0. if abs(den)<1e-14 else (t-pi@lo)/den
    theta=lo+alpha*(hi-lo)
    return binary_kernel(*args,theta),pi,theta

def confidence_radius(m,delta=.05,coordinates=4):
    m=np.asarray(m,float)
    good=m>0;rad=np.ones(m.shape)
    rad[good]=np.sqrt(np.log(2*coordinates*(np.pi**2/6)*m[good]**2/delta)/(2*m[good]))
    return np.minimum(rad,1.)

class BinaryConfidence:
    """Nested, anytime-valid θ boxes, using every observed exit from a row."""
    def __init__(self,args,delta=.05):
        assert 0<delta<1
        self.args=args;self.delta=delta
        _,_,self.p,self.q,*_,lo,hi=binary_data(*args)
        self.lo=lo.copy();self.hi=hi.copy();self.visits=np.zeros(4,int);self.successes=np.zeros(4,int);self.steps=0
    def observe(self,e,f):
        self.steps+=1;self.visits[e]+=1;self.successes[e]+=int(f==3)
        m=self.visits[e];hat=self.successes[e]/m;eps=min(1., math.sqrt(math.log(8*(math.pi**2/6)*m*m/self.delta)/(2*m)))
        self.lo[e]=max(self.lo[e],hat-eps);self.hi[e]=min(self.hi[e],hat+eps)
    def snapshot(self):
        return dict(N=self.steps,visits=self.visits.copy(),successes=self.successes.copy(),lo=self.lo.copy(),hi=self.hi.copy(),
                    radii=confidence_radius(self.visits,self.delta),theta_hat=np.divide(self.successes,self.visits,out=np.zeros(4,float),where=self.visits>0))

def restricted_flow(PU,PV,cost,L=None,H=None,maximize=False):
    """General finite-state row-cell confidence bounds: L*pi <= x <= H*pi."""
    A,b=flow_system(PU,PV);nu,nv=len(PU),len(PV);n=nu*nv;rows=[];fallback=[]
    # Zero-flow rows still need a feasible kernel row; never hide model failure.
    transport=np.vstack([np.kron(np.eye(nu),np.ones((1,nv))),
                         np.kron(np.ones((1,nu)),np.eye(nv))])
    for e in range(n):
        u,v=divmod(e,nv)
        lower=np.zeros(n) if L is None else L[e]
        upper=np.ones(n) if H is None else H[e]
        row=linprog(np.zeros(n),A_eq=transport,b_eq=np.r_[PU[u],PV[v]],
                    bounds=list(zip(lower,upper)),method='highs')
        if not row.success:raise ValueError('empty confidence consistency row '+str(e))
        fallback.append(row.x)
    if L is not None:
        for e in range(n):
            for f in range(n):
                row=np.zeros((n,n));row[e,:]=L[e,f];row[e,f]-=1;rows.append(row.ravel())
    if H is not None:
        for e in range(n):
            for f in range(n):
                row=np.zeros((n,n));row[e,:]=-H[e,f];row[e,f]+=1;rows.append(row.ravel())
    G=np.array(rows).reshape(-1,n*n);c=np.repeat(cost,n)*(-1 if maximize else 1)
    res=linprog(c,A_eq=A,b_eq=b,A_ub=G if len(G) else None,b_ub=np.zeros(len(G)) if len(G) else None,bounds=(0,None),method='highs')
    if not res.success:raise ValueError(res.message)
    # Dual for A x=b, Gx<=0: max b*y, A.T*y + G.T*z <= c, z<=0.
    mat=np.column_stack([A.T,G.T]);objective=np.r_[-b,np.zeros(len(G))]
    dual=linprog(objective,A_ub=mat,b_ub=c,bounds=[(None,None)]*len(b)+[(None,0)]*len(G),method='highs')
    if not dual.success:raise RuntimeError(dual.message)
    pi=res.x.reshape(n,n).sum(axis=1)
    P=np.array(fallback)
    for e in range(n):
        if pi[e]>1e-12:P[e]=res.x.reshape(n,n)[e]/pi[e]
    return dict(value=float(pi@cost),pi=pi,P=P,x=res.x.reshape(n,n),dual=dual.x,A=A,b=b,G=G,c=c,
                primal_residual=float(np.max(np.abs(A@res.x-b))),inequality_residual=float(max(0,np.max(G@res.x))) if len(G) else 0.,
                dual_violation=float(max(0,np.max(mat@dual.x-c))),duality_gap=float(c@res.x-b@dual.x[:len(b)]),maximize=maximize)

def binary_cell_bounds(args,lo,hi):
    *_,aa,bb,_,_=binary_data(*args)
    L=np.array([1-aa-bb+lo,bb-hi,aa-hi,lo]).T
    H=np.array([1-aa-bb+hi,bb-lo,aa-lo,hi]).T
    return np.maximum(0,L),np.minimum(1,H)

def generate_trajectory(P,N,seed=20260929,initial=0):
    rng=np.random.RandomState(seed);un=rng.random_sample(N);cdf=np.cumsum(P,axis=1)
    states=np.empty(N+1,np.int8);states[0]=initial
    for k in range(N):states[k+1]=np.searchsorted(cdf[states[k]],un[k],side='right')
    return states

def slow_fair_coupling(epsilon,a=1.,b=1.):
    """Both projections IID fair bits; agreement class has slow transitions."""
    x=np.array([1-epsilon*a,epsilon*b,epsilon*b,1-epsilon*a])
    return np.array([x/2,(1-x)/2,(1-x)/2,x/2]).T


class FiniteConfidence:
    """All-cell nested row-exit intervals; caller intersects with strong transports."""
    def __init__(self,states,delta=.05):
        assert states>0 and 0<delta<1
        self.n=states;self.delta=delta;self.visits=np.zeros(states,int)
        self.counts=np.zeros((states,states),int);self.lo=np.zeros((states,states));self.hi=np.ones((states,states));self.steps=0
    def observe(self,e,f):
        self.steps+=1;self.visits[e]+=1;self.counts[e,f]+=1
        m=self.visits[e];hat=self.counts[e]/m
        eps=min(1.,math.sqrt(math.log(2*self.n**2*(math.pi**2/6)*m*m/self.delta)/(2*m)))
        self.lo[e]=np.maximum(self.lo[e],hat-eps);self.hi[e]=np.minimum(self.hi[e],hat+eps)
    def snapshot(self):
        return dict(N=self.steps,visits=self.visits.copy(),counts=self.counts.copy(),lo=self.lo.copy(),hi=self.hi.copy())
