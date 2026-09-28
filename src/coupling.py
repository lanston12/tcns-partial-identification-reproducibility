"""Strong Markov transition couplings; stationary-pair optimization, no scheduler.
State ordering is (u,v), with v varying fastest. Float64 certificates are
numerical residual certificates, not exact rational computer-assisted proofs.
"""
import numpy as np
from scipy.optimize import linprog

def stationary(P):
    P=np.asarray(P,float); n=len(P)
    return np.linalg.lstsq(np.vstack([P.T-np.eye(n),np.ones(n)]),np.r_[np.zeros(n),1.],rcond=None)[0]

def binary_data(a,b,c,d):
    PU=np.array([[1-a,a],[b,1-b]])
    PV=np.array([[1-c,c],[d,1-d]])
    p=a/(a+b); q=c/(c+d)
    aa=np.repeat([a,1-b],2); bb=np.tile([c,1-d],2)
    lo=np.maximum(0,aa+bb-1); hi=np.minimum(aa,bb)
    return PU,PV,p,q,aa,bb,lo,hi

def binary_kernel(a,b,c,d,theta):
    _,_,_,_,aa,bb,lo,hi=binary_data(a,b,c,d)
    theta=np.asarray(theta)
    if np.any(theta<lo-1e-10) or np.any(theta>hi+1e-10):raise ValueError('infeasible row coupling')
    return np.array([1-aa-bb+theta,bb-theta,aa-theta,theta]).T

def binary_interval(a,b,c,d):
    _,_,p,q,aa,bb,lo,hi=binary_data(a,b,c,d)
    w=np.array([1-p-q,q,p,0]); z=np.array([1,-1,-1,1])
    lower=max(0,p+q-1); upper=min(p,q)
    # A*t <= B: exact half-space form handles vanishing coefficients.
    for A,B in [(z@lo-1,-w@lo),(1-z@hi,w@hi)]:
        if A>1e-13:upper=min(upper,B/A)
        elif A< -1e-13:lower=max(lower,B/A)
        elif B< -1e-12:raise ValueError('empty feasibility interval')
    return float(lower),float(upper)

def construct_binary(a,b,c,d,t):
    _,_,p,q,aa,bb,lo,hi=binary_data(a,b,c,d)
    left,right=binary_interval(a,b,c,d)
    if not left-1e-10<=t<=right+1e-10:raise ValueError('infeasible stationary overlap')
    pi=np.array([1-p-q+t,q-t,p-t,t])
    den=pi@(hi-lo)
    eta=0. if abs(den)<1e-14 else (t-pi@lo)/den
    theta=lo+eta*(hi-lo)
    return binary_kernel(a,b,c,d,theta),pi,theta

def channel_stats(PV,s,r=.9025):
    PV=np.asarray(PV);s=np.asarray(s); n=len(PV);I=np.eye(n);one=np.ones(n)
    F=PV@np.diag(1-s);B=PV@np.diag(s);N=np.linalg.inv(I-F)
    m=N@one; m2=(2*N-I)@m
    g=r*np.linalg.solve(I-r*F,B@one)
    beta=stationary(PV);lam=beta@s;nu=beta*s/lam
    return dict(F=F,B=B,N=N,m=m,m2=m2,g=g,beta=beta,lam=lam,nu=nu)

def state_cost(PV,s,R,r=.9025,Q=.1,xi=1.):
    R=np.asarray(R);s=np.asarray(s); h=channel_stats(PV,s,r)
    if r==1:
        C=xi*(R[:,None]*h['m']+Q*(h['m2']-h['m'])/2)
    else:
        S=Q/(1-r)
        C=xi*(S*h['m']+(R[:,None]-S)*(1-h['g'])/(1-r))
    return (C*s).ravel()

def marked_law(pi,PV,s,ell):
    pi=np.asarray(pi).reshape(-1,len(PV));h=channel_stats(PV,s)
    vprob=np.linalg.matrix_power(h['F'],ell-1)@h['B']@np.ones(len(PV))
    return (pi*np.asarray(s)*vprob).sum(axis=1)/h['lam']

def joint_resolvent(P,pi,s,R,r=.9025,Q=.1,xi=1.):
    nv=len(s); ns=len(P);q=np.tile(s,ns//nv); RR=np.repeat(R,nv)
    F=P@np.diag(1-q);B=P@np.diag(q);one=np.ones(ns);I=np.eye(ns)
    lam=np.asarray(pi)@q;nu=np.asarray(pi)*q/lam
    m=np.linalg.solve(I-F,one)
    if r==1:
        m2=(2*np.linalg.inv(I-F)-I)@m
        D=xi*lam*np.sum(nu*(RR*m+Q*(m2-m)/2))
    else:
        g=r*np.linalg.solve(I-r*F,B@one);S=Q/(1-r)
        D=xi*lam*np.sum(nu*(S*m+(RR-S)*(1-g)/(1-r)))
    H=np.linalg.solve(I-F,B)
    return dict(D=float(D),H=H,nu=nu,F=F,B=B,mean_gap=float(nu@m))

def flow_system(PU,PV):
    """x[e,f] only: row sum is pi[e]. Constraints A x = b."""
    nu,nv=len(PU),len(PV); n=nu*nv; rows=[];rhs=[]
    def add(row,b=0):rows.append(row.ravel());rhs.append(b)
    add(np.ones((n,n)),1)
    for e in range(n):
        row=np.zeros((n,n));row[e,:]+=1;row[:,e]-=1;add(row)
    for e in range(n):
        u,v=divmod(e,nv)
        for up in range(nu):
            row=np.zeros((n,n));row[e,:]-=PU[u,up];row[e,up*nv:(up+1)*nv]+=1;add(row)
        for vp in range(nv):
            row=np.zeros((n,n));row[e,:]-=PV[v,vp];row[e,vp::nv]+=1;add(row)
    return np.array(rows),np.array(rhs)

def solve_flow(PU,PV,cost,maximize=False):
    A,b=flow_system(PU,PV);n=len(PU)*len(PV)
    c=np.repeat(cost,n)*(-1 if maximize else 1)
    primal=linprog(c,A_eq=A,b_eq=b,bounds=(0,None),method='highs')
    if not primal.success:raise RuntimeError(primal.message)
    # Explicit dual: max b.y subject to A.T y <= c, y free.
    dual=linprog(-b,A_ub=A.T,b_ub=c,bounds=[(None,None)]*len(b),method='highs')
    if not dual.success:raise RuntimeError(dual.message)
    x=primal.x.reshape(n,n);pi=x.sum(axis=1);P=np.kron(PU,PV)
    for e in range(n):
        if pi[e]>1e-12:P[e]=x[e]/pi[e]
    return dict(value=float(np.asarray(cost)@pi),pi=pi,P=P,x=x,dual=dual.x,
                primal_residual=float(np.max(np.abs(A@primal.x-b))),
                dual_violation=float(max(0,np.max(A.T@dual.x-c))),
                duality_gap=float(c@primal.x-b@dual.x),maximize=maximize)

def frechet(PV,s,R,eta,r=.9025,Q=.1,xi=1.):
    """Exact binary quantile allocation, infinite tail handled by resolvents."""
    h=channel_stats(PV,s,r);F=h['F'];B=h['B'];nu=h['nu'];one=np.ones(len(PV))
    if not 0<=eta<=1:raise ValueError('quality mass outside [0,1]')
    def tail(k):
        w=nu@np.linalg.matrix_power(F,k);prob=w@one
        if r==1:return prob,w@(k*one+h['m'])
        return prob,prob-r**(k+1)*(w@np.linalg.solve(np.eye(len(F))-r*F,B@one))
    total=tail(0)[1]
    def lower_mass(mass):
        if mass<=1e-15:return 0.
        if mass>=1-1e-15:return total
        k=0
        while 1-tail(k+1)[0]<mass:k+=1
        pnext,gnext=tail(k+1)
        pk,gk=tail(k)
        g_atom=k+1 if r==1 else 1-r**(k+1)
        return total-gk+(mass-(1-pk))*g_atom
    low=lower_mass(eta);high=total-lower_mass(1-eta)
    if r==1:
        base=Q*(nu@(h['m2']-h['m']))/2
        scale=xi*h['lam'];offset=R[0]*total+base
    else:
        S=Q/(1-r);scale=xi*h['lam']/(1-r)
        offset=(R[0]-S)*total
    lower=scale*(offset+(R[1]-R[0])*low)
    upper=scale*(offset+(R[1]-R[0])*high)
    if r!=1:lower+=xi*S;upper+=xi*S
    return float(lower),float(upper)

def success_word_probability(P,pi,s,word):
    q=np.tile(s,len(P)//len(s));w=np.asarray(pi).copy()
    for i,y in enumerate(word):
        w*=q if y else 1-q
        if i+1<len(word):w=w@P
    return float(w.sum())
