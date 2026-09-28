"""Exact colored-error propagation for fixed stable linear cascade.
Fast isotropic-error implementation and an independent full augmented solver.
All covariance quantities are stationary expected moments, not white approximations.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov,expm
from coupling import stationary

def isotropic_moments(PV,s,pi,R,Acl,Be,a=.95,Q=.1):
    PV=np.asarray(PV);s=np.asarray(s);Acl=np.asarray(Acl);Be=np.asarray(Be)
    nv=len(PV);nz,ne=Be.shape;beta=stationary(PV);pi=np.asarray(pi).reshape(-1,nv)
    F=PV@np.diag(1-s)
    forcing=s*(np.asarray(R)@pi)+Q*beta*(1-s)
    q=np.linalg.solve(np.eye(nv)-a*a*F.T,forcing)
    pred=PV.T@q
    rhs=np.vstack([a*(1-s[v])*pred[v]*Be for v in range(nv)])
    cross=np.linalg.solve(np.eye(nv*nz)-a*np.kron(F.T,Acl),rhs).reshape(nv,nz,ne)
    C=cross.sum(axis=0);D=float(q.sum())
    innovation=Acl@C@Be.T+Be@C.T@Acl.T+Be@Be.T*D
    Z=solve_discrete_lyapunov(Acl,innovation)
    Z=(Z+Z.T)/2
    covariance=np.block([[Z,C],[C.T,D*np.eye(ne)]])
    return dict(q=q,cross=cross,Z=Z,C=C,D=D,covariance=covariance)

def output_cost(mom,Hz,He=None):
    Hz=np.asarray(Hz)
    if He is None:return float(np.trace(Hz@mom['Z']@Hz.T))
    H=np.column_stack([Hz,He]);return float(np.trace(H@mom['covariance']@H.T))

def affine_cost(PV,s,R,Acl,Be,Hz,He=None,a=.95,Q=.1):
    """c[u,v] includes the fixed process-noise baseline via sum(pi)=1.
    Basis pi need not be a probability law: linear covariance equations extend
    algebraically, with Q=0 for the basis responses.
    """
    nu,nv=len(R),len(PV);zero=np.zeros((nu,nv))
    base=output_cost(isotropic_moments(PV,s,zero,R,Acl,Be,a,Q),Hz,He)
    c=np.zeros((nu,nv))
    for u in range(nu):
        for v in range(nv):
            pi=zero.copy();pi[u,v]=1
            response=isotropic_moments(PV,s,pi,R,Acl,Be,a,0)
            c[u,v]=base+output_cost(response,Hz,He)
    return c.ravel()

def full_augmented(P,pi,s,R,Acl,Be,a=.95,Q=.1):
    """Independent joint-mode Kronecker solve, for small test systems."""
    ne=Be.shape[1];nz=Acl.shape[0];dim=nz+ne;n=len(P);nv=len(s)
    M0=np.block([[Acl,Be],[np.zeros((ne,nz)),a*np.eye(ne)]])
    M1=M0.copy();M1[nz:,:]=0
    ops=[(1-s[e%nv])*np.kron(M0,M0)+s[e%nv]*np.kron(M1,M1) for e in range(n)]
    T=np.zeros((n*dim*dim,n*dim*dim));rhs=np.zeros(n*dim*dim)
    for f in range(n):
        block=slice(f*dim*dim,(f+1)*dim*dim)
        for e in range(n):T[block,e*dim*dim:(e+1)*dim*dim]=P[e,f]*ops[f]
        H=np.zeros((dim,dim));H[nz:,nz:]=((1-s[f%nv])*Q+s[f%nv]*R[f//nv])*np.eye(ne)
        rhs[block]=pi[f]*H.reshape(-1,order='F')
    x=np.linalg.solve(np.eye(len(T))-T,rhs)
    X=np.array([x[e*dim*dim:(e+1)*dim*dim].reshape(dim,dim,order='F') for e in range(n)])
    return dict(covariance=X.sum(axis=0),mode_covariances=X,residual=float(np.max(np.abs(x-T@x-rhs))))

def autocovariance(PV,s,q,lag,a=.95):
    F=PV@np.diag(1-np.asarray(s))
    return float(a**lag*(np.asarray(q)@(np.linalg.matrix_power(F,lag)@np.ones(len(PV)))))

def platoon_model(N=10,dt=.05,tau=.5,kp=.4,kv=.8,h=1.2,kf=.7):
    aug=np.zeros((4,4));aug[:3,:3]=[[0,1,0],[0,0,1],[0,0,-1/tau]];aug[2,3]=1/tau
    Z=expm(aug*dt);Ad=Z[:3,:3];b=Z[:3,3]
    ks=np.array([-kp,-kp*h-kv,-kv*h]);kt=np.array([kp,kv,kf])
    A=np.kron(np.eye(N),Ad+np.outer(b,ks));B=np.kron(np.eye(N),b[:,None]*kf)
    spacing=np.zeros((N,3*N));velocity=np.zeros_like(spacing);acc=np.zeros_like(spacing)
    for i in range(N):
        spacing[i,3*i:3*i+3]=[-1,-h,0];velocity[i,3*i+1]=-1;acc[i,3*i+2]=1
        if i:
            A[3*i:3*i+3,3*i-3:3*i]=np.outer(b,kt);spacing[i,3*i-3]=1;velocity[i,3*i-2]=1
    return dict(Acl=A,Be=B,spacing=spacing/np.sqrt(N),velocity=velocity/np.sqrt(N),acceleration=acc/np.sqrt(N),
                jerk_z=acc@(A-np.eye(3*N))/(dt*np.sqrt(N)),jerk_e=acc@B/(dt*np.sqrt(N)),N=N,dt=dt)


def general_channel_moments(PV,s,pi,Sigmas,A,W,Acl,Be):
    """Nonisotropic source/control matrices: exact channel-tagged Kronecker solve."""
    nv=len(PV);nz=len(Acl);ne=len(A);dim=nz+ne;beta=stationary(PV);pi=np.asarray(pi).reshape(-1,nv)
    M0=np.block([[Acl,Be],[np.zeros((ne,nz)),A]]);M1=M0.copy();M1[nz:,:]=0
    K=[(1-s[v])*np.kron(M0,M0)+s[v]*np.kron(M1,M1) for v in range(nv)]
    op=np.zeros((nv*dim**2,nv*dim**2));rhs=np.zeros(nv*dim**2)
    for v in range(nv):
        block=slice(v*dim**2,(v+1)*dim**2)
        for w in range(nv):op[block,w*dim**2:(w+1)*dim**2]=PV[w,v]*K[v]
        H=np.zeros((dim,dim));H[nz:,nz:]=(1-s[v])*beta[v]*W+s[v]*sum(pi[u,v]*Sigmas[u] for u in range(len(pi)))
        rhs[block]=H.reshape(-1,order='F')
    xx=np.linalg.solve(np.eye(len(op))-op,rhs)
    X=np.array([xx[v*dim**2:(v+1)*dim**2].reshape(dim,dim,order='F') for v in range(nv)])
    return dict(covariance=X.sum(0),channel_covariances=X,residual=float(np.max(np.abs(xx-op@xx-rhs))))

def general_joint_moments(P,pi,s,Sigmas,A,W,Acl,Be):
    """Independent full environmental lift for comparison with channel reduction."""
    nv=len(s);nz=len(Acl);ne=len(A);dim=nz+ne;n=len(P)
    M0=np.block([[Acl,Be],[np.zeros((ne,nz)),A]]);M1=M0.copy();M1[nz:,:]=0
    op=np.zeros((n*dim**2,n*dim**2));rhs=np.zeros(n*dim**2)
    for f in range(n):
        v=f%nv;u=f//nv;block=slice(f*dim**2,(f+1)*dim**2)
        K=(1-s[v])*np.kron(M0,M0)+s[v]*np.kron(M1,M1)
        for e in range(n):op[block,e*dim**2:(e+1)*dim**2]=P[e,f]*K
        H=np.zeros((dim,dim));H[nz:,nz:]=pi[f]*((1-s[v])*W+s[v]*Sigmas[u]);rhs[block]=H.reshape(-1,order='F')
    xx=np.linalg.solve(np.eye(len(op))-op,rhs)
    X=np.array([xx[e*dim**2:(e+1)*dim**2].reshape(dim,dim,order='F') for e in range(n)])
    return dict(covariance=X.sum(0),mode_covariances=X,residual=float(np.max(np.abs(xx-op@xx-rhs))))
