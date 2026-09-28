"""Reproduce the two additional frozen validation artifacts (no new scenarios)."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from coupling import *
from certificates import *
from closed_loop import *
def write(name,obj):
 (ROOT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')
cfg=json.loads((ROOT/'results/raw/config.json').read_text(encoding='utf-8'))
# General nonbinary trajectory confidence LP; saved separately from binary figures.
PU=np.array([[.8,.2],[.3,.7]]);PV=np.array([[.6,.3,.1],[.15,.7,.15],[.1,.2,.7]]);P=np.kron(PU,PV);pi=stationary(P)
states=generate_trajectory(P,50000,20261002,0);conf=FiniteConfidence(6)
for e,f in zip(states[:-1],states[1:]):conf.observe(e,f)
c=state_cost(PV,np.array([.9,.5,.2]),[.05,1.]);lo=restricted_flow(PU,PV,c,conf.lo,conf.hi);hi=restricted_flow(PU,PV,c,conf.lo,conf.hi,True)
np.savez_compressed(ROOT/'results/raw/six_state_trajectory.npz',states=states)
write('results/raw/six_state_check.json',dict(seed=20261002,N=50000,PU=PU,PV=PV,P=P,pi=pi,c=c,confidence=conf.snapshot(),lower=lo,upper=hi,true_D=pi@c))
# Save the nonnormal/anisotropic lift check rather than relying solely on test text.
PV=np.array(cfg['PV']);P=np.array(cfg['true_P']);pi=np.array(cfg['true_pi']);s=np.array(cfg['s'])
A=np.array([[.8,.2],[0,.7]]);Acl=np.array([[.6,.1],[-.1,.75]]);Be=np.array([[.2,.1],[.05,-.1]]);W=np.array([[.1,.03],[.03,.2]]);Sigmas=[np.array([[.05,.01],[.01,.08]]),np.array([[1.,.2],[.2,.7]])]
ch=general_channel_moments(PV,s,pi,Sigmas,A,W,Acl,Be);jo=general_joint_moments(P,pi,s,Sigmas,A,W,Acl,Be)
diff=float(np.max(abs(ch['covariance']-jo['covariance'])));assert diff<1e-10
write('results/raw/general_matrix_lift_check.json',dict(A=A,Acl=Acl,Be=Be,W=W,Sigmas=Sigmas,channel=ch,joint=jo,max_difference=diff))
