"""Independent TCNS experiment; figures read only saved outputs. No seed search."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from coupling import *
from certificates import *
from closed_loop import *

def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save(path,x):
 (ROOT/path).write_text(json.dumps(x,indent=2,default=plain),encoding='utf8')

def coverage(P,args,costs,N=50000,reps=200):
 rng=np.random.RandomState(20260930);idx=np.arange(reps);states=np.zeros(reps,int)
 visits=np.zeros((reps,4),int);success=np.zeros_like(visits)
 *_,low,high=binary_data(*args);lo=np.tile(low,(reps,1));hi=np.tile(high,(reps,1));theta=P[:,3]
 ever=np.zeros(reps,bool);checkpoints=[100,1000,10000,N];snap=[];cdf=np.cumsum(P,1)
 for k in range(1,N+1):
  nxt=(rng.random_sample(reps)[:,None]>cdf[states]).sum(1)
  visits[idx,states]+=1;success[idx,states]+=(nxt==3)
  m=visits[idx,states];hat=success[idx,states]/m;eps=confidence_radius(m)
  lo[idx,states]=np.maximum(lo[idx,states],hat-eps);hi[idx,states]=np.minimum(hi[idx,states],hat+eps)
  ever|=np.any((theta<lo-1e-12)|(theta>hi+1e-12),axis=1);states=nxt
  if k in checkpoints:
   rows=[];empty=0;contains=0
   for j in range(reps):
    try:
     di=cost_interval(.25,.5625,lo[j],hi[j],costs['estimation'])[:2]
     rows.append(di);contains+=di[0]-1e-12<=stationary(P)@costs['estimation']<=di[1]+1e-12
    except ValueError:rows.append([None,None]);empty+=1
   snap.append(dict(N=k,row_coverage_fraction=float(1-ever.mean()),performance_coverage_fraction=contains/reps,
                    empty_sets=empty,intervals=rows,visits=visits.copy(),lo=lo.copy(),hi=hi.copy()))
 return dict(seed=20260930,reps=reps,N=N,delta=.05,anytime_row_failures=int(ever.sum()),snapshots=snap)

def simulate_vehicle(P,pi,model,s,R,a=.95,Q=.1,reps=20,burn=10000,T=100000):
 rng=np.random.RandomState(20261001);N=model['N'];nz=len(model['Acl']);cdf=np.cumsum(P,1)
 states=rng.choice(4,size=reps,p=pi);z=np.zeros((reps,nz));err=np.zeros((reps,N))
 names=['spacing','velocity','acceleration','jerk'];total=np.zeros((reps,4));trace=[]
 for k in range(burn+T):
  if k>=burn:
   outputs=[z@model[x].T for x in names[:3]]+[z@model['jerk_z'].T+err@model['jerk_e'].T]
   values=np.column_stack([np.sum(o*o,axis=1) for o in outputs]);total+=values
   if k<burn+12000 and (k-burn)%10==0:trace.append([k-burn,*values[0]])
  z=z@model['Acl'].T+err@model['Be'].T
  nxt=(rng.random_sample(reps)[:,None]>cdf[states]).sum(1)
  delivered=rng.random_sample(reps)<np.asarray(s)[nxt%2]
  propagated=a*err+np.sqrt(Q)*rng.standard_normal(err.shape)
  reset=np.sqrt(np.asarray(R)[nxt//2])[:,None]*rng.standard_normal(err.shape)
  err=np.where(delivered[:,None],reset,propagated);states=nxt
 means=total/T
 return dict(seed=20261001,reps=reps,burn=burn,T=T,metrics=names,replicate_energies=means,
             mean=means.mean(0),standard_error=means.std(0,ddof=1)/np.sqrt(reps),
             mean_RMS=np.sqrt(means.mean(0)),trace_replica0=np.array(trace),
             note='Independent replicate means; SE is descriptive, not a dependent-slot confidence proof. Finite burn and finite horizon are disclosed.')

if __name__=='__main__':
 start=time.time();args=(.05,.15,.45,.35);s=np.array([.9,.2]);R=np.array([.05,1.]);a=.95;Q=.1
 PU,PV,p,q,aa,bb,lo,hi=binary_data(*args);theta=np.array([.02,.035,.40,.58]);P=binary_kernel(*args,theta);pi=stationary(P)
 genericA=np.array([[.8]]);genericB=np.array([[.25]]);model=platoon_model()
 costs={'estimation':state_cost(PV,s,R),'generic_control':affine_cost(PV,s,R,genericA,genericB,np.eye(1))}
 for name in ['spacing','velocity','acceleration']:
  costs[name]=affine_cost(PV,s,R,model['Acl'],model['Be'],model[name])
 costs['jerk']=affine_cost(PV,s,R,model['Acl'],model['Be'],model['jerk_z'],model['jerk_e'])
 config=dict(args=args,s=s,R=R,a=a,Q=Q,PU=PU,PV=PV,true_theta=theta,true_P=P,true_pi=pi,
             p=p,q=q,delta=.05,trajectory_seed=20260928,initial_state=0,
             declared_joint_pstar=.01,true_min_transition=float(P.min()),costs=costs,
             controller_parameters=dict(N=10,dt=.05,tau=.5,kp=.4,kv=.8,h=1.2,kf=.7),
             controller_spectral_radius=float(max(abs(np.linalg.eigvals(model['Acl'])))))
 save('results/raw/config.json',config);np.savez_compressed(ROOT/'results/raw/platoon_matrices.npz',**model)
 print('saved model and exact affine costs',flush=True)
 endpts=[]
 for t in binary_interval(*args):
  Pe,pie,the=construct_binary(*args,t);mom=isotropic_moments(PV,s,pie,R,genericA,genericB)
  full=full_augmented(Pe,pie,s,R,genericA,genericB)
  white=solve_discrete_lyapunov(genericA,genericB@genericB.T*mom['D'])[0,0]
  endpts.append(dict(t=t,P=Pe,pi=pie,theta=the,metrics={k:float(pie@c) for k,c in costs.items()},
                    generic_white_cost=white,augmented_difference=float(np.max(np.abs(full['covariance']-mom['covariance']))),
                    autocov=[autocovariance(PV,s,mom['q'],j) for j in range(21)]))
 save('results/raw/endpoints.json',endpts)
 states=generate_trajectory(P,1000000,20260928,0);np.savez_compressed(ROOT/'results/raw/synchronized_trajectory.npz',states=states)
 conf=BinaryConfidence(args);checkpoints=set([0,100,300,1000,3000,10000,30000,100000,300000,1000000]);snaps=[];dual=[]
 def checkpoint():
  snap=conf.snapshot();snap['intervals']={};snap['witnesses']={}
  for name,c in costs.items():
   left,right,ends=cost_interval(p,q,conf.lo,conf.hi,c);snap['intervals'][name]=[left,right]
  snap['overlap']=overlap_interval(p,q,conf.lo,conf.hi)
  for side,t in zip(['left','right'],snap['overlap']):
   Pw,pw,thw=refined_witness(args,conf.lo,conf.hi,t);snap['witnesses'][side]=dict(P=Pw,pi=pw,theta=thw)
  L,H=binary_cell_bounds(args,conf.lo,conf.hi)
  for name in ['estimation','spacing']:
   for maximize in [False,True]:
    r=restricted_flow(PU,PV,costs[name],L,H,maximize);r.update(N=conf.steps,metric=name);dual.append(r)
  snaps.append(snap)
 checkpoint()
 for k in range(len(states)-1):
  conf.observe(int(states[k]),int(states[k+1]))
  if conf.steps in checkpoints:checkpoint()
 save('results/raw/confidence_trace.json',snaps);save('results/raw/flow_certificates.json',dual)
 print('saved million-step dependent trajectory and nested exact certificates',flush=True)
 cov=coverage(P,args,costs);save('results/raw/coverage_replicates.json',cov)
 print('coverage failures',cov['anytime_row_failures'],'/',cov['reps'],flush=True)
 sim=simulate_vehicle(P,pi,model,s,R);save('results/raw/vehicle_simulation.json',sim)
 print('vehicle independent replicate means',sim['mean'],flush=True)
 truth={name:float(pi@c) for name,c in costs.items()}
 summary=dict(truth=truth,true_t=float(pi[3]),initial_intervals=snaps[0]['intervals'],final_intervals=snaps[-1]['intervals'],
              trajectory_counts=snaps[-1]['visits'],coverage_failures=cov['anytime_row_failures'],coverage_reps=cov['reps'],
              vehicle_truth_RMS={k:float(np.sqrt(truth[k])) for k in ['spacing','velocity','acceleration','jerk']},
              vehicle_sim_mean=sim['mean'],vehicle_sim_SE=sim['standard_error'],
              white_generic_true=float(solve_discrete_lyapunov(genericA,genericB@genericB.T*truth['estimation'])[0,0]),
              runtime_seconds=time.time()-start)
 save('results/processed/summary.json',summary)
 save('results/raw/framework.json',dict(nodes=['Separate sensing model','Separate channel model','Strong consistency set','Synchronized trajectory','Nested confidence set','Exact estimation + colored closed-loop intervals'],arrows=[[0,2],[1,2],[2,4],[3,4],[4,5]]))
 print(json.dumps(summary,default=plain),flush=True)
