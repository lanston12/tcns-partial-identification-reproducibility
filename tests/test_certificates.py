import unittest,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from coupling import *
from certificates import *
from closed_loop import *

class Certificates(unittest.TestCase):
 def setUp(self):
  self.args=(.05,.15,.45,.35);self.PU,self.PV,self.p,self.q,*_,self.lo,self.hi=binary_data(*self.args)
  self.s=np.array([.9,.2]);self.R=np.array([.05,1.]);self.P=binary_kernel(*self.args,[.02,.035,.40,.58]);self.pi=stationary(self.P)
 def test_retained_exact_estimation(self):
  c=state_cost(self.PV,self.s,self.R);x=cost_interval(self.p,self.q,self.lo,self.hi,c)
  np.testing.assert_allclose(x[:2],[.3056139198849776,.4090165944436552],atol=1e-13)
 def test_refined_binary_flow_dual(self):
  lo=self.lo.copy();hi=self.hi.copy();lo[2]=.37;hi[3]=.61
  c=state_cost(self.PV,self.s,self.R);ends=cost_interval(self.p,self.q,lo,hi,c);L,H=binary_cell_bounds(self.args,lo,hi)
  for mx,v in zip([False,True],ends[:2]):
   r=restricted_flow(self.PU,self.PV,c,L,H,mx);self.assertAlmostEqual(r['value'],v,10)
   self.assertLess(r['primal_residual'],1e-9);self.assertLess(abs(r['duality_gap']),1e-9)
   self.assertLess(r['dual_violation'],1e-9);np.testing.assert_allclose(r['pi']@r['P'],r['pi'],atol=1e-10)
 def test_refined_witness(self):
  lo=self.lo.copy();hi=self.hi.copy();lo[2]=.36;hi[3]=.62
  for t in overlap_interval(self.p,self.q,lo,hi):
   P,pi,theta=refined_witness(self.args,lo,hi,t);np.testing.assert_allclose(pi@P,pi,atol=1e-12)
   self.assertTrue(np.all(theta>=lo-1e-10));self.assertTrue(np.all(theta<=hi+1e-10))
 def test_empty_box_fails(self):
  lo=self.lo.copy();lo[0]=.2
  with self.assertRaises(ValueError):overlap_interval(self.p,self.q,lo,self.hi)
 def test_zero_flow_row_feasibility_checked(self):
  L,H=binary_cell_bounds(self.args,self.lo,self.hi);L[0]=[.8,.1,.1,0];H[0]=L[0]
  with self.assertRaises(ValueError):restricted_flow(self.PU,self.PV,np.ones(4),L,H)
 def test_radius_at_unvisited_and_rate(self):
  r=confidence_radius([0,10,100,1000,10000]);self.assertEqual(r[0],1);self.assertTrue(np.all(np.diff(r)<=0))
  self.assertLess(r[-1],.04)
 def test_nested_dependent_trajectory(self):
  states=generate_trajectory(self.P,20000,123,0);conf=BinaryConfidence(self.args);oldlo=conf.lo.copy();oldhi=conf.hi.copy()
  for e,f in zip(states[:-1],states[1:]):
   conf.observe(e,f);self.assertTrue(np.all(conf.lo>=oldlo));self.assertTrue(np.all(conf.hi<=oldhi));oldlo=conf.lo.copy();oldhi=conf.hi.copy()
  self.assertTrue(np.all(self.P[:,3]>=conf.lo));self.assertTrue(np.all(self.P[:,3]<=conf.hi));self.assertEqual(conf.visits.sum(),20000)
 def test_saved_all_checkpoints_truth_and_nesting(self):
  rows=json.loads((ROOT/'reference/confidence_trace.json').read_text());old={k:float('inf') for k in rows[0]['intervals']}
  for row in rows:
   for name,iv in row['intervals'].items():
    self.assertLessEqual(iv[1]-iv[0],old[name]+1e-12);old[name]=iv[1]-iv[0]
   self.assertLessEqual(row['overlap'][0],self.pi[3]);self.assertGreaterEqual(row['overlap'][1],self.pi[3])
 def test_marginal_IID_slow_joint_counterexample(self):
  eps=1e-5
  for aa,bb,t in [(1,1,.25),(1,3,.375)]:
   P=slow_fair_coupling(eps,aa,bb);pi=stationary(P);self.assertAlmostEqual(pi[3],t,8)
   np.testing.assert_allclose(P[:,[0,1]].sum(1),.5);np.testing.assert_allclose(P[:,[0,2]].sum(1),.5)
   self.assertGreater(P.min(),0)
 def test_arbitrary_initial_trajectory(self):
  st=generate_trajectory(self.P,100,8,3);self.assertEqual(st[0],3);self.assertEqual(len(st),101)

class ColoredLoop(unittest.TestCase):
 setUp=Certificates.setUp
 def test_augmented_matches_reduction_multidimensional(self):
  A=np.array([[.75,.12],[0,.65]]);B=np.array([[.1,.02],[.04,.2]])
  x=isotropic_moments(self.PV,self.s,self.pi,self.R,A,B);y=full_augmented(self.P,self.pi,self.s,self.R,A,B)
  np.testing.assert_allclose(x['covariance'],y['covariance'],atol=1e-12);self.assertLess(y['residual'],1e-12)
  self.assertGreater(np.linalg.norm(x['C']),.001)
 def test_all_same_pi_kernels_same_second_moments(self):
  P1,pi,theta=construct_binary(*self.args,.14);delta=np.array([1,-1,0,0.])*min(pi[:2])*.005/pi
  P2=binary_kernel(*self.args,theta+delta);np.testing.assert_allclose(pi@P2,pi,atol=1e-12)
  A=np.array([[.8]]);B=np.array([[.25]])
  x=full_augmented(P1,pi,self.s,self.R,A,B);y=full_augmented(P2,pi,self.s,self.R,A,B)
  self.assertGreater(np.max(abs(P1-P2)),.001);np.testing.assert_allclose(x['covariance'],y['covariance'],atol=1e-12)
 def test_affine_cost_matches_full_solve(self):
  A=np.array([[.8]]);B=np.array([[.25]]);c=affine_cost(self.PV,self.s,self.R,A,B,np.eye(1))
  for t in [.09375,.12,.16,.1875]:
   P,pi,_=construct_binary(*self.args,t);full=full_augmented(P,pi,self.s,self.R,A,B)
   self.assertAlmostEqual(pi@c,full['covariance'][0,0],12)
 def test_white_proxy_is_inexact(self):
  from scipy.linalg import solve_discrete_lyapunov
  A=np.array([[.8]]);B=np.array([[.25]]);x=isotropic_moments(self.PV,self.s,self.pi,self.R,A,B)
  white=solve_discrete_lyapunov(A,B@B.T*x['D'])[0,0];self.assertGreater(x['Z'][0,0]/white,2)
 def test_error_zero_lag_and_decay(self):
  x=isotropic_moments(self.PV,self.s,self.pi,self.R,np.array([[.8]]),np.array([[.25]]))
  self.assertAlmostEqual(autocovariance(self.PV,self.s,x['q'],0),x['D'],12)
  self.assertGreater(autocovariance(self.PV,self.s,x['q'],1),0);self.assertLess(autocovariance(self.PV,self.s,x['q'],20),1e-5)
 def test_vehicle_fixed_controller_stability_and_jerk(self):
  m=platoon_model();self.assertLess(max(abs(np.linalg.eigvals(m['Acl']))),1)
  rng=np.random.RandomState(11);z=rng.randn(30);e=rng.randn(10);zn=m['Acl']@z+m['Be']@e
  direct=m['acceleration']@(zn-z)/m['dt'];lift=m['jerk_z']@z+m['jerk_e']@e
  np.testing.assert_allclose(direct,lift,atol=1e-14)
 def test_vehicle_all_outputs_nonnegative_and_truth_covered(self):
  config=json.loads((ROOT/'reference/config.json').read_text());summary=json.loads((ROOT/'reference/summary.json').read_text())
  for name in ['spacing','velocity','acceleration','jerk']:
   self.assertGreater(summary['truth'][name],0);lo,hi=summary['final_intervals'][name];self.assertTrue(lo<=summary['truth'][name]<=hi)
 def test_analytic_vs_independent_replication_sanity(self):
  summary=json.loads((ROOT/'reference/summary.json').read_text())
  for j,name in enumerate(['spacing','velocity','acceleration','jerk']):
   self.assertLess(abs(summary['vehicle_sim_mean'][j]-summary['truth'][name]),4*summary['vehicle_sim_SE'][j])


class GeneralLift(unittest.TestCase):
 def test_nonnormal_source_anisotropic_noise_joint_agreement(self):
  args=(.05,.15,.45,.35);PU,PV,*_=binary_data(*args);P,pi,_=construct_binary(*args,.14)
  A=np.array([[.8,.2],[0,.7]]);Acl=np.array([[.6,.1],[-.1,.75]]);Be=np.array([[.2,.1],[.05,-.1]])
  W=np.array([[.1,.03],[.03,.2]]);Sigmas=[np.array([[.05,.01],[.01,.08]]),np.array([[1.,.2],[.2,.7]])];ss=np.array([.9,.2])
  x=general_channel_moments(PV,ss,pi,Sigmas,A,W,Acl,Be);y=general_joint_moments(P,pi,ss,Sigmas,A,W,Acl,Be)
  np.testing.assert_allclose(x['covariance'],y['covariance'],atol=1e-12);self.assertGreater(np.linalg.eigvalsh(x['covariance']).min(),0)
 def test_matrix_controller_identical_to_frozen_reference(self):
  m=platoon_model()
  with np.load(ROOT/'reference/platoon_matrices.npz',allow_pickle=False) as frozen:
   np.testing.assert_array_equal(m['Acl'],frozen['Acl']);np.testing.assert_array_equal(m['Be'],frozen['Be'])
   np.testing.assert_array_equal(m['spacing'],frozen['spacing']);np.testing.assert_array_equal(m['velocity'],frozen['velocity'])

class FiniteData(unittest.TestCase):
 def test_six_state_trajectory_and_bounded_flow(self):
  PU=np.array([[.8,.2],[.3,.7]]);PV=np.array([[.6,.3,.1],[.15,.7,.15],[.1,.2,.7]]);P=np.kron(PU,PV);pi=stationary(P)
  states=generate_trajectory(P,50000,20261002,0);conf=FiniteConfidence(6)
  for e,f in zip(states[:-1],states[1:]):conf.observe(e,f)
  self.assertTrue(np.all(P>=conf.lo-1e-12));self.assertTrue(np.all(P<=conf.hi+1e-12));self.assertEqual(conf.visits.sum(),50000)
  c=state_cost(PV,np.array([.9,.5,.2]),[.05,1.])
  low=restricted_flow(PU,PV,c,conf.lo,conf.hi);high=restricted_flow(PU,PV,c,conf.lo,conf.hi,True)
  self.assertTrue(low['value']<=pi@c<=high['value']);self.assertLess(abs(low['duality_gap']),1e-9);self.assertLess(abs(high['duality_gap']),1e-9)
if __name__=='__main__':unittest.main()


