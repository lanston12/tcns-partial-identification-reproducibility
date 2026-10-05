"""Generate all six figures exclusively from saved experiment outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'paper/figures'
def read(name):return json.loads((ROOT/name).read_text())
cfg=read('results/raw/config.json');ends=read('results/raw/endpoints.json');trace=read('results/raw/confidence_trace.json');summary=read('results/processed/summary.json');framework=read('results/raw/framework.json');sim=read('results/raw/vehicle_simulation.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#246A9E','#BA4C32','#438B70','#8064A2']
def save(fig,name):
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=210,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(3.5,2.05));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
positions=[(.03,.76,.44,.19),(.53,.76,.44,.19),(.03,.43,.44,.19),(.53,.43,.44,.19),(.03,.1,.44,.19),(.53,.1,.44,.19)]
labels=['Separate sensing\n$P_U$','Separate channel\n$P_V$','Strong consistency\n$\mathcal{P}_{\\rm cons}$','Synchronized path\n$(U_k,V_k)$','Row confidence set\n$\mathcal{P}_N$','Exact task intervals\n$D$, $J_c$, RMS']
for j,(x,y,w,h) in enumerate(positions):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',edgecolor=colors[0],facecolor='#EEF5FA',linewidth=.8));ax.text(x+w/2,y+h/2,labels[j],ha='center',va='center')
for i,j in framework['arrows']:
 x,y,w,h=positions[i];x2,y2,w2,h2=positions[j]
 start=(x+w/2,y) if y>y2 else (x+w,y+h/2);stop=(x2+w2/2,y2+h2) if y>y2 else (x2,y2+h2/2)
 ax.annotate('',xy=stop,xytext=start,arrowprops=dict(arrowstyle='->',lw=.9,color='#555555'))
save(fig,'fig1_framework')
fig,axs=plt.subplots(1,2,figsize=(3.5,2.2),gridspec_kw={'width_ratios':[1.2,1]})
for j,(row,col,label) in enumerate([(trace[0],colors[0],'Separate models'),(trace[-1],colors[1],'$N=10^6$')]):
 for e in range(4):axs[0].plot([row['lo'][e],row['hi'][e]],[e+j*.18,e+j*.18],lw=3,color=col,label=label if e==0 else None)
axs[0].plot(cfg['true_theta'],np.arange(4)+.09,'k.',ms=4,label='Truth');axs[0].set_yticks(range(4));axs[0].set_yticklabels(['00','01','10','11']);axs[0].set(xlabel=r'Row $\theta_e$',ylabel='Current pair $e$');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.08),ncol=3,frameon=False,fontsize=6)
for j,row in enumerate([trace[0],trace[-1]]):axs[1].plot(row['overlap'],[j,j],lw=5,color=colors[j])
axs[1].axvline(summary['true_t'],ls='--',color='k',lw=.8);axs[1].set_yticks([0,1]);axs[1].set_yticklabels(['$N=0$','$10^6$']);axs[1].set(xlabel=r'Stationary overlap $t$',xlim=(.08,.205));fig.tight_layout(w_pad=.5);save(fig,'fig2_ambiguity')
fig,axs=plt.subplots(1,2,figsize=(3.5,2.0));ts=[ends[0]['t'],cfg['p']*cfg['q'],ends[1]['t']];ps=np.array([[1-cfg['p']-cfg['q']+t,cfg['q']-t,cfg['p']-t,t] for t in ts]);labels=['Low $t$','Product','High $t$']
for ax,key,ylabel in zip(axs,['estimation','generic_control'],['Estimation energy $D$','Closed-loop energy $J_c$']):
 vals=ps@np.array(cfg['costs'][key]);ax.bar(range(3),vals,color=[colors[1],colors[2],colors[0]],width=.65);ax.set_xticks(range(3));ax.set_xticklabels(labels,rotation=22);ax.set_ylabel(ylabel);ax.set_ylim(0,max(vals)*1.2)
fig.tight_layout(w_pad=.6);save(fig,'fig3_same_models')
fig,axs=plt.subplots(1,2,figsize=(3.5,2.05));N=np.array([x['N'] for x in trace[1:]]);width=np.array([np.diff(x['intervals']['estimation'])[0] for x in trace[1:]]);initial=np.diff(trace[0]['intervals']['estimation'])[0]
axs[0].semilogx(N,width,'o-',color=colors[0],ms=3);axs[0].axhline(initial,color='#777777',ls=':',lw=.8,label='$N=0$ width');axs[0].set(xlabel='Synchronized transitions $N$',ylabel='Estimation interval width');axs[0].legend(frameon=False)
cover=read('results/raw/coverage_replicates.json');xx=[x['N'] for x in cover['snapshots']];yy=[x['row_coverage_fraction'] for x in cover['snapshots']]
axs[1].semilogx(xx,yy,'o-',color=colors[2],ms=3,label='200 paths');axs[1].axhline(.95,color=colors[1],ls='--',lw=.8,label='Guarantee');axs[1].set(xlabel='Transitions per path',ylabel='Empirical anytime coverage',ylim=(.93,1.015));axs[1].legend(loc='lower right',frameon=False);fig.tight_layout(w_pad=.7);save(fig,'fig4_data_width')
fig,axs=plt.subplots(1,2,figsize=(3.5,2.05));tt=np.linspace(ends[0]['t'],ends[1]['t'],101);pp=np.array([[1-cfg['p']-cfg['q']+t,cfg['q']-t,cfg['p']-t,t] for t in tt]);D=pp@cfg['costs']['estimation'];J=pp@cfg['costs']['generic_control'];white=D*.25**2/(1-.8**2)
axs[0].plot(D,J,color=colors[0],label='Exact colored');axs[0].plot(D,white,'--',color=colors[1],label='White proxy');axs[0].plot(summary['truth']['estimation'],summary['truth']['generic_control'],'k.',ms=5);axs[0].set(xlabel='Estimation energy $D$',ylabel='Closed-loop energy $J_c$');axs[0].legend(frameon=False)
for e,col in zip(ends,[colors[1],colors[0]]):axs[1].semilogy(range(21),e['autocov'],'o-',ms=2,color=col,label='$t=$'+format(e['t'],'.5f'))
axs[1].set(xlabel='Error lag $\ell$',ylabel=r'$\Gamma_e(\ell)$');axs[1].legend(frameon=False);fig.tight_layout(w_pad=.7);save(fig,'fig5_colored_propagation')
fig,axs=plt.subplots(2,2,figsize=(3.5,3.3));names=['spacing','velocity','acceleration','jerk'];units=['m','m/s','m/s$^2$','m/s$^3$']
for j,(ax,name,unit) in enumerate(zip(axs.ravel(),names,units)):
 iv=np.sqrt(np.array([x['intervals'][name] for x in trace[1:]]));true=np.sqrt(summary['truth'][name]);ax.fill_between(N,iv[:,0],iv[:,1],color=colors[0],alpha=.22,label='95% confidence interval');ax.plot(N,iv[:,0],color=colors[0],lw=.8);ax.plot(N,iv[:,1],color=colors[0],lw=.8);ax.axhline(true,color='k',ls='--',lw=.9,label='Exact truth');mc=np.sqrt(sim['mean'][j]);ax.plot(N[-1],mc,'o',color=colors[1],ms=3,label='Independent MC');ax.set_xscale('log');ax.set(xlabel='Synchronized transitions $N$',ylabel=name.capitalize()+' RMS ('+unit+')')
 if j==0:fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.045),ncol=3,frameon=False,fontsize=6)
fig.tight_layout(w_pad=.6,h_pad=.65);save(fig,'fig6_platoon')
(ROOT/'results/processed/figure_data.json').write_text(json.dumps(dict(curve_t=tt.tolist(),curve_D=D.tolist(),curve_J=J.tolist(),curve_white=white.tolist(),data_N=N.tolist(),data_width=width.tolist(),source_files=['config.json','endpoints.json','confidence_trace.json','framework.json','coverage_replicates.json','vehicle_simulation.json']),indent=2),encoding='utf8')
print('six PDF/PNG figures generated from saved raw outputs')
