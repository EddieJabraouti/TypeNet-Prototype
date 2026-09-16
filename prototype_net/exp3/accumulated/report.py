"""Render already-sealed results; never trains, selects, or reevaluates models."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'runs/v1'
    results=json.loads((out/'results.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    dev=json.loads((out/'development.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    colors={'validation':'#176b87','test':'#bd5a26'}
    for role in ('validation','test'):
        ms=[results[role][str(n)]['calibrated'] for n in (5,10)]
        acc=np.array([m['accuracy'] for m in ms])*100
        ci=np.array([m['accuracy_cluster_bootstrap_95ci'] for m in ms])*100
        axes[0].errorbar([5,10],acc,yerr=np.maximum(0,np.stack([acc-ci[:,0],ci[:,1]-acc])),
                         marker='o',capsize=4,label=role.title(),color=colors[role])
        axes[1].plot([5,10],[m['ece_15_equal_width']*100 for m in ms],marker='o',label=role.title(),color=colors[role])
    axes[0].axhline(70,color='gray',linestyle='--',linewidth=1,label='70% target')
    axes[0].set(title='Four-class synthetic severity accuracy',ylabel='Accuracy (%)',ylim=(25,85))
    axes[1].set(title='Calibrated confidence error',ylabel='15-bin ECE (percentage points)',ylim=(0,None))
    for ax in axes:
        ax.set(xlabel='Distinct windows per side (baseline = query)',xticks=[5,10],xlim=(4,11))
        ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Matched participants · windows contain 6–50 keys\nAccuracy intervals resample participants; no extrapolation beyond 10 windows',fontsize=11)
    fig.savefig(HERE/'accumulation_results.png',dpi=170)
    lines=['# Accumulated-window results','',
           'This is a matched 5-versus-10-window experiment. Neither budget reached the requested 70–80% validation/test accuracy. It does not evaluate hundreds of windows or clinical severity.','',
           '## Four-class accuracy and calibration','']
    for role in ('validation','test'):
        lines += [f'### {role.title()}','']
        for n in ('5','10'):
            m=results[role][n]['calibrated']; ci=m['accuracy_cluster_bootstrap_95ci']
            lines += [f'- **{n} baseline + {n} query windows:** accuracy {100*m["accuracy"]:.2f}% (95% participant-bootstrap CI {100*ci[0]:.2f}–{100*ci[1]:.2f}%), ECE {100*m["ece_15_equal_width"]:.2f} percentage points, pooled detection EER {100*m["pooled_detection_eer"]:.2f}%.',
                      f'  Log loss {m["log_loss"]:.4f}; multiclass Brier {m["multiclass_brier"]:.4f}; mean confidence {100*m["mean_confidence"]:.2f}%.',
                      '  Class recall: '+', '.join(f'{k} {100*v:.1f}%' for k,v in m['per_class_recall'].items())+'.']
        d=results[role]['paired_10_minus_5']; ci=d['participant_bootstrap_95ci']
        lines += ['',f'Paired 10-minus-5 improvement: **{100*d["accuracy_difference"]:+.2f} percentage points**, 95% participant-bootstrap CI {100*ci[0]:+.2f} to {100*ci[1]:+.2f}.','']
    lines += ['## Training and selection','',
              'Both models are new gradient-boosted timing-summary trees. Neither uses TypeNet or prior exp3 weights.','']
    for n in ('5','10'):
        lines += [f'- {n} windows per side: fitted-training accuracy {100*dev[n]["train"]["accuracy"]:.2f}%; selection accuracy {100*dev[n]["selection"]["accuracy"]:.2f}%. Training accuracy is in-sample, not evidence of generalization.']
    lines += ['', 'The large training-to-selection gaps indicate substantial overfitting. The measured benefit of more windows does not establish a performance ceiling or imply that this model family is optimal.', '', 'Cohorts: '+', '.join(f'{r} {len(ids):,}' for r,ids in protocol['splits'].items())+'.','',
              'Three configurations per budget were selected using only selection identities. Separate scalar temperatures used calibration identities. Models, temperatures, participant lists, preprocessing and generator hashes were frozen before a single final evaluation. Four labels of a person remain together for splitting and bootstrap uncertainty.','',
              '## Data limitations','',
              'The complete Aalto census found 168,593 people, each with 15 sentences. The second Aalto folder adds no participants or histories. Recovering discarded keystrokes yields enough distinct variable-length windows for this comparison, but only three people have 40 total windows and nobody has 100. The planned 20/50/100/200 windows per side cannot be evaluated at cohort scale.','',
              'Participants in this experiment must have at least ten windows in each chronological sentence half. Both budgets use the same eligible people, nested baseline/query observations, and shared synthetic query profiles. This conditions the results on people with sufficient recording length; it is not directly comparable to earlier exp3 cohorts or models. Windows within one sentence may be correlated. No sentence crosses the baseline/query boundary.','',
              'All final evaluation identities are excluded from the previous exp3 selection/calibration/validation/test assignments. The historical v4 cohort has nevertheless contributed earlier aggregate results: this is not untouched external validation.','',
              'The generator is unchanged, severity is stable throughout each query period, and calibration assumes balanced synthetic classes. These recordings cannot test long-term baseline drift, changing severity, real-world prevalence, or cognitive-health validity. More distinct chronological recordings per person are required for the intended deployment-scale experiment.','',
              '## Verification','',
              '17 control tests passed. A training-only smoke run exercised fitting, calibration, freezing and sealing. Feature generation and inference matched exactly in eight smoke cases (all four classes at both budgets). Existing exp3 frozen sources/results were not modified.','',
              'See `README.md` for methods and inference, and `runs/v1/` for protocol, complete selection history, audits, models, calibrated predictions and metrics.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
