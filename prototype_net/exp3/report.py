"""Render the already-sealed results. Never fits or evaluates a model."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    directory=HERE/'runs/final_v1'
    results=json.loads((directory/'results.json').read_text())
    frozen=json.loads((directory/'freeze.json').read_text())
    labels=['Normal','Mild','Moderate','Severe']
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    for column,budget in enumerate(('single','bundle')):
        title='One query session' if budget=='single' else 'Five query sessions'
        ax=axes[0,column]
        for kind,color,label in (('raw','#999999','Before calibration'),('calibrated','#176a96','After calibration')):
            bins=results['test'][budget][kind]['reliability']
            x=[b['confidence']*100 for b in bins]; y=[b['accuracy']*100 for b in bins]
            sizes=[8+55*np.sqrt(b['count']/max(z['count'] for z in bins)) for b in bins]
            ax.plot(x,y,color=color,lw=1)
            ax.scatter(x,y,s=sizes,color=color,label=label)
        ax.plot([0,100],[0,100],'--',color='black',lw=.7,label='Perfect calibration')
        ax.set(xlim=(0,100),ylim=(0,100),xlabel='Mean predicted confidence (%)',ylabel='Observed 4-way accuracy (%)',
               title=f'Test confidence reliability — {title}')
        ax.legend(fontsize=8,frameon=False); ax.grid(alpha=.15)
        m=results['test'][budget]['calibrated']; matrix=np.asarray(m['confusion_counts'])
        proportions=matrix/matrix.sum(axis=1,keepdims=True)*100
        ax=axes[1,column]; ax.imshow(proportions,cmap='Blues',vmin=0,vmax=100)
        for i in range(4):
            for j in range(4): ax.text(j,i,f'{proportions[i,j]:.1f}%',ha='center',va='center',color='white' if proportions[i,j]>55 else 'black',fontsize=10)
        ax.set(xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,
               xlabel='Predicted class',ylabel='True synthetic class',title=f'Test confusion, row-normalized — {title}')
    fig.suptitle('Exp3: sealed synthetic-severity evaluation',fontsize=15)
    fig.supxlabel('Source: final_v1/results.json · 3,899 test participants · 15 confidence bins; marker size scales with bin count',fontsize=8)
    fig.savefig(HERE/'calibration_and_confusion.png',dpi=160)
    plt.close(fig)
    lines=['# Exp3 result: the 70–80% target was not met','',
           'Final models and calibration were frozen before the final validation/test pass. Both observation budgets classify all four classes; chance accuracy is 25%.', '',
           '## Sealed scores','']
    for role in ('validation','test'):
        for budget in ('single','bundle'):
            m=results[role][budget]['calibrated']; lo,hi=m['accuracy_cluster_bootstrap_95ci']
            lines.append(f'- **{role.title()}, {budget}: {m["accuracy"]:.2%} accuracy** (participant-bootstrap 95% CI {lo:.2%}–{hi:.2%}); pooled detection EER {m["pooled_detection_eer"]:.2%}; calibration ECE {100*m["ece_15_equal_width"]:.2f} percentage points; NLL {m["log_loss"]:.4f}; multiclass Brier {m["multiclass_brier"]:.4f}. {m["participants"]:,} participants, {m["samples"]:,} labeled observations.')
    lines+=['','The five-session result uses five distinct query sessions under one shared synthetic profile, plus ten clean enrollment sessions. It is not a single-session accuracy. The test confidence interval is below 70%. These experiments do not establish that 70% is mathematically impossible.','',
            '## What was selected','',
            'The final classifier combines a summary-feature boosted tree with the full-gallery temporal model. The sequence-only model supplied initialization but received zero direct ensemble weight.']
    for budget,choice in frozen['choices'].items():
        lines.append(f'- {budget}: tree/sequence-only/full-gallery weights {choice["weights"]}; temperature {choice["temperature"]:.6f}; selection accuracy {choice["selection_accuracy"]:.2%}.')
    lines+=['','The first temporal run used 100 epochs; the full-gallery run stopped at epoch 64 after 20 epochs without selection improvement. The full-gallery checkpoint selected for both budgets is epoch 44. The two temporal training runs took approximately 44.5 minutes, excluding preprocessing, other model searches and final evaluation.', '',
            '## What remains difficult','']
    for budget in ('single','bundle'):
        m=results['test'][budget]['calibrated']
        lines.append(f'- {budget} test recall: '+', '.join(f'{k} {v:.1%}' for k,v in m['per_class_recall'].items())+'.')
    lines+=['','The errors are concentrated in adjacent severity classes, especially mild and moderate. Confidence calibration does not recover missing class separation. Reasonable calibration here applies to the balanced synthetic distribution, not an independently validated clinical population.','',
            '## Integrity and scope','',
            '- The generator and severity labels were unchanged. No source-query twin, latent burden, pause mask or class metadata enters inference.',
            '- All classes share millisecond rounding and terminal-transition exclusion. A selection-only 5-ms stress audit reduced sequence-only accuracy by 0.25 points (single) and 0.17 points (five sessions).',
            '- Training, model selection, calibration, final validation and test use disjoint participants. The final test subset was specified before modeling; 3,899 of 4,000 requested participants passed the predeclared completeness rule.',
            '- This is a subset of the historical v4 test cohort, whose aggregate results were already discussed in earlier experiments. It is not a new external dataset. Exp3 used new synthetic draws and no final-test fitting or selection.',
            '- A queued process imported the full-gallery architecture before a later source edit. This caused an invalid initial checkpoint replay. The actual trained architecture was restored; every weight tensor was unchanged; its selection accuracy and log loss reproduced exactly. The invalid replay, original metadata, recovery record and proof are retained. No final-validation or test data had been opened when this was corrected.',
            '- Twelve control tests passed. Inference matched evaluation exactly on all four synthetic classes and both observation budgets.',
            '- Calibration and test artifacts include full confusion counts, reliability-bin counts, participant-bootstrap intervals and per-observation probabilities. No abstention, relabeling, favorable-seed selection or test-driven retraining was used.', '',
            '## Files and use','',
            '- Main frozen choices and temperatures: `runs/final_v1/freeze.json`.',
            '- Sealed metrics: `runs/final_v1/results.json`; saved probabilities: `runs/final_v1/{validation,test}_{single,bundle}_predictions.npz`.',
            '- Recovery proof: `runs/relational_v1/reproduced_selection.json` and `architecture_recovery.json`.',
            '- Invalid replay: `runs/final_v1_invalid_architecture_replay/` (not a valid experimental trial).',
            '- Inference parity proof: `runs/final_v1/inference_parity.json`.',
            '', '```bash', 'python3 -m prototype_net.exp3.final_predict observed_sessions.npz --budget bundle', '```', '',
            'Inputs are ten clean enrollment sessions and five query sessions for bundle mode. See `final_predict.py` and `README.md` for shapes and the single-session option. Generated models and caches remain on disk and are excluded from Git.', '',
            '![Test reliability and confusion](calibration_and_confusion.png)','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
