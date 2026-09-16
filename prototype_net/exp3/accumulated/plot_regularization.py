"""Plot saved matched comparison results without fitting or selecting anything."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
    old=json.loads((HERE/'runs/v1/development.json').read_text())
    new=json.loads((HERE/'runs/regularization_v1/development.json').read_text())
    results=json.loads((HERE/'runs/regularization_v1/results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,key in zip(axes,['5','10']):
        a=[old[key][r]['accuracy'] for r in ['train','selection']]+[results[r][key]['baseline']['accuracy'] for r in ['validation','test']]
        b=[new[key][r]['raw']['accuracy'] for r in ['train','selection']]+[results[r][key]['regularized']['accuracy'] for r in ['validation','test']]
        x=np.arange(4)
        for vals,offset,color,label in [(a,-.19,'#a6611a','Original'),(b,.19,'#176b87','Regularized')]:
            bars=ax.bar(x+offset,np.array(vals)*100,width=.36,color=color,label=label)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set(xticks=x,xticklabels=['Train','Selection','Validation','Test'],ylim=(0,106),ylabel='Four-class accuracy (%)',title=f'{key} baseline + {key} query windows')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper right')
    fig.suptitle('Regularization with identical data in every role\nValidation/test are previously inspected comparison sets; values come from saved run artifacts',fontsize=11)
    fig.savefig(HERE/'regularization_comparison.png',dpi=170)

if __name__=='__main__':main()
