"""Optional scientific error-bar figure from the completed new-run results."""
import json
import os
from common import ROOT, REPRESENTATIONS, MODELS, MODES, CONDITIONS
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results'/'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows=json.loads((ROOT/'results'/'summary.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    labels=['Clean','White 20 dB','White 10 dB','Pink 10 dB']
    for i,rep in enumerate(REPRESENTATIONS):
        for j,model in enumerate(MODELS):
            ax=axes[i,j]
            for mode,style,color in [('clean_matched','--','#ba651a'),('augmented','-','#245eb0')]:
                selected=[next(r for r in rows if r['run_id']=='main' and r['representation']==rep
                               and r['model']==model and r['mode']==mode and r['condition']==c) for c in CONDITIONS]
                ax.errorbar(range(4),[r['macro_f1_mean'] for r in selected],
                            yerr=[r['macro_f1_sd'] for r in selected],marker='o',linestyle=style,color=color,
                            capsize=3,label='Clean training' if mode=='clean_matched' else 'Clean + white training')
            ax.set_title(f'{rep} | {model}');ax.set_ylim(0,1.04)
            ax.set_xticks(range(4),labels,rotation=20);ax.set_ylabel('Macro-F1 (mean +/- fold SD)')
            ax.set_xlabel('Test condition (unseen speakers)');ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('AudioMNIST | Fresh 10 x 3 speaker-grouped nested CV',fontsize=15)
    fig.savefig(ROOT/'results'/'robustness_errorbars.png',dpi=160)
    plt.close(fig)
    print('FIGURE_SAVED')


if __name__=='__main__':main()
