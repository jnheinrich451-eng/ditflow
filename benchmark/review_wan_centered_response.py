"""CPU review of the four-dose response experiment; no model execution."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.wan_centered_attribution import digest, read_references, regions
from benchmark.wan_centered_gradients import compare
from benchmark.wan_centered_huber import scores, separation
from benchmark.wan_centered_response import loss_response, torso_response


def check_values(actual, expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:
            check_values(actual[key],expected[key])
    elif isinstance(actual,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            check_values(a,b)
    elif actual is None or isinstance(actual,str):
        assert actual==expected
    else:
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-10)


def read_npz(path):
    with np.load(path,allow_pickle=False) as values:
        return {k:values[k].copy() for k in values.files}


def review(run, pilot, gradients):
    run,pilot,gradients=map(Path,(run,pilot,gradients))
    report=json.loads((run/'response_report.json').read_text())
    protocol=json.loads((run/'started.json').read_text())
    provenance={}
    for prefix,root,checks in [('pilot',pilot,protocol['inputs_sha256']),
        ('gradients',gradients,protocol['gradient_artifacts_sha256']),
        ('source',Path(__file__).parent,{**protocol['helper_sources_sha256'],'wan_centered_response.py':protocol['script_sha256']})]:
        for name,expected in checks.items():
            path=root/name
            provenance[prefix+'/'+name]=('match' if digest(path)==expected else 'MISMATCH') if path.is_file() else 'not supplied locally'
    if 'MISMATCH' in provenance.values():
        raise ValueError(provenance)
    refs=read_references(pilot)
    base=read_npz(gradients/'readout_derivatives.npz')
    before=read_npz(gradients/'before_flow.npz')['flow']
    np.testing.assert_array_equal(before,base['flow'])
    gs={a:read_npz(gradients/f'{a}_latent_gradient.npz')['gradient'] for a in refs}
    common=refs['forward'][1]&refs['reverse'][1]
    fields,deltas,conditions={},{},{}
    for name,row in report['conditions'].items():
        arm=row['arm']
        fields[name]=read_npz(run/name/'flow.npz')['flow']
        deltas[name]=read_npz(run/name/'actual_delta.npz')
        diagnostics=read_npz(run/name/'readout_derivatives.npz')
        assert all(np.isfinite(v).all() for v in [fields[name],*deltas[name].values(),*diagnostics.values()])
        np.testing.assert_array_equal(diagnostics['flow'],fields[name])
        check_values(loss_response(before,fields[name],refs,gs,deltas[name]['fp32']),row['losses'])
        check_values(torso_response(before,fields[name],*refs[arm],common,base,diagnostics,arm),row['torso'])
        for key,column in [('fp32','delta_fp32'),('after_cast','delta_after_cast')]:
            values=deltas[name][key].astype(float)
            check_values(dict(rms=float(np.sqrt((values**2).mean())),max_abs=float(np.abs(values).max()),
                changed_fraction=float((values!=0).mean())),row[column])
        check_values(compare(deltas[name]['fp32'],deltas[name]['after_cast'])['cosine'],row['cast_delta_vs_fp32_cosine'])
        own=row['losses'][arm]
        # A supplementary smooth-surrogate dot product at actual model-input changes.
        # NOT a derivative of discrete rounding or a new matched finite-difference test.
        cast_prediction=float((gs[arm].astype(float)*deltas[name]['after_cast']).sum())
        sharp=row['torso']['before_sharp']
        without=row['torso']['common_without_outlier']
        conditions[name]=dict(fp32_rms=row['delta_fp32']['rms'],cast_rms=row['delta_after_cast']['rms'],
            cast_changed_fraction=row['delta_after_cast']['changed_fraction'],cast_cosine=row['cast_delta_vs_fp32_cosine'],
            own_loss_before=own['before']['huber_delta1'],own_loss_after=own['after']['huber_delta1'],
            loss_change=own['actual_huber_change'],fp32_linear_prediction=own['first_order_huber_change'],
            observed_to_fp32_prediction=own['observed_to_predicted_ratio'],
            cast_delta_linear_prediction=cast_prediction,
            observed_to_cast_prediction=own['actual_huber_change']/cast_prediction if cast_prediction else None,
            sharp_count=sharp['count'],sharp_still_above_099=sharp['after_same_probability_above_099_count'],
            sharp_loss_before=sharp['before']['huber_delta1'],sharp_loss_after=sharp['after']['huber_delta1'],
            sharp_gain=sharp['change_projection_gain'],
            sharp_sensitivity_before=sharp['before_median_sensitivity'],sharp_sensitivity_after=sharp['after_median_sensitivity'],
            common_without_outlier_huber_before=without['before']['huber_delta1'],
            common_without_outlier_huber_after=without['after']['huber_delta1'],
            common_without_outlier_mse_before=without['before']['mse'],
            common_without_outlier_mse_after=without['after']['mse'],
            outlier_flow=row['dominated_patch_flow'])
    doses={}
    excluded={a:(f,m.copy()) for a,(f,m) in refs.items()}
    for _,mask in excluded.values():
        mask[2,12*52+33]=False
    for label,row in report['by_dose'].items():
        f,r='forward_'+label,'reverse_'+label
        measured=separation(fields[f],fields[r],refs)
        without=separation(fields[f],fields[r],excluded)
        check_values(measured,row['separation'])
        check_values(without,row['separation_excluding_shared_patch'])
        for key,column in [('fp32','fp32_update_comparison'),('after_cast','cast_update_comparison')]:
            check_values(compare(deltas[f][key],deltas[r][key]),row[column])
        # Normalize gains by measured joint update RMS for an efficiency diagnostic,
        # not as a motion acceptance criterion or excuse to select a dose post hoc.
        joint_rms=float(np.sqrt(np.mean(np.r_[deltas[f]['after_cast'].ravel(),deltas[r]['after_cast'].ravel()].astype(float)**2)))
        doses[label]=dict(separation=measured,without_outlier=without,
            fp32_cosine=row['fp32_update_comparison']['cosine'],cast_cosine=row['cast_update_comparison']['cosine'],
            joint_cast_rms=joint_rms,without_outlier_gain_per_cast_rms=without['all_common']['projection_gain']/joint_rms)
    result=dict(provenance=provenance,counts=report['counts'],conditions=conditions,doses=doses,
        limitations='Cast deltas and flow responses were saved by the GPU run. Original before latents are not supplied locally, '
        'so casts cannot be independently reconstructed here. Dot products are local smooth-autograd predictions; '
        'they do not differentiate discrete BF16 rounding. No new model, timing, precision or decoded comparison was run.')
    out=run/'review'; out.mkdir(exist_ok=True)
    (out/'review_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot(result,out)
    print('PROVENANCE',provenance)
    for name,row in conditions.items():
        print(name,json.dumps(row))
    for label,row in doses.items():
        print(label,json.dumps({**row,'separation':row['separation']['all_common'],'without_outlier':row['without_outlier']['all_common']}))
    return result


def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    labels=['Original full','Original tenth','Reverse full','Reverse tenth']
    rows=[result['conditions'][name] for name in ('forward_full','forward_tenth','reverse_full','reverse_tenth')]
    x=np.arange(4)
    axs[0,0].bar(x,[r['cast_cosine'] for r in rows],color=['#348abd','#8fc5e3']*2)
    axs[0,0].set(title='Casting changes the proposed direction',ylabel='Cosine: actual cast delta vs FP32 delta',ylim=(0,1))
    axs[0,1].bar(x,[100*r['cast_changed_fraction'] for r in rows],color=['#348abd','#8fc5e3']*2)
    axs[0,1].set(title='How many model-input values change?',ylabel='Percent changed after casting',ylim=(0,100))
    axs[1,0].bar(x,[100*(r['sharp_loss_after']-r['sharp_loss_before'])/r['sharp_loss_before'] for r in rows],color=['#e24a33','#f8b09d']*2)
    axs[1,0].set(title='Initially sharp torso queries remain poorly corrected',ylabel='Relative own Huber error change (%)')
    axs[1,0].axhline(0,color='black',lw=.8)
    for ax in (axs[0,0],axs[0,1],axs[1,0]):
        ax.set_xticks(x,labels,rotation=15)
    allg=[result['doses'][d]['separation']['all_common']['projection_gain'] for d in ('full','tenth')]
    nog=[result['doses'][d]['without_outlier']['all_common']['projection_gain'] for d in ('full','tenth')]
    axs[1,1].bar(np.arange(2)-.18,allg,width=.36,label='All common torso')
    axs[1,1].bar(np.arange(2)+.18,nog,width=.36,label='Shared patch excluded')
    axs[1,1].set_xticks([0,1],['Full dose','Tenth dose'])
    axs[1,1].set(title='Reference-aligned separation stays small',ylabel='Projection gain (1 = requested separation)')
    axs[1,1].axhline(0,color='black',lw=.8)
    axs[1,1].legend()
    fig.suptitle('Four-forward response review: numerical/readout evidence, not decoded motion transfer')
    fig.savefig(out/'response_diagnostic.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--pilot',default='probe_runs/wan_centered_pilot_camel_s1')
    p.add_argument('--gradients',default='probe_runs/wan_centered_gradients_camel_s1')
    a=p.parse_args()
    review(a.run,a.pilot,a.gradients)
