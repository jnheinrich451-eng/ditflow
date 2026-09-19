"""Review returned gradient evidence and reconstruct first Adam proposals on CPU."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmark.wan_centered_attribution import digest, read_references, regions


def compare(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    shared = float(np.linalg.norm((a+b)/2))
    return dict(forward_norm=na, reverse_norm=nb,
        cosine=float((a*b).sum()/(na*nb)) if na*nb else None,
        differential_to_shared_ratio=float(np.linalg.norm((a-b)/2)/shared) if shared else None)


def aggregate(rows):
    aa = sum(r['forward_norm']**2 for r in rows)
    bb = sum(r['reverse_norm']**2 for r in rows)
    ab = sum((r['cosine'] or 0)*r['forward_norm']*r['reverse_norm'] for r in rows)
    return dict(forward_norm=math.sqrt(aa), reverse_norm=math.sqrt(bb),
        cosine=ab/math.sqrt(aa*bb), differential_to_shared_ratio=math.sqrt((aa+bb-2*ab)/(aa+bb+2*ab)))


def review(run, pilot, huber):
    run, pilot, huber = map(Path, (run,pilot,huber))
    report = json.loads((run/'gradient_report.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    old = json.loads((huber/'huber_report.json').read_text())
    checks = {}
    for name, expected in protocol['inputs_sha256'].items():
        path = pilot/name
        checks['pilot/'+name] = ('match' if digest(path)==expected else 'MISMATCH') if path.exists() else 'not supplied locally'
    checks['huber_report'] = 'match' if digest(huber/'huber_report.json')==protocol['huber_report_sha256'] else 'MISMATCH'
    for name, expected in {**protocol['helper_sources_sha256'], 'wan_centered_gradients.py':protocol['script_sha256']}.items():
        checks['source/'+name] = 'match' if digest(Path(__file__).with_name(name))==expected else 'MISMATCH'
    if 'MISMATCH' in checks.values():
        raise ValueError(checks)
    refs = read_references(pilot)
    with np.load(run/'readout_derivatives.npz',allow_pickle=False) as data:
        fields = {k:data[k].copy() for k in data.files}
    with np.load(run/'before_flow.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(fields['flow'],data['flow'])
    assert all(np.isfinite(v).all() for v in fields.values())
    assert fields['same_probability'].min() >= 0 and fields['same_probability'].max() <= 1.000001
    assert fields['flow_jacobian_frobenius'].min() >= 0
    assert json.loads((run/'readout_analysis.json').read_text()) == report['readout']
    gradients, proposals, arms = {}, {}, {}
    roi = regions()
    for arm,(target,valid) in refs.items():
        np.testing.assert_array_equal(valid,fields[arm+'_valid'])
        with np.load(huber/arm/'flow_state_00.npz') as data:
            np.testing.assert_array_equal(data['flow'],fields['flow'])
        with np.load(run/f'{arm}_latent_gradient.npz') as data:
            gradient = data['gradient'].copy()
        assert gradient.shape == (1,16,6,60,104) and np.isfinite(gradient).all()
        gradients[arm] = gradient
        # First Adam update, zero moments, default eps/betas/weight_decay, no latent/cast yet.
        proposals[arm] = -.001*gradient/(np.abs(gradient)+1e-8)
        rms = float(np.sqrt(np.mean(proposals[arm].astype(float)**2)))
        mask = valid & roi['torso']
        saturated = mask & (fields['same_probability'] > .99)
        other = mask & ~saturated
        sensitivity = fields['flow_jacobian_frobenius']
        gradient_rows = fields[arm+'_centered_score_gradient_norm'].astype(float)
        energy = gradient_rows**2
        regions_summary = {}
        for name,region in roi.items():
            selected = valid & region
            e = energy[selected]
            indices = np.argsort(np.where(selected,energy,-1).ravel())[::-1][:5]
            top = []
            for idx in indices:
                pair,pos = divmod(int(idx),1560)
                y,x = divmod(pos,52)
                top.append(dict(pair=[pair,pair+1], row=y, column=x,
                    region_energy_share=float(energy[pair,pos]/e.sum()),
                    same_probability=float(fields['same_probability'][pair,pos]),
                    sensitivity=float(sensitivity[pair,pos]),
                    residual=float(fields[arm+'_residual_norm'][pair,pos])))
            regions_summary[name] = dict(count=int(selected.sum()),
                share_of_global_centered_score_gradient_energy=float(e.sum()/energy.sum()),
                top_patches=top)
        arms[arm] = dict(valid_torso_count=int(mask.sum()), sharp_torso_count=int(saturated.sum()),
            sharp_fraction=float(saturated.sum()/mask.sum()),
            sharp_median_sensitivity=float(np.median(sensitivity[saturated])),
            other_median_sensitivity=float(np.median(sensitivity[other])),
            median_sensitivity_ratio_other_to_sharp=float(np.median(sensitivity[other])/np.median(sensitivity[saturated])),
            sharp_median_residual=float(np.median(fields[arm+'_residual_norm'][saturated])),
            sharp_min_residual=float(fields[arm+'_residual_norm'][saturated].min()),
            sharp_share_of_torso_centered_score_gradient_energy=float(energy[saturated].sum()/energy[mask].sum()),
            proposed_first_adam_rms=rms, archived_first_adam_rms=old['arms'][arm]['history'][0]['actual_update_fp32']['rms'],
            relative_gradient_norm_difference=(float(np.linalg.norm(gradient.astype(float)))/report['gradient_norm_replay'][arm]['archived_huber']-1),
            regions=regions_summary)
        # Independent check of dL/dF per-pair values against saved report.
        dflow = np.clip(fields['flow']-target,-1,1)*valid[...,None]/(2*valid.sum())
        for i,pair in enumerate(report['readout']['per_pair_gradient_comparisons']):
            np.testing.assert_allclose(np.linalg.norm(dflow[i].astype(float)),pair['original_masks']['flow'][arm+'_norm'],rtol=1e-6)
    measured = compare(gradients['forward'],gradients['reverse'])
    for key,value in measured.items():
        np.testing.assert_allclose(value,report['latent_gradient_comparison'][key],rtol=1e-10,atol=1e-12)
    stages = {support:{stage:aggregate([row[support][stage] for row in report['readout']['per_pair_gradient_comparisons']])
                      for stage in ('flow','centered_scores','raw_scores')}
              for support in ('original_masks','common_mask_control')}
    result = dict(provenance=checks, counts=report['counts'], replay=report['replay'],
        analytic_flow_max_abs=report['readout']['captured_flow_max_abs'], arms=arms,
        latent=measured, score_stages=stages,
        first_adam_proposal=dict(**compare(proposals['forward'],proposals['reverse']),
            sign_disagreement_fraction=float(np.mean(np.sign(gradients['forward'])!=np.sign(gradients['reverse'])))),
        limitations='Adam proposals reconstructed offline from the new diagnostic gradients; not actual Huber update replay. '
            'No before latent was supplied locally, so no cast or model response was recomputed. '
            'Centered-score row gradient energy is NOT latent-gradient attribution. '
            'Raw-score comparisons are aggregated from reported pair statistics, not recaptured full score matrices.')
    first_step = {}
    for direction in ('forward','reverse'):
        with np.load(huber/direction/'flow_state_01.npz') as data:
            after = data['flow']
        first_step[direction] = {}
        for objective,(target,valid) in refs.items():
            def huber_score(flow):
                absolute = np.abs(flow.astype(float)[valid]-target[valid])
                return float(np.where(absolute <= 1,.5*absolute**2,absolute-.5).mean())
            predicted = float((gradients[objective].astype(float)*proposals[direction]).sum())
            actual = huber_score(after)-huber_score(fields['flow'])
            first_step[direction][objective] = dict(reconstructed_linear_loss_change=predicted,
                archived_actual_loss_change=actual,
                caveat='New diagnostic gradient/proposal versus archived Huber step; not an exact finite-difference pair.')
    result['first_step_loss_comparison'] = first_step
    out = run/'review'; out.mkdir(exist_ok=True)
    (out/'review_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot(result,fields,refs,out)
    print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))
    print(json.dumps({a:{k:v for k,v in r.items() if k!='regions'} for a,r in arms.items()},indent=2))
    for a,r in arms.items():
        print(a, 'score gradient energy by region:',{name:value['share_of_global_centered_score_gradient_energy'] for name,value in r['regions'].items()})
        print(a,'top torso score-gradient patch:',r['regions']['torso']['top_patches'][0])
    return result


def plot(result,fields,refs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    support = refs['forward'][1] & refs['reverse'][1] & regions()['torso']
    prob = fields['same_probability'][support]
    sens = fields['flow_jacobian_frobenius'][support]
    axes[0].scatter(prob,np.maximum(sens,1e-14),s=12,alpha=.6)
    axes[0].set(yscale='log',xlabel='Probability of same spatial location',ylabel='Expected-flow Jacobian norm (log scale)',
        title=f'All {int(support.sum())} common-valid torso positions')
    axes[0].axvline(.99,ls='--',color='gray',label='0.99 descriptive split')
    axes[0].legend()
    values = [result['score_stages']['original_masks'][s]['cosine'] for s in ('flow','centered_scores','raw_scores')]
    values += [result['latent']['cosine'],result['first_adam_proposal']['cosine']]
    axes[1].bar(range(5),values,color=['#348abd']*3+['#e24a33','#988ed5'])
    axes[1].set_xticks(range(5),['Flow loss\nderivative','Centered\nscores','Raw\nscores','Latent\ngradient','First Adam\nproposal'])
    axes[1].set(ylim=(-1,1),ylabel='Original / reversed cosine',title='Global objectives: distinction persists at the latent')
    axes[1].axhline(0,color='black',lw=.7)
    for i,value in enumerate(values):
        axes[1].text(i,value+(.04 if value>=0 else -.09),f'{value:.3f}',ha='center')
    fig.suptitle('Saved gradient review: no optimizer step, model rerun, or decoded motion measurement')
    fig.savefig(out/'gradient_diagnostic.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--pilot',default='probe_runs/wan_centered_pilot_camel_s1')
    p.add_argument('--huber',default='probe_runs/wan_centered_huber_camel_s1')
    args=p.parse_args()
    review(args.run,args.pilot,args.huber)
