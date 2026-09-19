"""Review the fixed common-torso support experiment on CPU; no model execution."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest, read_references, regions
from benchmark.wan_centered_destination import destination_indices, summarize_target_probability
from benchmark.wan_centered_gradients import compare
from benchmark.wan_centered_huber import scores, separation, torso_scores_by_pair
from benchmark.wan_centered_response import torso_response
from benchmark.wan_centered_torso import common_torso_refs, support_nll


def gradient_alignment_bounds(max_probability, target_logp, target_indices, support, rounding_slack=1e-6):
    """Bound cosine at centered/raw score gradients using saved probability summaries.

    Equal support gives dL/dC proportional to (p-onehot(target)) per selected row.
    If T=sum ||p||^2, A=sum p(target_f), B=sum p(target_r), n=selected rows and
    e=equal destinations, squared norms are T-2A+n, T-2B+n; dot is T-A-B+e.
    max(p)^2 <= ||p||^2 <= max(p), so full distributions need not be archived.

    Column centering backprop subtracts the mean over ALL spatial source rows.
    A gradient supported on m rows loses at most m/N of its squared norm under
    this projection (Cauchy-Schwarz). This bounds how far centering can raise cosine.
    Bounds concern the intended smooth score derivatives, not discrete BF16 rounding.
    """
    if support.ndim != 2 or support.dtype != np.bool_ or not support.any():
        raise ValueError('Expected nonempty shared boolean support')
    pmax = np.asarray(max_probability, float)[support]
    pf, pr = [np.exp(np.asarray(target_logp[a], float)[support]) for a in ('forward','reverse')]
    n = int(support.sum())
    same = int(((target_indices['forward'] == target_indices['reverse']) & support).sum())
    low_t = float((np.clip(pmax-rounding_slack, 0, 1)**2).sum())
    high_t = float(np.clip(pmax+rounding_slack, 0, 1).sum())
    af, bf = [float(np.clip(pf+s*rounding_slack, 0, 1).sum()) for s in (-1,1)]
    ar, br = [float(np.clip(pr+s*rounding_slack, 0, 1).sum()) for s in (-1,1)]
    aa = (low_t-2*bf+n, high_t-2*af+n)
    bb = (low_t-2*br+n, high_t-2*ar+n)
    ab = (low_t-bf-br+same, high_t-af-ar+same)
    if aa[0] <= 0 or bb[0] <= 0:
        cosine = [-1., 1.]
    else:
        denominator = (np.sqrt(aa[0]*bb[0]), np.sqrt(aa[1]*bb[1]))
        corners = [v/d for v in ab for d in denominator]
        cosine = [max(-1., min(corners)), min(1., max(corners))]
    rho = float(support.sum(1).max()/support.shape[1])
    numerator = cosine[1]+rho
    upper = 1. if rho == 1 else min(1., numerator/(1-rho) if numerator >= 0 else numerator)
    return dict(count=n, same_destination_count=same,
        probability_rounding_slack=rounding_slack, probability_squared_norm_sum_bounds=[low_t,high_t],
        centered_score_gradient_cosine_bounds=cosine,
        max_centering_removed_energy_fraction=rho, raw_score_gradient_cosine_upper_bound=upper,
        meaning='Analytical bounds from identical support and saved probabilities. '
        'Near-alignment beyond this bound must arise later on the score-to-latent path, '
        'including Q/K geometry, transformer derivatives and mixed-precision backward operations.')


def review(run, pilot, gradients, response, destination):
    run, pilot, gradients, response, destination = map(Path, (run, pilot, gradients, response, destination))
    report = json.loads((run/'torso_report.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    control = json.loads((destination/'destination_report.json').read_text())
    provenance = {}
    groups = [('pilot', pilot, protocol['inputs_sha256']),
        ('gradient', gradients, protocol['control_sha256']['gradient_artifacts']),
        ('response', response, protocol['control_sha256']['response_artifacts']),
        ('destination', destination, protocol['control_sha256']['destination_artifacts']),
        ('source', Path(__file__).parent, {**protocol['helper_sources_sha256'],
                                         'wan_centered_torso.py':protocol['script_sha256']})]
    for prefix, root, entries in groups:
        for name, expected in entries.items():
            path = root/name
            provenance[prefix+'/'+name] = ('match' if digest(path) == expected else 'MISMATCH') if path.is_file() else 'not supplied locally'
    if 'MISMATCH' in provenance.values():
        raise ValueError(provenance)
    assert report['counts'] == dict(positive_forwards_attempted=3, positive_forwards_completed=3,
        latent_backwards_attempted=2, latent_backwards_completed=2, first_adam_proposals=2)
    check_values(report['counts'], json.loads((run/'progress.json').read_text()))
    assert report['baseline_unchanged'] and not report['port_success']
    before = read_npz(run/'before_flow.npz')['flow']
    base = read_npz(gradients/'readout_derivatives.npz')
    np.testing.assert_array_equal(before, base['flow'])
    refs = read_references(pilot)
    optimization_refs, support = common_torso_refs(refs)
    saved_support = read_npz(run/'optimization_support.npz')
    np.testing.assert_array_equal(support, saved_support['common_torso'])
    for a in refs:
        np.testing.assert_array_equal(saved_support[a+'_original_valid'], refs[a][1])
    indices = {a:destination_indices(*v) for a,v in refs.items()}
    common = refs['forward'][1] & refs['reverse'][1]
    log_before = read_npz(run/'before_target_log_probability.npz')
    check_values(summarize_target_probability(log_before, refs, base), report['before_target_probability'])
    check_values(support_nll(log_before, support), report['before_optimization_nll'])
    for a,v in read_npz(destination/'before_target_log_probability.npz').items():
        np.testing.assert_array_equal(log_before[a], v)
    roi = regions()
    fields, deltas, grads, arms = {}, {}, {}, {}
    for a, row in report['arms'].items():
        fields[a] = read_npz(run/a/'flow.npz')['flow']
        after = read_npz(run/a/'readout_derivatives.npz')
        log_after = read_npz(run/a/'target_log_probability.npz')
        deltas[a] = read_npz(run/a/'actual_delta.npz')
        grads[a] = read_npz(run/f'{a}_latent_gradient.npz')['gradient']
        arrays = [fields[a], *after.values(), *log_after.values(), *deltas[a].values(), grads[a]]
        assert all(np.isfinite(v).all() for v in arrays)
        assert all((v <= 1e-6).all() for v in log_after.values())
        np.testing.assert_array_equal(fields[a], after['flow'])
        for target in refs:
            np.testing.assert_array_equal(after[target+'_valid'], refs[target][1])
            check_values(dict(before=scores(before, *refs[target]), after=scores(fields[a], *refs[target])),
                         row['original_objectives'][target])
        check_values(summarize_target_probability(log_after, refs, base), row['target_probability'])
        check_values(torso_response(before, fields[a], *refs[a], common, base, after, a), row['torso'])
        check_values(torso_scores_by_pair(fields[a], *refs[a], common), row['torso_scores_by_pair'])
        check_values(row['full_field_control'], {key:control['arms'][a][key] for key in row['full_field_control']})
        check_values(support_nll(log_after, support), row['optimization_nll'])
        check_values(float(np.linalg.norm(grads[a].astype(float))), row['gradient_norm'])
        for key, column in [('fp32','delta_fp32'), ('after_cast','delta_after_cast')]:
            values = deltas[a][key].astype(float)
            check_values(dict(rms=float(np.sqrt((values*values).mean())), max_abs=float(np.abs(values).max()),
                              changed_fraction=float((values != 0).mean())), row[column])
        check_values(compare(deltas[a]['fp32'], deltas[a]['after_cast'])['cosine'], row['cast_delta_vs_fp32_cosine'])
        valid = refs[a][1]
        improvement = log_after[a].astype(float)-log_before[a]
        total = float(improvement[valid].sum())
        regional = {}
        for name, mask in roi.items():
            regional_support = valid & mask
            values = improvement[regional_support]
            regional[name] = dict(count=int(regional_support.sum()), nll_decrease_sum=float(values.sum()),
                share_of_net_nll_decrease=float(values.sum()/total),
                mean_nll_change=float(-values.mean()))
        sharp = support & (base['same_probability'] > .99)
        optimized_sharp = torso_response(before, fields[a], refs[a][0], support, common, base, after, a)['before_sharp']
        control_after = read_npz(destination/a/'readout_derivatives.npz')
        control_sharp = torso_response(before, control_after['flow'], refs[a][0], support, common, base, control_after, a)['before_sharp']
        # Pairwise log odds against the SAME spatial destination. Partition function cancels.
        # Probability arrays can round to 1; log(1)=0 is valid for these sharp queries.
        margin_before = np.log(base['same_probability'][sharp].astype(float))-log_before[a][sharp]
        margin_after = np.log(after['same_probability'][sharp].astype(float))-log_after[a][sharp]
        own_before = report['before_optimization_nll'][a]['mean']
        own_after = row['optimization_nll'][a]['mean']
        predicted = {key:float((grads[a].astype(float)*deltas[a][key]).sum()) for key in deltas[a]}
        arms[a] = dict(nll_before=own_before, nll_after=own_after,
            relative_nll_decrease=(own_before-own_after)/own_before,
            old_errors=row['original_objectives'][a], full_field_regional_loss_accounting=regional,
            optimized_nll_by_pair_before=report['before_optimization_nll'][a]['per_pair'],
            optimized_nll_by_pair_after=row['optimization_nll'][a]['per_pair'],
            common_errors=row['torso']['common_valid'],
            optimized_sharp=optimized_sharp, control_optimized_sharp=control_sharp,
            sharp=row['torso']['before_sharp'], common_without_outlier=row['torso']['common_without_outlier'],
            sharp_target_probability_before=summarize_target_probability(log_before, optimization_refs, base)[a]['before_sharp_torso'],
            sharp_target_probability_after=summarize_target_probability(log_after, optimization_refs, base)[a]['before_sharp_torso'],
            sharp_same_vs_target_log_odds_before_median=float(np.median(margin_before)),
            sharp_same_vs_target_log_odds_after_median=float(np.median(margin_after)),
            predicted_nll_change=predicted, observed_nll_change=own_after-own_before,
            delta_fp32=row['delta_fp32'], delta_after_cast=row['delta_after_cast'],
            gradient_norm=row['gradient_norm'], outlier_before=before[2,12*52+33].tolist(),
            outlier_after=fields[a][2,12*52+33].tolist(),
            torso_per_pair=row['torso_scores_by_pair'])
    excluded = {a:(f,m.copy()) for a,(f,m) in refs.items()}
    for _,m in excluded.values():
        m[2,12*52+33] = False
    for label, masks in [('separation', refs), ('separation_excluding_shared_patch', excluded)]:
        check_values(separation(fields['forward'], fields['reverse'], masks), report[label])
        control_fields = {a:read_npz(destination/a/'flow.npz')['flow'] for a in refs}
        check_values(separation(control_fields['forward'], control_fields['reverse'], masks), report['full_field_control_'+label])
    check_values(compare(grads['forward'], grads['reverse']), report['latent_gradient_comparison'])
    for key, column in [('fp32','fp32_update_comparison'), ('after_cast','cast_update_comparison')]:
        check_values(compare(deltas['forward'][key], deltas['reverse'][key]), report[column])
    result = dict(provenance=provenance, counts=report['counts'], before_replay_exact=True,
        arms=arms, separation=report['separation'], without_outlier=report['separation_excluding_shared_patch'],
        control_separation=report['full_field_control_separation'],
        control_without_outlier=report['full_field_control_separation_excluding_shared_patch'],
        gradient_comparison=report['latent_gradient_comparison'],
        fp32_comparison=report['fp32_update_comparison'], cast_comparison=report['cast_update_comparison'],
        optimization_support=dict(count=int(support.sum()), per_pair=support.sum(1).tolist(),
            initially_sharp=int((support & (base['same_probability'] > .99)).sum())),
        control_gradient_comparison=control['latent_gradient_comparison'],
        score_alignment_bounds=gradient_alignment_bounds(base['max_probability'], log_before, indices, support),
        target_common_torso_different_count=int((common & roi['torso'] & (indices['forward'] != indices['reverse'])).sum()),
        limitations='Two original trace NPZs are not supplied locally, so before-latent immutability and cast deltas '
        'cannot be independently reconstructed here. Regional loss accounting is not latent-gradient attribution. '
        'Log odds and flow describe the readout, not physical trajectories. No model or decoded video was run.')
    out = run/'review'
    out.mkdir(exist_ok=True)
    (out/'review_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    plot(result, out)
    print('PROVENANCE', json.dumps(provenance))
    for a,row in arms.items():
        print(a, json.dumps({k:v for k,v in row.items() if k not in ('torso_per_pair','sharp','common_without_outlier')}))
    print('SEPARATION', json.dumps({k:result[k]['all_common'] for k in ('separation','without_outlier','control_separation','control_without_outlier')}))
    return result


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2, figsize=(11,8), constrained_layout=True)
    rows = list(result['arms'].values())
    x = np.arange(2); names = ['Original','Reversed']
    for offset,key,label in [(-.17,'nll_before','Before'),(.17,'nll_after','After')]:
        ax[0,0].bar(x+offset,[r[key] for r in rows],.34,label=label)
    ax[0,0].set(xticks=x,xticklabels=names,ylabel='Mean destination NLL',title='Restricted objective improves')
    ax[0,0].legend()
    for offset,key,label in [(-.17,'before','Before'),(.17,'after','After')]:
        ax[0,1].bar(x+offset,[r['common_errors'][key]['huber_delta1'] for r in rows],.34,label=label)
    ax[0,1].set(xticks=x,xticklabels=names,ylabel='Coordinate Huber error',title='Own-target error on optimized torso queries')
    ax[0,1].legend()
    for offset,keys,label in [(-.17,('control_separation','control_without_outlier'),'Full-field NLL'),
                              (.17,('separation','without_outlier'),'Common-torso NLL')]:
        ax[1,0].bar(x+offset,[result[k]['all_common']['projection_gain'] for k in keys],.34,label=label)
    ax[1,0].set(xticks=x,xticklabels=['All shared torso','Exclude known outlier'],
                ylabel='Projection onto requested AMF difference',title='Reference separation remains unaligned or weak')
    ax[1,0].axhline(0,color='black',linewidth=.6);ax[1,0].legend()
    for offset,key,label in [(-.17,'control_optimized_sharp','Full-field NLL'),(.17,'optimized_sharp','Common-torso NLL')]:
        values=[r[key]['after_same_probability_above_099_count'] for r in rows]
        bars=ax[1,1].bar(x+offset,values,.34,label=label)
        ax[1,1].bar_label(bars,padding=3)
    ax[1,1].set(xticks=x,xticklabels=names,ylim=(0,200),
                ylabel='Count out of 182 initially sharp optimized queries',
                title='Sharp same-coordinate matches persist')
    ax[1,1].legend(loc='lower right')
    fig.suptitle('Loss-support mechanism test: no decoded-motion claim',fontsize=13)
    fig.savefig(out/'torso_diagnostic.png',dpi=170)
    plt.close(fig)
    bounds=result['score_alignment_bounds']
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    values=[bounds['centered_score_gradient_cosine_bounds'][1],
            bounds['raw_score_gradient_cosine_upper_bound'],result['gradient_comparison']['cosine']]
    bars=ax.bar(np.arange(3),values,color=['#82aecf','#82aecf','#e78a2f'])
    bars[0].set_hatch('//');bars[1].set_hatch('//')
    for i,v in enumerate(values):
        ax.text(i,v+.025,('upper bound '+f'{v:.3f}') if i<2 else ('measured '+f'{v:.3f}'),ha='center')
    ax.set(xticks=np.arange(3),xticklabels=['Centered-score gradients','Raw-score gradients\n(after centering backward)',
            'Noisy-latent gradients'],ylim=(0,1.1),ylabel='Original/reversed gradient cosine similarity',
            title='Near-alignment arises later in the score-to-latent path')
    fig.savefig(out/'gradient_alignment.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='probe_runs/wan_centered_torso_camel_s1')
    parser.add_argument('--pilot',default='probe_runs/wan_centered_pilot_camel_s1')
    parser.add_argument('--gradients',default='probe_runs/wan_centered_gradients_camel_s1')
    parser.add_argument('--response',default='probe_runs/wan_centered_response_camel_s1')
    parser.add_argument('--destination',default='probe_runs/wan_centered_destination_camel_s1')
    args=parser.parse_args()
    review(args.run,args.pilot,args.gradients,args.response,args.destination)
