"""Review archived destination-loss results on CPU; no model execution."""
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


def review(run, pilot, gradients, response):
    run, pilot, gradients, response = map(Path, (run, pilot, gradients, response))
    report = json.loads((run/'destination_report.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    control = json.loads((response/'response_report.json').read_text())
    provenance = {}
    groups = [('pilot', pilot, protocol['inputs_sha256']),
        ('gradient', gradients, protocol['control_sha256']['gradient_artifacts']),
        ('response', response, protocol['control_sha256']['response_artifacts']),
        ('source', Path(__file__).parent, {**protocol['helper_sources_sha256'],
                                         'wan_centered_destination.py':protocol['script_sha256']})]
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
    indices = {a:destination_indices(*v) for a,v in refs.items()}
    common = refs['forward'][1] & refs['reverse'][1]
    log_before = read_npz(run/'before_target_log_probability.npz')
    check_values(summarize_target_probability(log_before, refs, base), report['before_target_probability'])
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
        check_values(row['huber_control'], control['conditions'][a+'_full'])
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
            support = valid & mask
            values = improvement[support]
            regional[name] = dict(count=int(support.sum()), nll_decrease_sum=float(values.sum()),
                share_of_net_nll_decrease=float(values.sum()/total),
                mean_nll_change=float(-values.mean()))
        sharp = valid & roi['torso'] & (base['same_probability'] > .99)
        # Pairwise log odds against the SAME spatial destination. Partition function cancels.
        # Probability arrays can round to 1; log(1)=0 is valid for these sharp queries.
        margin_before = np.log(base['same_probability'][sharp].astype(float))-log_before[a][sharp]
        margin_after = np.log(after['same_probability'][sharp].astype(float))-log_after[a][sharp]
        own_before = report['before_target_probability'][a]['all_valid']['nll_mean']
        own_after = row['target_probability'][a]['all_valid']['nll_mean']
        predicted = {key:float((grads[a].astype(float)*deltas[a][key]).sum()) for key in deltas[a]}
        arms[a] = dict(nll_before=own_before, nll_after=own_after,
            relative_nll_decrease=(own_before-own_after)/own_before,
            old_errors=row['original_objectives'][a], regional_loss_accounting=regional,
            sharp=row['torso']['before_sharp'], common_without_outlier=row['torso']['common_without_outlier'],
            sharp_target_probability_before=report['before_target_probability'][a]['before_sharp_torso'],
            sharp_target_probability_after=row['target_probability'][a]['before_sharp_torso'],
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
        control_fields = {a:read_npz(response/(a+'_full')/'flow.npz')['flow'] for a in refs}
        check_values(separation(control_fields['forward'], control_fields['reverse'], masks), report['huber_control_'+label])
    check_values(compare(grads['forward'], grads['reverse']), report['latent_gradient_comparison'])
    for key, column in [('fp32','fp32_update_comparison'), ('after_cast','cast_update_comparison')]:
        check_values(compare(deltas['forward'][key], deltas['reverse'][key]), report[column])
    result = dict(provenance=provenance, counts=report['counts'], before_replay_exact=True,
        arms=arms, separation=report['separation'], without_outlier=report['separation_excluding_shared_patch'],
        huber_separation=report['huber_control_separation'],
        huber_without_outlier=report['huber_control_separation_excluding_shared_patch'],
        gradient_comparison=report['latent_gradient_comparison'],
        fp32_comparison=report['fp32_update_comparison'], cast_comparison=report['cast_update_comparison'],
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
    print('SEPARATION', json.dumps({k:result[k]['all_common'] for k in ('separation','without_outlier','huber_separation','huber_without_outlier')}))
    return result


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2, figsize=(11,8), constrained_layout=True)
    names = ['Original','Reversed']
    rows = list(result['arms'].values())
    x = np.arange(2)
    ax[0,0].bar(x-.17, [r['nll_before'] for r in rows], .34, label='Before')
    ax[0,0].bar(x+.17, [r['nll_after'] for r in rows], .34, label='After')
    ax[0,0].set(xticks=x, xticklabels=names, ylabel='Mean negative log probability', title='New loss decreases slightly')
    ax[0,0].legend()
    ax[0,1].bar(x, [r['sharp']['after_same_probability_above_099_count']/r['sharp']['count']*100 for r in rows])
    ax[0,1].set(xticks=x, xticklabels=names, ylim=(0,105), ylabel='Percent of initially sharp torso queries', title='Still P(same spatial destination) > 0.99')
    for i,r in enumerate(rows):
        ax[0,1].text(i, 80, f"{r['sharp']['after_same_probability_above_099_count']}/{r['sharp']['count']}", ha='center', color='white')
    labels = ['All shared torso','Exclude known outlier']
    for offset, keys, label in [(-.17, ('huber_separation','huber_without_outlier'), 'Huber control'),
                                (.17, ('separation','without_outlier'), 'Destination NLL')]:
        ax[1,0].bar(x+offset, [result[k]['all_common']['projection_gain'] for k in keys], .34, label=label)
    ax[1,0].axhline(0, color='black', linewidth=.6)
    ax[1,0].set(xticks=x, xticklabels=labels, ylabel='Projection onto requested AMF difference', title='Reference separation remains weak')
    ax[1,0].legend()
    for offset, key, label in [(-.17,'before','Before'),(.17,'after','After')]:
        ax[1,1].bar(x+offset, [r['common_without_outlier'][key]['huber_delta1'] for r in rows], .34, label=label)
    ax[1,1].set(xticks=x, xticklabels=names, ylabel='Mean coordinate Huber error', title='Own-target torso error, shared support minus outlier')
    ax[1,1].legend()
    fig.suptitle('One destination-loss update: readout diagnostic, not decoded motion', fontsize=13)
    fig.savefig(out/'destination_diagnostic.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='probe_runs/wan_centered_destination_camel_s1')
    parser.add_argument('--pilot', default='probe_runs/wan_centered_pilot_camel_s1')
    parser.add_argument('--gradients', default='probe_runs/wan_centered_gradients_camel_s1')
    parser.add_argument('--response', default='probe_runs/wan_centered_response_camel_s1')
    args = parser.parse_args()
    review(args.run, args.pilot, args.gradients, args.response)
