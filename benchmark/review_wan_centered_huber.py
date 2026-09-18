"""CPU review of saved Huber states; no model loading or new experiments."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.wan_centered_attribution import digest, read_references, regions
from benchmark.wan_centered_huber import scores, separation, torso_scores_by_pair


def review(run, pilot, attribution):
    run, pilot, attribution = map(Path, (run, pilot, attribution))
    report = json.loads((run/'huber_report.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    checks = {}
    for name, expected in protocol['inputs_sha256'].items():
        path = pilot/name
        checks['pilot/'+name] = ('match' if digest(path) == expected else 'MISMATCH') if path.is_file() else 'not supplied locally'
    checks['attribution.json'] = 'match' if digest(attribution/'attribution.json') == protocol['archived_attribution_sha256'] else 'MISMATCH'
    for name, expected in protocol['archived_fields_sha256'].items():
        checks[name+'_flow.npz'] = 'match' if digest(attribution/f'{name}_flow.npz') == expected else 'MISMATCH'
    if 'MISMATCH' in checks.values():
        raise ValueError(checks)
    # Cross-platform CRLF conversion may alter a Python file's byte hash.
    checks['runner_bytes'] = 'match' if digest(Path(__file__).with_name('wan_centered_huber.py')) == protocol['script_sha256'] else 'different local bytes; inspect source before replay'
    refs = read_references(pilot)
    common = refs['forward'][1] & refs['reverse'][1] & regions()['torso']
    exclude = common.copy()
    exclude[2, 12*52+33] = False
    old = {}
    for name in ('before', 'forward', 'reverse'):
        with np.load(attribution/f'{name}_flow.npz') as values:
            old[name] = values['flow'].copy()
    fields, result = {}, dict(provenance=checks, arms={}, states=[])
    for arm in ('forward', 'reverse'):
        arm_report = report['arms'][arm]
        assert len(arm_report['history']) == 5 and len(arm_report['readout_states']) == 6
        fields[arm] = []
        states = []
        for j, readout in enumerate(arm_report['readout_states']):
            assert readout['after_updates'] == j
            with np.load(run/arm/f'flow_state_{j:02d}.npz') as values:
                field = values['flow'].copy()
            assert field.shape == (5, 1560, 2) and np.isfinite(field).all()
            fields[arm].append(field)
            measured = scores(field, *refs[arm])
            np.testing.assert_allclose(list(measured.values()), list(readout['scores'].values()), atol=1e-10)
            if 'torso_scores_by_pair' in readout:
                assert readout['torso_scores_by_pair'] == torso_scores_by_pair(field, *refs[arm], refs['forward'][1] & refs['reverse'][1])
            rows = readout['probability']
            assert len(rows) == 21
            dominant = [p for p in rows if p['selected_because'] == 'previous dominant patch'][0]
            fixed = [p for p in rows if p['selected_because'] == 'fixed torso query']
            assert len(fixed) == 20
            logger_error = max(p['readout_check_max_abs'] for p in rows)
            assert logger_error < 1e-4
            probs = np.array([p['same_spatial_probability'] for p in fixed])
            entropy = np.array([p['entropy_nats'] for p in fixed])
            peak = np.array([p['max_probability'] for p in fixed])
            assert np.isfinite(np.r_[probs, entropy, peak]).all()
            query_valid = np.array([refs[arm][1][p['pair'][0], p['row']*52+p['column']] for p in fixed])
            query_common = np.array([common[p['pair'][0], p['row']*52+p['column']] for p in fixed])
            states.append(dict(after_updates=j, scores=measured,
                common_torso_scores=scores(field, refs[arm][0], common),
                common_torso_excluding_outlier_scores=scores(field, refs[arm][0], exclude),
                dominant=dominant, fixed_queries=dict(count=20, same_probability_median=float(np.median(probs)),
                    same_over_099_count=int((probs > .99).sum()), max_over_099_count=int((peak > .99).sum()),
                    own_valid_count=int(query_valid.sum()), own_valid_same_over_099_count=int(((probs > .99) & query_valid).sum()),
                    common_valid_count=int(query_common.sum()), common_valid_same_over_099_count=int(((probs > .99) & query_common).sum()),
                    entropy_median=float(np.median(entropy))), logger_max_abs=logger_error,
                per_pair=torso_scores_by_pair(field, *refs[arm], refs['forward'][1] & refs['reverse'][1])))
        with np.load(run/arm/'flow.npz') as values:
            np.testing.assert_array_equal(values['flow'], fields[arm][-1])
        np.testing.assert_array_equal(fields[arm][0], old['before'])
        for entry in arm_report['history']:
            assert np.isfinite(entry['gradient_norm']) and entry['gradient_norm'] > 0
            assert entry['actual_update_fp32']['rms'] > 0 and entry['actual_update_after_cast']['rms'] > 0
        target, valid = refs[arm]
        improvement = (((old['before'].astype(float)-target)**2 - (fields[arm][-1].astype(float)-target)**2).mean(-1))
        torso = valid & regions()['torso']
        share = float(improvement[torso].max()/improvement[torso].sum())
        np.testing.assert_allclose(share, arm_report['largest_torso_patch_share_of_net_mse_reduction'])
        result['arms'][arm] = dict(states=states, scores=arm_report['scores'], dominant_share_of_net_torso_mse_reduction=share,
            update_fp32=arm_report['update_fp32'], update_after_cast=arm_report['update_after_input_cast'],
            archived_mse_common_torso_scores=scores(old[arm],target,common),
            archived_mse_common_torso_excluding_outlier_scores=scores(old[arm],target,exclude))
    for j in range(6):
        measured = separation(fields['forward'][j], fields['reverse'][j], refs)
        assert measured == {k:v for k,v in report['separation_by_updates'][j].items() if k != 'after_updates'}
        delta = fields['forward'][j].astype(float)-fields['reverse'][j]
        desired = refs['forward'][0].astype(float)-refs['reverse'][0]
        projection = float((delta[exclude]*desired[exclude]).sum()/(desired[exclude]**2).sum())
        df, dr = [fields[a][j].astype(float)[common]-old['before'][common] for a in ('forward', 'reverse')]
        denom = np.linalg.norm(df)*np.linalg.norm(dr)
        result['states'].append(dict(after_updates=j, **measured, excluding_outlier_gain=projection,
            arm_update_cosine=float((df*dr).sum()/denom) if denom else None))
    result['archived_mse'] = separation(old['forward'], old['reverse'], refs)
    output = run/'review'
    output.mkdir(exist_ok=True)
    (output/'review_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    plot(result, output)
    print('PROVENANCE', checks)
    for arm, values in result['arms'].items():
        print(arm, 'scores', values['scores'], 'outlier net torso MSE share', values['dominant_share_of_net_torso_mse_reduction'])
        print('state / common torso MSE excluding outlier / dominant P(same) / fixed queries P(same) > .99')
        for s in values['states']:
            print(s['after_updates'], s['common_torso_excluding_outlier_scores']['mse'], s['dominant']['same_spatial_probability'], s['fixed_queries'])
        print('archived MSE excluding outlier', values['archived_mse_common_torso_excluding_outlier_scores'])
    print('state / gain / cosine / pair gains / gain excluding outlier / arm update cosine')
    for s in result['states']:
        print(s['after_updates'], s['all_common']['projection_gain'], s['all_common']['cosine'],
              [p['projection_gain'] for p in s['pairs']], s['excluding_outlier_gain'], s['arm_update_cosine'])
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(6)
    ax = axes[0, 0]
    ax.plot(x, [s['all_common']['projection_gain'] for s in result['states']], 'o-', label='Huber')
    ax.axhline(result['archived_mse']['all_common']['projection_gain'], color='gray', ls='--', label='Archived MSE final')
    ax.axhline(0, color='black', lw=.5)
    ax.set(title='Separation aligned with reference difference', ylabel='Projection gain (1 = requested separation)')
    ax.legend()
    ax = axes[0, 1]
    for arm, values in result['arms'].items():
        ax.plot(x, [s['common_torso_excluding_outlier_scores']['mse'] for s in values['states']], 'o-', label=arm)
    ax.set(title='Own-target torso error: shared mask, outlier excluded', ylabel='Mean squared error (patches squared)')
    ax.legend()
    ax = axes[1, 0]
    for arm, values in result['arms'].items():
        ax.plot(x, [s['dominant']['same_spatial_probability'] for s in values['states']], 'o-', label=arm)
    ax.set(title='Previously dominant patch: same-location probability', ylabel='Centered T8 probability', ylim=(0, 1.05))
    ax.legend()
    ax = axes[1, 1]
    for arm, values in result['arms'].items():
        ax.plot(x, [s['fixed_queries']['same_over_099_count'] for s in values['states']], 'o-', label=arm)
    ax.set(title='Fixed torso queries with same-location probability > 0.99', ylabel='Query count (20 sampled; not all torso patches)', ylim=(0, 20))
    ax.legend()
    for ax in axes.flat:
        ax.set_xlabel('Completed optimizer updates')
        ax.set_xticks(x)
        ax.grid(alpha=.2)
    fig.suptitle('Huber diagnostic: AMF readout only, no decoded motion measurement')
    fig.savefig(output/'huber_diagnostic.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--pilot', default='probe_runs/wan_centered_pilot_camel_s1')
    parser.add_argument('--attribution', default='probe_runs/wan_centered_attribution_camel_s1')
    args = parser.parse_args()
    review(args.run, args.pilot, args.attribution)
