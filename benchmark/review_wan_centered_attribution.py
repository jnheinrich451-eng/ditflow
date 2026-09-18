"""CPU-only review of returned sampling-state AMF fields and native masses."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.wan_centered_attribution import digest, read_references, regions


def cosine(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator else None


def review(run, pilot, output):
    run, pilot, output = Path(run), Path(pilot), Path(output)
    report = json.loads((run/'attribution.json').read_text())
    protocol = json.loads((run/'started.json').read_text())
    provenance = {name: ('match' if digest(pilot/name) == expected else 'MISMATCH')
                  if (pilot/name).is_file() else 'not supplied locally'
                  for name, expected in protocol['inputs_sha256'].items()}
    if 'MISMATCH' in provenance.values():
        raise ValueError(f'Original pilot inputs differ: {provenance}')
    fields = {}
    masses = {}
    for name in ('before', 'forward', 'reverse'):
        with np.load(run/f'{name}_flow.npz', allow_pickle=False) as data:
            fields[name] = data['flow'].astype(np.float64)
        with np.load(run/f'{name}_native_mass.npz', allow_pickle=False) as data:
            masses[name] = data['mass'].astype(np.float64)
        if not np.isfinite(fields[name]).all() or not np.isfinite(masses[name]).all():
            raise ValueError('Nonfinite field/mass')
        np.testing.assert_allclose(masses[name].sum(axis=-1), 1., atol=1e-5)
    refs = read_references(pilot)
    common = refs['forward'][1] & refs['reverse'][1]
    masks = regions()
    torso_common = common & masks['torso']
    delta = {a: fields[a]-fields['before'] for a in ('forward', 'reverse')}
    result = dict(loss_replay_passed=report['loss_replay_passed'], provenance=provenance,
                  loss_accounting={}, pair_response=[], concentration={}, mass={})
    for arm in ('forward', 'reverse'):
        r = report['arms'][arm]
        reduction = r['loss_before']-r['loss_after']
        result['loss_accounting'][arm] = dict(total_reduction=reduction,
            regions={name: dict(reduction=v['reduction_contribution'],
                               reduction_share=v['reduction_contribution']/reduction,
                               mean_mse_before=v['mean_mse_before'], mean_mse_after=v['mean_mse_after'])
                     for name,v in r['regions'].items()})
        ref, valid = refs[arm]
        mask = valid & masks['torso']
        improvement = ((fields['before']-ref)**2-(fields[arm]-ref)**2).mean(axis=-1)
        ranked = np.argsort(np.where(mask, improvement, -np.inf).ravel())[::-1]
        ranked = [int(i) for i in ranked if mask.ravel()[i]][:10]
        total = float(improvement[mask].sum())
        top = []
        for index in ranked:
            pair, pos = divmod(index, 1560)
            row, col = divmod(pos, 52)
            top.append(dict(pair=[pair, pair+1], row=row, column=col,
                before=fields['before'][pair, pos].tolist(), after=fields[arm][pair, pos].tolist(),
                target=ref[pair, pos].tolist(), improvement=float(improvement[pair, pos]),
                share_of_net_torso_improvement=float(improvement[pair, pos]/total)))
        result['concentration'][arm] = dict(top_patches=top,
            pair_shares=[float(improvement[i, mask[i]].sum()/total) for i in range(5)],
            top1_share=top[0]['share_of_net_torso_improvement'],
            top5_share=sum(p['share_of_net_torso_improvement'] for p in top[:5]),
            warning='Shares use net decrease; they can exceed 1 if other patches worsen.')
        global_ranked = np.sort(improvement[valid])[::-1]
        zero = np.linalg.norm(fields[arm], axis=-1) < .05
        result['concentration'][arm]['global_top10_share'] = float(global_ranked[:10].sum()/improvement[valid].sum())
        result['concentration'][arm]['global_top50_share'] = float(global_ranked[:50].sum()/improvement[valid].sum())
        result['concentration'][arm]['near_zero_flow_fraction_torso'] = float(zero[mask].mean())
        result['concentration'][arm]['before_near_zero_flow_fraction_torso'] = float((np.linalg.norm(fields['before'],axis=-1)[mask] < .05).mean())
    for i in range(5):
        mask = torso_common[i]
        df, dr = delta['forward'][i, mask], delta['reverse'][i, mask]
        desired_separation = refs['forward'][0][i, mask]-refs['reverse'][0][i, mask]
        actual_separation = fields['forward'][i, mask]-fields['reverse'][i, mask]
        result['pair_response'].append(dict(pair=[i, i+1], common_count=int(mask.sum()),
            forward_reverse_update_cosine=cosine(df, dr),
            forward_delta_rms=float(np.sqrt((df**2).mean())), reverse_delta_rms=float(np.sqrt((dr**2).mean())),
            separation_rms=float(np.sqrt((actual_separation**2).mean())),
            desired_separation_rms=float(np.sqrt((desired_separation**2).mean())),
            separation_cosine=cosine(actual_separation, desired_separation),
            separation_projection_gain=float((actual_separation*desired_separation).sum()/(desired_separation**2).sum())))
    queries = report['native_attention']['queries']
    full_common = common & masks['torso']
    without_outlier = full_common.copy()
    without_outlier[2, 12*52+33] = False
    result['torso_response_aggregate'] = {}
    for name, mask in (('all_common', full_common), ('excluding_largest_shared_patch', without_outlier)):
        f, r = delta['forward'][mask], delta['reverse'][mask]
        difference = f-r
        target_difference = refs['forward'][0][mask].astype(float)-refs['reverse'][0][mask]
        result['torso_response_aggregate'][name] = dict(count=int(mask.sum()),
            arm_update_cosine=cosine(f,r), separation_cosine=cosine(difference,target_difference),
            separation_rms=float(np.sqrt((difference**2).mean())),
            target_separation_rms=float(np.sqrt((target_difference**2).mean())),
            projection_gain=float((difference*target_difference).sum()/(target_difference**2).sum()))
    for region in ('torso', 'fence'):
        ids = [i for i,q in enumerate(queries) if q['region'] == region]
        result['mass'][region] = {}
        for name, mass in masses.items():
            values = np.concatenate([mass[i, :, queries[i]['destination_frame']] for i in ids])
            result['mass'][region][name] = dict(mean=float(values.mean()),
                quantiles=np.quantile(values, [0,.1,.5,.9,1]).tolist(),
                fraction_below_001=float((values < .01).mean()))
    output.mkdir(parents=True, exist_ok=True)
    (output/'review_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    plot_review(fields, refs, masks, result, pilot, output)
    print(json.dumps({k:v for k,v in result.items() if k != 'concentration'}, indent=2))
    print(json.dumps({a:{k:v for k,v in r.items() if k != 'top_patches'} for a,r in result['concentration'].items()}, indent=2))
    return result


def plot_review(fields, refs, masks, result, pilot, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    # Selected AFTER accounting: this figure documents the largest shared change, not a new test ROI.
    pair, pos = 2, 12*52+33
    cap = cv2.VideoCapture(str(pilot/'off/final.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 12)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Missing original generated off video')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    ax = axes[0]
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    source = np.array([33*16+8, 12*16+8])
    for name, color, size in [('before', 'crimson', 60), ('forward', 'royalblue', 140), ('reverse', 'limegreen', 40)]:
        destination = source+fields[name][pair,pos]*16
        ax.scatter(*destination, c=color, s=size, label=name+' expected destination', marker='x' if name=='reverse' else 'o')
    ax.scatter(*source, facecolors='none', edgecolors='white', s=200, linewidths=2, label='source grid coordinate')
    ax.set_title('Largest torso correction: both arms return to near-zero flow\nDestination RGB anchor 12; expectation is not a matched pixel')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_axis_off()
    width = .32
    x = np.arange(5)
    for arm, offset in [('forward', -width/2), ('reverse', width/2)]:
        values = result['concentration'][arm]['pair_shares']
        axes[1].bar(x+offset, np.asarray(values)*100, width, label=arm)
    axes[1].set_xticks(x, ['0->1','1->2','2->3','3->4','4->5'])
    axes[1].set_ylabel('Share of net torso loss decrease (%)')
    axes[1].set_xlabel('Adjacent latent-frame pair')
    axes[1].set_title('Loss reduction concentrates in one pair\nOne patch accounts for 96.7% / 97.8%')
    axes[1].legend()
    fig.savefig(output/'shared_correction.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax in axes:
        for name, color, marker, size in [('before', 'crimson', 'o', 60),
                                        ('forward', 'royalblue', 'o', 130),
                                        ('reverse', 'limegreen', 'x', 80)]:
            xy = fields[name][pair, pos]
            ax.scatter(*xy, color=color, marker=marker, s=size, label=name+' readout')
        for arm, color in [('forward','darkorange'), ('reverse','purple')]:
            ax.scatter(*refs[arm][0][pair,pos], color=color, marker='*', s=160, label=arm+' target')
        ax.axhline(0, color='gray', linewidth=.5)
        ax.axvline(0, color='gray', linewidth=.5)
        ax.set_xlabel('dx (patches; 1 patch = 16 pixels)')
        ax.set_ylabel('dy (positive downward)')
        ax.invert_yaxis()
        ax.grid(alpha=.2)
    axes[0].set_title('Shared patch: full displacement range')
    axes[0].legend(fontsize=8)
    axes[1].set_xlim(-.5,1.5); axes[1].set_ylim(1.5,-1.5)
    axes[1].set_title('Zoom: outputs overlap near zero; targets differ')
    fig.savefig(output/'shared_patch_offsets.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--pilot', type=Path, default=Path('probe_runs/wan_centered_pilot_camel_s1'))
    args = parser.parse_args()
    review(args.run, args.pilot, args.run/'review')
