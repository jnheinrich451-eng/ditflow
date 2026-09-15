"""Audit saved camel captures and diagnose soft correspondence, no model loads."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(run, output, qk=False):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run/'plan.json').read_text())
    meta = json.loads((run/'readouts/metadata.json').read_text())
    report = json.loads((run/'correspondence_report.json').read_text())
    source = json.loads((run/'source_revision.json').read_text())
    expected_events = [(control,state) for control in plan['controls'] for state in plan['states']]
    assert [(e['control'],e['state']) for e in meta['events']] == expected_events
    assert all(e['qk_shape']==[1,9360,40,128] for e in meta['events'])
    assert meta['conditioning_sha256']==plan['conditioning_sha256']
    assert Path(meta['checkpoint']).name==plan['checkpoint_revision']
    assert meta['sigmas']==plan['sigmas'] and meta['timesteps']==plan['timesteps']
    for name,expected in plan['input_sha256'].items():
        assert sha(run/name)==expected, name
    for name,expected in meta['source_sha256'].items():
        assert source['sha256'][name]==expected, name
    rows=[]
    for control,state in expected_events:
        for field in ('hard','soft'):
            for region in ('subject','background','subject_relative_to_background'):
                subset=[r for r in report['rows'] if r['control']==control and r['state']==state
                        and r['field']==field and r['region']==region and not r['includes_corrupt_endpoint']]
                row=dict(control=control,state=state,field=field,region=region,pairs=len(subset),
                         patches=sum(r['patches'] for r in subset))
                for key in ('amf_dx','image_dx','cosine','epe','zero_amf_fraction'):
                    values=[r[key] for r in subset if r[key] is not None]
                    row[key]=float(np.mean(values)) if values else None
                rows.append(row)
    summary=dict(audit_passed=True,source_commit=source['git']['commit'],checkpoint=plan['checkpoint_revision'],
                 conditioning_sha256=meta['conditioning_sha256'],sigma_step9=meta['sigmas'][9],
                 aggregation='Equal mean of per-pair statistics, excluding corrupted endpoints; patch units (16 RGB pixels).',
                 caveat='Image flow estimate within manual torso/fence ROIs, not physical ground truth.',rows=rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    names=['forward','reverse','static']; x=np.arange(3)
    def values(state,field,key):
        return [next(r[key] for r in rows if r['control']==c and r['state']==state and
                     r['field']==field and r['region']=='subject_relative_to_background') for c in names]
    for shift,label,data,color in [(-.25,'Decoded RGB estimate',values('clean','soft','image_dx'),'#333333'),
                                   (0,'AMF soft: clean',values('clean','soft','amf_dx'),'#297bb0'),
                                   (.25,'AMF soft: step 9',values('step_09','soft','amf_dx'),'#d35a32')]:
        ax.bar(x+shift,data,.23,label=label,color=color)
    ax.axhline(0,color='black',linewidth=.8)
    ax.set_xticks(x,['Original vanilla video\n(leftward relative to fence)',
                    'Reversed vanilla video\n(rightward relative to fence)','Repeated middle frame\n(static)'])
    ax.set_ylabel('Horizontal motion relative to fence (patches per frame pair)')
    ax.set_title('At step 9, soft AMF reports leftward relative motion for all controls')
    ax.legend(loc='lower left'); ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'motion_comparison.png',dpi=160); plt.close(fig)
    relative=[r for r in rows if r['region']=='subject_relative_to_background']
    print('control state field | image dx | AMF dx | endpoint error')
    for row in relative:
        print(f"{row['control']:7} {row['state']:7} {row['field']:4} | {row['image_dx']:+.3f} | {row['amf_dx']:+.3f} | {row['epe']:.3f}")
    if qk:
        inspect_qk(run,output,plan)
    return summary


def inspect_qk(run, output, plan):
    """One fixed central pair, both ROIs, all six captures; no head/temp sweep."""
    import torch
    torch.set_num_threads(8)
    f,h,w=plan['grid']; hw=h*w
    pair_i,pair_j=2,3  # Forward frames 8->12; reversed frames 12->8; neither corrupted.
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    coords=np.stack([xx.ravel(),yy.ravel()],axis=-1)
    with np.load(run/'rgb_motion.npz') as rgb:
        regions={name:rgb[name+'_region'].ravel() for name in ('subject','background')}
    selected=np.flatnonzero(regions['subject']|regions['background'])
    records=[]
    for control in plan['controls']:
        for state in plan['states']:
            print('Saved-Q/K readout:',control,state,flush=True)
            with np.load(run/'readouts'/f'{control}_{state}.npz') as capture:
                # Slice before creating Torch copies; at most one archive resident.
                q=capture['query'][0,pair_i*hw+selected].reshape(len(selected),-1)
                k=capture['key'][0,pair_j*hw:(pair_j+1)*hw].reshape(hw,-1)
                # Match production BF16 matmul and scalar rounding, then FP32 softmax.
                logits=(torch.from_numpy(q).bfloat16()@torch.from_numpy(k).bfloat16().T)*(1/(40*np.sqrt(128)))
                probs=(logits.float()*2.).softmax(-1).numpy()
                expected=probs@coords.astype(np.float32)
                reconstructed=expected-coords[selected]
                saved=capture['soft'][pair_i*f+pair_j,selected]
                error=float(np.max(np.abs(reconstructed-saved)))
                hard=capture['hard'][pair_i*f+pair_j,selected]
            entropy=-(probs*np.log(np.maximum(probs,1e-30))).sum(-1)/np.log(hw)
            peak=probs.max(-1)
            # Probability near the hard match: source location + hard displacement.
            match=coords[selected]+hard
            local=(np.abs(coords[None,:,0]-match[:,None,0])<=2)&(np.abs(coords[None,:,1]-match[:,None,1])<=2)
            mass=(probs*local).sum(-1)
            for name,mask in regions.items():
                chosen=mask[selected]
                records.append(dict(control=control,state=state,region=name,pair=[pair_i,pair_j],
                    rgb_anchors=[8,12] if control!='reverse' else [12,8],patches=int(chosen.sum()),
                    normalized_entropy_mean=float(entropy[chosen].mean()),peak_probability_mean=float(peak[chosen].mean()),
                    mass_within_2_patches_of_argmax_mean=float(mass[chosen].mean()),
                    source_x_mean=float(coords[selected][chosen,0].mean()),
                    expected_destination_x_mean=float(expected[chosen,0].mean()),
                    soft_dx_mean=float(reconstructed[chosen,0].mean()),
                    hard_dx_mean=float(hard[chosen,0].mean()),
                    reconstructed_soft_max_abs_error=error))
    (output/'qk_concentration.json').write_text(json.dumps(dict(
        method='CPU BF16-rounded mean-head logits; temperature 2; one fixed central frame pair; saved tensors only. CPU reconstruction differs from saved GPU soft flow by up to the reported error; this is not an exact backend parity test.',
        rows=records),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qk',action='store_true')
    args=parser.parse_args()
    review(args.run,args.output,args.qk)
