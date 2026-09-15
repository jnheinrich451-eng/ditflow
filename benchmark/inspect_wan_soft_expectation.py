"""One offline sharpening candidate on saved step-39 Q/K; no model inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from guidance_utils.wan_reference_diagnostics import comparison


def inspect(run,output):
    run,output=Path(run),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((run/'plan.json').read_text())
    assert plan['states']==['step_39'] and plan['grid']==[6,30,52] and plan['block']==20
    print('HYPOTHESIS: probability on distant matches biases the soft expectation. '
          'EXPECTED: temperature 8 improves reversed/static motion over temperature 2. '
          'LIMIT: 3 saved archives, 5 adjacent pairs each, temperatures 2 and 8 only; zero model calls.',flush=True)
    torch.set_num_threads(8)
    f,h,w=plan['grid']; hw=h*w
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    coordinates=torch.tensor(np.stack([xx.ravel(),yy.ravel()],axis=-1),dtype=torch.float32)
    rows=[]; replay_errors=[]
    with np.load(run/'rgb_motion.npz') as rgb:
        subject=rgb['subject_region'].ravel(); background=rgb['background_region'].ravel()
        indices=np.flatnonzero(subject|background)
        for control in plan['controls']:
            print('Reading cached Q/K:',control,flush=True)
            with np.load(run/'readouts'/f'{control}_step_39.npz') as data:
                q=data['query'][0].reshape(f,hw,-1)
                k=data['key'][0].reshape(f,hw,-1)
                saved=data['soft']
                for pair in range(f-1):
                    logits=(torch.from_numpy(q[pair,indices]).bfloat16() @
                            torch.from_numpy(k[pair+1]).bfloat16().T)*(1/(40*np.sqrt(128)))
                    for temp in (2.,8.):
                        probabilities=(logits.float()*temp).softmax(-1)
                        flow=(probabilities@coordinates-coordinates[indices]).numpy()
                        if temp==2.:
                            replay_errors.append(float(np.max(np.abs(flow-saved[pair*f+pair+1,indices]))))
                        truth=rgb[control+'_flow'][pair].reshape(hw,2)[indices]
                        valid=rgb[control+'_valid'][pair].ravel()[indices]
                        bg=background[indices]&valid
                        for region in ('subject','background','subject_relative_to_background'):
                            selected=(subject if region.startswith('subject') else background)[indices]&valid
                            a,b=flow,truth
                            if region=='subject_relative_to_background':
                                if not bg.any(): selected=np.zeros_like(selected)
                                else: a,b=a-np.median(a[bg],axis=0),b-np.median(b[bg],axis=0)
                            values=comparison(a,b,selected)
                            values['epe']=float(np.linalg.norm(a[selected]-b[selected],axis=-1).mean()) if selected.any() else None
                            rows.append(dict(control=control,temperature=temp,region=region,pair=pair,
                                corrupt_endpoint=(control=='forward' and pair==0) or (control=='reverse' and pair==f-2),**values))
                del q,k,saved
    summaries=[]
    for control in plan['controls']:
        for temp in (2.,8.):
            for region in ('subject','background','subject_relative_to_background'):
                subset=[r for r in rows if r['control']==control and r['temperature']==temp and r['region']==region and not r['corrupt_endpoint']]
                entry=dict(control=control,temperature=temp,region=region)
                for key in ('amf_dx','image_dx','cosine','epe'):
                    numbers=[r[key] for r in subset if r[key] is not None]
                    entry[key]=float(np.mean(numbers)) if numbers else None
                summaries.append(entry)
    result=dict(port_success=False,model_calls=0,temperatures=[2,8],
        method='CPU BF16-rounded logits from saved post-RoPE Q/K, all heads, every adjacent pair, same RGB-valid torso/fence support.',
        limitation='Offline readout screen only; CPU rounding and gradient behavior require validation before generating with this change.',
        baseline_replay_max_abs_error=max(replay_errors),summary=summaries,rows=rows)
    (output/'sharpening.json').write_text(json.dumps(result,indent=2))
    for row in summaries:
        if row['region']=='subject_relative_to_background': print(row)
    print('Maximum CPU baseline replay difference (patches):',result['baseline_replay_max_abs_error'])
    return result


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('run',type=Path); args.add_argument('--output',type=Path,required=True)
    opts=args.parse_args(); inspect(opts.run,opts.output)
