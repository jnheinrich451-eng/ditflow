"""One no-grad Wan capture, two readout-only backwards to locate gradient alignment."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmark.wan_centered_attribution import digest
from benchmark.wan_centered_destination import ARMS
from benchmark.wan_centered_gradients import capture, compare, write_json
from benchmark.wan_centered_huber import frozen_model
from benchmark.wan_centered_torso import prepare_torso_inputs, common_torso_refs, support_nll


def readout_with_taps(q, k, refs, indices, height=30, width=52):
    """Frozen destination NLL algebra, exposing score tensors as backward endpoints.

    Q/K must be detached leaves in native dtype. Autograd stops there: no Wan backward.
    All source rows participate in centering; mask only the final loss reduction.
    """
    if not (q.is_leaf and k.is_leaf and q.requires_grad and k.requires_grad):
        raise ValueError('Q/K must be detached differentiable leaves')
    if q.shape != k.shape or q.ndim != 4 or q.shape[1] != height*width:
        raise ValueError('Expected frame/spatial/head/dimension Q/K')
    raw, centered = [], []
    parts, logs = {a:[] for a in ARMS}, {a:[] for a in ARMS}
    masks = {a:torch.as_tensor(refs[a][1], device=q.device) for a in ARMS}
    ids = {a:torch.as_tensor(indices[a], device=q.device) for a in ARMS}
    if not torch.equal(masks['forward'], masks['reverse']) or not masks['forward'].any():
        raise ValueError('Requires identical nonempty support')
    heads, dim = q.shape[-2:]
    with torch.autocast(device_type=q.device.type, enabled=False):
        for i in range(q.shape[0]-1):
            logits = (q[i].flatten(1)@k[i+1].flatten(1).T)*(1/(heads*math.sqrt(dim)))
            logits = logits.float() if q.dtype != torch.float64 else logits
            scores = logits-logits.mean(0, keepdim=True)
            raw.append(logits); centered.append(scores)
            logp = (scores*8).log_softmax(-1)
            for a in ARMS:
                chosen = logp.gather(1, ids[a][i,:,None]).squeeze(1)
                parts[a].append(-chosen[masks[a][i]].sum())
                logs[a].append(chosen.detach().cpu().numpy())
    losses = {a:torch.stack(parts[a]).sum()/masks[a].sum() for a in ARMS}
    return losses, {a:np.stack(logs[a]) for a in ARMS}, tuple(centered), tuple(raw)


def moments(a, b, axes):
    """Sufficient statistics for cosine and shared/differential gradient norms."""
    a, b = np.asarray(a,np.float64), np.asarray(b,np.float64)
    return np.stack(((a*a).sum(axis=axes), (b*b).sum(axis=axes), (a*b).sum(axis=axes)), -1)


def summarize_moments(values):
    aa, bb, ab = np.asarray(values,np.float64).reshape(-1,3).sum(0)
    difference = max(0., float(aa+bb-2*ab))
    shared = max(0., float(aa+bb+2*ab))
    return dict(forward_norm=float(np.sqrt(aa)), reverse_norm=float(np.sqrt(bb)),
        cosine=float(np.clip(ab/np.sqrt(aa*bb),-1,1)) if aa*bb else None,
        difference_norm=math.sqrt(difference), shared_mean_norm=.5*math.sqrt(shared),
        differential_to_shared_ratio=math.sqrt(difference/shared) if shared else None)


def feature_moments(a, b):
    # Convert one frame at a time. Do not archive full Q/K or gradients (~hundreds of MB).
    return np.stack([moments(x.float().numpy(),y.float().numpy(),axes=(0,2))
                     for x,y in zip(a,b)])


def prepare_qk_inputs(pilot, attribution, huber, gradients, response, destination, torso):
    checked = prepare_torso_inputs(pilot, attribution, huber, gradients, response, destination)
    inputs, _, indices, _, _, provenance, geometry = checked
    refs, support = common_torso_refs(inputs[2])
    root = Path(torso)
    protocol = json.loads((root/'started.json').read_text())
    report = json.loads((root/'torso_report.json').read_text())
    if protocol['inputs_sha256'] != inputs[4] or protocol['control_sha256'] != provenance:
        raise ValueError('Torso control used different saved inputs')
    if not report['baseline_unchanged'] or report['counts'] != dict(
            positive_forwards_attempted=3, positive_forwards_completed=3,
            latent_backwards_attempted=2, latent_backwards_completed=2, first_adam_proposals=2):
        raise ValueError('Requires completed common-torso control')
    for name, expected in {**protocol['helper_sources_sha256'],
                          'wan_centered_torso.py':protocol['script_sha256']}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Control helper source changed: {name}')
    paths = [root/'started.json',root/'torso_report.json',root/'before_flow.npz',
             root/'before_target_log_probability.npz',root/'optimization_support.npz']
    with np.load(paths[2],allow_pickle=False) as data:
        np.testing.assert_array_equal(data['flow'],inputs[5]['before'])
    with np.load(paths[4],allow_pickle=False) as data:
        np.testing.assert_array_equal(data['common_torso'],support)
        for a in ARMS:
            np.testing.assert_array_equal(data[a+'_original_valid'],inputs[2][a][1])
    with np.load(paths[3],allow_pickle=False) as data:
        before_logp = {a:data[a].copy() for a in ARMS}
    for a,v in before_logp.items():
        if v.shape != support.shape or not np.isfinite(v).all() or (v>1e-6).any():
            raise ValueError('Invalid saved target log probabilities')
        actual = support_nll(before_logp,support)[a]
        expected = report['before_optimization_nll'][a]
        if not np.isclose(actual['mean'],expected['mean'],rtol=1e-9,atol=1e-9) or not np.allclose(
                actual['per_pair'],expected['per_pair'],rtol=1e-9,atol=1e-9):
            raise ValueError('Saved before NLL does not match control')
    saved_gradients = {}
    for a in ARMS:
        path=root/f'{a}_latent_gradient.npz';paths.append(path)
        with np.load(path,allow_pickle=False) as data:
            grad=data['gradient'].copy()
        if grad.shape != inputs[1]['before'].shape or grad.dtype != np.float32 or not np.isfinite(grad).all():
            raise ValueError('Invalid saved latent gradient')
        saved_gradients[a]=grad
    latent = compare(saved_gradients['forward'],saved_gradients['reverse'])
    for key,value in latent.items():
        expected=report['latent_gradient_comparison'][key]
        if isinstance(value,str) or value is None:
            if value != expected:
                raise ValueError('Saved gradient comparison disagrees')
        elif not np.isclose(value,expected,rtol=1e-6,atol=1e-9):
            raise ValueError('Saved gradient comparison disagrees')
    return dict(inputs=inputs,refs=refs,support=support,indices=indices,before_logp=before_logp,
        latent_comparison=latent, geometry=geometry,
        provenance={**provenance,'torso_artifacts':{str(p.relative_to(root)):digest(p) for p in paths}})


def run_qk(g, pilot, attribution, huber, gradients, response, destination, torso, output):
    from benchmark.wan_centered_pilot import clear
    output=Path(output)
    if (output/'started.json').exists():
        raise RuntimeError('Q/K diagnostic budget already started; preserve existing artifacts')
    prepared=prepare_qk_inputs(pilot,attribution,huber,gradients,response,destination,torso)
    manifest,latents,_,_,hashes,old=prepared['inputs']
    live=frozen_model(g,manifest)
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(
        hypothesis='The score-to-Q/K mapping may already align the two references gradients before the remaining transformer backward.',
        expected='Measure centered-score, raw-score, Q-only, K-only and joint-Q/K cosine, then compare with the archived latent-gradient cosine.',
        max_positive_forwards=1,max_readout_backwards=2,max_latent_backwards=0,
        max_optimizer_steps=0,max_scheduler_steps=0,max_decodes=0,
        fixed=dict(index=39,sigma=manifest['experiment']['guidance_sigma'],block=20,heads='all 40',
                   multiplier=8,loss='destination NLL',support='same 324 common torso queries',injection=False),
        inputs_sha256=hashes,control_sha256=prepared['provenance'],live=live,
        script_sha256=digest(__file__),helper_sources_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ('wan_centered_torso.py','wan_centered_destination.py','wan_centered_response.py',
             'wan_centered_gradients.py','wan_centered_huber.py','wan_centered_attribution.py','wan_centered_pilot.py')},
        limits='One Wan capture under no_grad. Backward stops at detached native-dtype Q/K leaves. '
            'Joint Q/K concatenates feature coordinates; gradient cosines are coordinate-dependent. '
            'A later change locates alignment, not necessarily an implementation bug or motion cause. '
            'No full Q/K archives or gradients are exported; save exact per-frame/head norm/dot sufficient statistics.')
    with (output/'started.json').open('x',encoding='utf-8') as handle:
        json.dump(protocol,handle,indent=2)
    print('HYPOTHESIS:',protocol['hypothesis'],flush=True)
    print('LIMIT: one no-grad model capture; two readout-only backwards; no latent backwards or updates.',flush=True)
    progress=dict(positive_forwards_attempted=0,positive_forwards_completed=0,
                  readout_backwards_attempted=0,readout_backwards_completed=0)
    first=None
    try:
        with torch.no_grad():
            x=torch.from_numpy(latents['before']).to(g.device)
            progress['positive_forwards_attempted']=1;write_json(output/'progress.json',progress)
            flow,q0,k0=capture(g,x)
            progress['positive_forwards_completed']=1;write_json(output/'progress.json',progress)
            before=flow.float().cpu().numpy().copy()
            if not np.isfinite(before).all() or not np.allclose(before,old['before'],rtol=.005,atol=.005):
                raise RuntimeError('Before flow replay failed; no readout backwards performed')
            np.savez_compressed(output/'before_flow.npz',flow=before)
            q,k=q0.detach().requires_grad_(True),k0.detach().requires_grad_(True)
            del flow,q0,k0
        clear(g)
        with torch.enable_grad():
            losses,logp,centered,raw=readout_with_taps(q,k,prepared['refs'],prepared['indices'])
            replay={}
            for a in ARMS:
                expected=prepared['before_logp'][a]
                # Whole fields, not only scalar loss. No extra model capture on discrepancy.
                if not np.isfinite(logp[a]).all() or not np.allclose(logp[a],expected,rtol=.0005,atol=.005):
                    raise RuntimeError('Before target probabilities do not replay; no backwards performed')
                replay[a]=dict(log_probability_max_abs=float(np.abs(logp[a]-expected).max()),loss=float(losses[a].detach()))
            np.savez_compressed(output/'before_target_log_probability.npz',**logp)
            np.savez_compressed(output/'optimization_support.npz',common_torso=prepared['support'])
            write_json(output/'replay.json',dict(flow_max_abs=float(np.abs(before-old['before']).max()),arms=replay))
            stages={};gradient_properties={}
            for i,a in enumerate(ARMS):
                progress['readout_backwards_attempted']+=1;write_json(output/'progress.json',progress)
                values=torch.autograd.grad(losses[a],(q,k,*centered,*raw),retain_graph=i==0)
                if not all(torch.isfinite(v).all() for v in values):
                    raise RuntimeError('Nonfinite readout gradient')
                cpu=tuple(v.detach().cpu() for v in values)
                del values
                # Counts/max are diagnostic; magnitudes across different variable spaces are not comparable.
                gradient_properties[a]={name:dict(dtype=str(v.dtype),shape=list(v.shape),
                    nonzero_count=int(torch.count_nonzero(v)),count=v.numel(),max_abs=float(v.abs().max()))
                    for name,v in zip(('Q','K'),cpu[:2])}
                if not any(gradient_properties[a][name]['nonzero_count'] for name in ('Q','K')):
                    raise RuntimeError('Zero joint Q/K gradient')
                if i==0:
                    first=cpu
                else:
                    stages['Q']=feature_moments(first[0],cpu[0])
                    stages['K']=feature_moments(first[1],cpu[1])
                    pairs=len(centered)
                    for name,start in [('centered_scores',2),('raw_scores',2+pairs)]:
                        stages[name]=np.stack([moments(f.float().numpy(),r.float().numpy(),axes=None)
                            for f,r in zip(first[start:start+pairs],cpu[start:start+pairs])])
                progress['readout_backwards_completed']+=1;write_json(output/'progress.json',progress)
                print(a,'readout backward completed; Q/K leaves only.',flush=True)
            if q.grad is not None or k.grad is not None or x.requires_grad or x.grad is not None:
                raise RuntimeError('Unexpected accumulated gradient or latent graph')
            np.testing.assert_array_equal(x.cpu().numpy(),latents['before'])
        np.savez_compressed(output/'gradient_moments.npz',**stages)
        summary={name:summarize_moments(v) for name,v in stages.items()}
        summary['joint_QK']=summarize_moments(np.concatenate([stages['Q'],stages['K']],axis=0))
        report=dict(port_success=False,status='Q/K gradient-stage diagnostic only; no motion claim',counts=progress,
            baseline_unchanged=True,latent_graph_created=False,qk_dtype=str(q.dtype),
            replay=replay,flow_replay_max_abs=float(np.abs(before-old['before']).max()),
            stage_comparison=summary,archived_latent_comparison=prepared['latent_comparison'],
            gradient_properties=gradient_properties,
            by_frame={name:[dict(frame=i,**summarize_moments(v)) for i,v in enumerate(stages[name])] for name in ('Q','K')},
            by_head={name:[dict(head=i,**summarize_moments(stages[name][:,i])) for i in range(stages[name].shape[1])] for name in ('Q','K')},
            score_by_pair={name:[dict(pair=[i,i+1],**summarize_moments(v)) for i,v in enumerate(stages[name])]
                           for name in ('centered_scores','raw_scores')},
            interpretation='Q/K near-alignment places it in the score-to-feature mapping including native-dtype readout backward. '
                'Distinct Q/K gradients followed by aligned archived latent gradients place it in the remaining transformer/precision path. '
                'Check Q and K separately: joint norms can hide a smaller component. '
                'Per-head statistics describe this one all-head objective, not alternative head trials. '
                'This is directional specificity, not proof of vanishing gradients, a code bug or decoded motion transfer.')
        write_json(output/'qk_report.json',report)
        return report
    except Exception as error:
        write_json(output/'failure.json',dict(error_type=type(error).__name__,error=str(error),counts=progress))
        raise
    finally:
        clear(g)
