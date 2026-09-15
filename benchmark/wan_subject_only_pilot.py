"""Frozen subject-only comparison against the completed 50/50 subject experiment."""
import base64
import hashlib
import html
import json
import math
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from benchmark import wan_subject_pilot as previous
from benchmark import wan_head_pilot as head
from benchmark import wan_pair_pilot as pairs
from benchmark import wan_control_pilot as control
from probe_report import load_trace

STAGE = 'subject_only_step9_v1'
ARMS = ('subject_only_forward', 'subject_only_reverse')
PROTOCOL = dict(previous.PROTOCOL, stage=STAGE, arms=list(ARMS), subject_weight=1., background_weight=0.)
read_json, arrays, inventory = previous.read_json, previous.arrays, previous.inventory
rms, cosine = previous.rms, previous.cosine


def environment_snapshot():
    value = previous.environment_snapshot()
    for name in ('benchmark/wan_subject_only_pilot.py', 'probe_wan_subject_only.py'):
        value['source_sha256'][name] = head.digest(name)
    code = '''import hashlib, inspect, json
from importlib.metadata import version, PackageNotFoundError
from diffusers.pipelines.wan import pipeline_wan as wan
from diffusers.utils import is_ftfy_available
if not is_ftfy_available():
    raise RuntimeError('Missing ftfy: install ftfy==6.3.1 and restart the kernel before setup.')
def installed(name):
    try: return version(name)
    except PackageNotFoundError: return None
print(json.dumps(dict(packages={p: installed(p) for p in
('ftfy','tokenizers','sentencepiece','protobuf','safetensors','regex')},
cleaner_sha256=hashlib.sha256((inspect.getsource(wan.basic_clean)+inspect.getsource(wan.whitespace_clean)+inspect.getsource(wan.prompt_clean)).encode()).hexdigest(),
cleaning_canary=wan.prompt_clean(chr(0xff0c)))))'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError('Text-processing preflight failed before model loading:\n'+result.stderr[-3000:])
    value['text_processing'] = json.loads(result.stdout.strip().splitlines()[-1])
    if value['text_processing']['cleaning_canary'] != ',':
        raise ValueError('Wan text cleaning is inconsistent with the tested ftfy behavior')
    return value


def restore_archive(archive, destination):
    """Restore a complete subject results ZIP, with safe paths and immutable caches."""
    destination = Path(destination).expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.wan_subject_only_restore_', dir=destination.parent) as temp:
        staging = Path(temp); local_zip = staging/'results.zip'
        shutil.copyfile(Path(archive).expanduser(), local_zip)
        with zipfile.ZipFile(local_zip) as bundle:
            files = {}; seen = set()
            for info in bundle.infolist():
                name = PurePosixPath(info.filename.replace('\\', '/'))
                if (name.is_absolute() or '..' in name.parts or any(':' in p for p in name.parts)
                        or stat.S_ISLNK(info.external_attr >> 16) or name in seen):
                    raise ValueError('Unsafe or duplicate archive member: '+info.filename)
                seen.add(name)
                if not info.is_dir(): files[name] = info
            markers = ('plan.sha256', 'environment.json', *(a+'_done.json' for a in previous.ARMS))
            roots = [p.parent for p in files if p.name=='plan.json'
                     and all(p.parent/m in files for m in markers)
                     and json.loads(bundle.read(files[p])).get('stage')==previous.STAGE]
            if len(roots) != 1:
                raise ValueError('Use the full completed SUBJECT results ZIP with all four arm markers, not the review ZIP or control ZIP')
            restored = staging/'result'; restored.mkdir()
            for name, info in files.items():
                if name.is_relative_to(roots[0]):
                    target = restored.joinpath(*name.relative_to(roots[0]).parts); target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, target.open('wb') as output: shutil.copyfileobj(source, output)
        expected = inventory(restored)
        if destination.exists():
            if destination.is_dir() and inventory(destination)==expected: return destination
            identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:16]
            destination = destination.with_name(destination.name+'_restored_'+identity)
            if destination.exists():
                if destination.is_dir() and inventory(destination)==expected: return destination
                raise ValueError('Restored cache differs; choose a new destination')
        if destination.parent.resolve()!=staging.parent.resolve() or destination.exists():
            raise ValueError('Restore requires an unused sibling of the staging directory')
        restored.rename(destination)
    print('Restored completed subject experiment:', destination)
    return destination


def require_same_environment(expected, current):
    def differences(a, b, prefix=''):
        if isinstance(a, dict) and isinstance(b, dict):
            return [line for key in sorted(a.keys() | b.keys()) for line in
                    differences(a.get(key, '<not recorded>'), b.get(key, '<not recorded>'), prefix+key+'.')]
        return [] if a==b else [f'{prefix.rstrip(".")}: recorded={a!r}, current={b!r}']
    changed = differences(expected, current)
    if changed:
        raise RuntimeError('Subject-only experiment environment changed:\n  '+'\n  '.join(changed)+
                           '\nRestore the recorded environment; keep the archived evidence unchanged.')


def direction_of(arm):
    if arm not in ARMS: raise ValueError('Unknown subject-only arm')
    return arm.rsplit('_', 1)[1]


def baseline_directory(root, arm):
    return pairs.completed(Path(root)/'previous', 'balanced_'+direction_of(arm))


def fixed_config(plan, arm, output):
    config = control.fixed_config(plan, direction_of(arm), output)
    config.update(visual_protocol=STAGE, alignment_mode='subject_only',
                  alignment_target=str(Path(plan['target_root'])/(direction_of(arm)+'.npz')),
                  record_region_gradients=True)
    return config


def validate_previous(root):
    root = Path(root); plan = head.load_plan(root)
    if any(plan.get(k)!=v for k,v in previous.PROTOCOL.items()):
        raise ValueError('Use the completed frozen subject-alignment experiment')
    if inventory(root/'previous')!=plan['previous_inventory'] or inventory(root/'targets')!=plan['target_inventory']:
        raise ValueError('Changed prior evidence or frozen targets')
    if head.digest(root/'input_manifest.csv')!=plan['input_manifest_sha256']:
        raise ValueError('Changed prior input manifest')
    _, controls = previous.validate_previous(root/'previous')
    reports = {a: validate_result(pairs.completed(root,a), root, plan, a, prior=True) for a in previous.ARMS}
    return plan, dict(arms=reports, controls=controls)


def make_plan(inputs, root, prior, input_zip=None):
    root, prior = Path(root).resolve(), Path(prior).resolve()
    if root==prior or root in prior.parents or prior in root.parents:
        raise ValueError('Create a fresh experiment separate from prior evidence')
    if root.exists() and any(root.iterdir()): raise ValueError('Use a fresh setup directory')
    env = environment_snapshot()
    previous.require_reusable_environment(read_json(prior/'environment.json'), env)
    old_plan, audit = validate_previous(prior)
    inputs, rows = head.direction.prepare_inputs(inputs, input_zip)
    if head.digest(inputs/'manifest.csv')!=old_plan['input_manifest_sha256']:
        raise ValueError('Reference input manifest changed')
    car = next(r for r in rows if r['clip_id']=='car-turn')
    plan = dict(PROTOCOL, inputs=str(inputs), input_manifest_sha256=old_plan['input_manifest_sha256'],
                prompt=old_plan['prompt'], clip_id='car-turn', video=str(inputs/car['video_path']),
                generation_video=str(inputs/car['video_path']), reverse_video=str(root/'previous/previous/reference_reverse'),
                target_root=str(root/'targets'), target_inventory=old_plan['target_inventory'],
                previous_inventory=inventory(prior), previous_name=prior.name,
                annotation_sha256=old_plan['annotation_sha256'])
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(prior, root/'previous'); shutil.copytree(prior/'targets', root/'targets')
    head.write_json(root/'environment.json', env); head.write_json(root/'previous_audit.json', audit)
    plan['previous_audit_sha256'] = head.digest(root/'previous_audit.json')
    head.write_json(root/'plan.json', plan); (root/'plan.sha256').write_text(head.digest(root/'plan.json'), encoding='utf-8')
    shutil.copyfile(inputs/'manifest.csv', root/'input_manifest.csv')
    return plan


def checked_plan(root):
    root = Path(root); plan = head.load_plan(root)
    if any(plan.get(k)!=v for k,v in PROTOCOL.items()): raise ValueError('Subject-only protocol changed; create a fresh setup')
    if inventory(root/'previous')!=plan['previous_inventory'] or inventory(root/'targets')!=plan['target_inventory']:
        raise ValueError('Previous results or frozen targets changed')
    prior = head.load_plan(root/'previous')
    if plan['prompt']!=prior['prompt'] or plan['target_inventory']!=prior['target_inventory']:
        raise ValueError('Prompt or mapped targets differ from the matched subject experiment')
    inputs, _ = head.checked_inputs(plan)
    if (head.digest(inputs/'manifest.csv')!=plan['input_manifest_sha256']
            or head.digest(root/'input_manifest.csv')!=plan['input_manifest_sha256']
            or head.digest(root/'previous_audit.json')!=plan['previous_audit_sha256']):
        raise ValueError('Input manifest or prior audit changed')
    if plan['generation_video']!=plan['video']:
        raise ValueError('Generation input changed')
    if (Path(plan['target_root']).resolve()!=(root/'targets').resolve()
            or Path(plan['reverse_video']).resolve()!=(root/'previous/previous/reference_reverse').resolve()):
        raise ValueError('Restore the experiment and reference inputs at their recorded runtime paths')
    return plan


def require_previous_reference(directory, root, arm):
    directory, old = Path(directory), baseline_directory(root,arm)
    control.require_common_start(read_json(directory/'initial_state.json'), read_json(old/'initial_state.json'))
    if (read_json(directory/'reference_protocol.json')!=read_json(old/'reference_protocol.json')
            or not pairs.npz_equal(directory/'reference_inputs.npz', old/'reference_inputs.npz')):
        raise ValueError('Reference encoding/noise changed')
    a,_,ae = load_trace(directory); b,_,be = load_trace(old)
    for name in ('training_reference','selected_pair_reference','aligned_reference'):
        if not pairs.npz_equal(a/pairs.one_event(ae,name)['file'], b/pairs.one_event(be,name)['file']):
            raise ValueError('Source/mapped reference changed: '+name)


def require_previous_start(directory, root, arm):
    directory, old = Path(directory), baseline_directory(root,arm)
    if read_json(directory/'step9_state.json')!=read_json(old/'step9_state.json'):
        raise ValueError('Pre-update latent or scheduler changed')
    control.require_same_capture(directory/'control/before_09.npz', old/'control/before_09.npz')
    a,_,ae = load_trace(directory); b,_,be = load_trace(old)
    af = next(e for e in ae if e['kind']=='full_pair_target'); bf = next(e for e in be if e['kind']=='full_pair_target')
    if not pairs.npz_equal(a/af['file'], b/bf['file']): raise ValueError('Pre-update AMF changed')
    if head.timing.decoded_digest(directory/'estimated_clean_before.mp4')!=head.timing.decoded_digest(old/'estimated_clean_before.mp4'):
        raise ValueError('Pre-update decoded estimate changed')


def audit_losses(directory, target_path):
    trace,_,events = load_trace(directory); target = arrays(target_path)
    ref = pairs.one_event(events,'aligned_reference')
    if ref['mode']!='subject_only' or not pairs.npz_equal(trace/ref['file'],target_path):
        raise ValueError('Subject-only target differs from frozen target')
    captures = [e for e in events if e['kind']=='full_pair_target']
    records = [e for e in events if e['kind']=='aligned_loss']; totals = [e for e in events if e['kind']=='full_pair_loss']
    if not len(captures)==len(records)==len(totals)==7: raise ValueError('Missing subject-only loss evidence')
    rows=[]
    for capture, record, total in zip(captures,records,totals):
        if (capture['file']!=record['file'] or capture['file']!=total['file'] or record['mode']!='subject_only'
                or capture['flow_head']!=30 or capture['q_dtype']!='torch.bfloat16' or capture['k_dtype']!='torch.bfloat16'):
            raise ValueError('Changed subject-only loss identity or precision')
        flow=arrays(trace/capture['file'])['flow']
        if flow.shape!=target['flow'].shape or not np.isfinite(flow).all(): raise ValueError('Invalid AMF field')
        score=previous.region_scores(flow,target,'balanced')
        score.update(total=score['subject_mse'],subject_weight=1.)
        if any(not math.isclose(v,record[k],rel_tol=2e-5,abs_tol=2e-5) for k,v in score.items()):
            raise ValueError('Subject-only loss failed recomputation')
        if total['loss']!=record['total'] or record['total']!=record['subject_mse']:
            raise ValueError('The actual optimized loss must equal the subject mean')
        rows.append(score)
    return rows


def gradient_rows(directory):
    directory=Path(directory); _,_,events=load_trace(directory)
    updates=[e for e in events if e['kind']=='optimization']; records=[e for e in events if e['kind']=='region_gradient']
    paths=sorted((directory/'region_gradients').glob('*.npz'))
    if not len(updates)==len(records)==len(paths)==5: raise ValueError('Missing subject-only gradients')
    values=[arrays(p) for p in paths]
    first=arrays(directory/'control/before_09.npz')['latent']; last=arrays(directory/'control/after_09.npz')['latent']
    if not np.array_equal(values[0]['latent'],first): raise ValueError('Gradient starting latent changed')
    rows=[]
    for i,(value,event) in enumerate(zip(values,records)):
        if set(value)!={'latent','subject_gradient','background_gradient'} or any(
            a.dtype!=np.float32 or a.shape!=first.shape or not np.isfinite(a).all() for a in value.values()):
            raise ValueError('Invalid region-gradient arrays')
        if (event['iteration']!=i or event['step']!=9 or directory/event['file']!=paths[i]
                or paths[i].name!=f'iteration_{i:02d}.npz' or event['subject_weight']!=1. or event['background_weight']!=0.):
            raise ValueError('Background gradient must have zero optimization weight')
        update=(values[i+1]['latent'] if i<4 else last).astype(float)-value['latent']
        if not math.isclose(rms(update),updates[i]['update']['rms'],rel_tol=2e-5,abs_tol=1e-8):
            raise ValueError('Gradient trajectory disagrees with actual Adam update')
        fg,bg=value['subject_gradient'],value['background_gradient']
        rows.append(dict(iteration=i,subject_gradient_rms=rms(fg),background_gradient_rms=rms(bg),
            subject_weight=1.,background_weight=0.,weighted_subject_gradient_rms=rms(fg),weighted_background_gradient_rms=0.,
            gradient_cosine=cosine(fg,bg),update_rms=rms(update),
            update_cosine_to_negative_subject_gradient=cosine(update,-fg),update_cosine_to_negative_background_gradient=cosine(update,-bg)))
    return rows


def validate_result(directory, root, plan, arm, prior=False):
    import imageio.v3 as iio
    from omegaconf import OmegaConf
    directory,root=Path(directory),Path(root); _,metadata,events=load_trace(directory); config=metadata['config']
    expected=(previous.fixed_config if prior else fixed_config)(plan,arm,config['output_path'])
    def canonical(k,v): return v.replace('\\','/') if k=='alignment_target' and isinstance(v,str) else v
    if (any(canonical(k,config.get(k))!=canonical(k,v) for k,v in expected.items())
            or config!=OmegaConf.to_container(OmegaConf.load(directory/'suite_config.yaml'))
            or read_json(directory/'complete.json')!=dict(arm=arm,native_rope_unchanged=True,frozen_weights=True)):
        raise ValueError('Changed configuration or incomplete generation')
    report=control.noised.audit_trace(events,config,'candidate')
    target=root/'targets'/(arm.rsplit('_',1)[1]+'.npz')
    report['losses']=previous.audit_losses(directory,target,config['alignment_mode']) if prior else audit_losses(directory,target)
    expected_captures={'before_09','after_09',*(f'denoise_{i:02d}' for i in control.CHECKPOINTS)}
    if {p.stem for p in (directory/'control').glob('*.npz')}!=expected_captures: raise ValueError('Missing sampler captures')
    report['captures']=[control.audit_capture(directory/'control'/f'{name}.npz') for name in sorted(expected_captures)]
    control.require_same_capture(directory/'control/after_09.npz',directory/'control/denoise_09.npz')
    (previous.require_previous_reference if prior else require_previous_reference)(directory,root,arm)
    (previous.require_previous_start if prior else require_previous_start)(directory,root,arm)
    report['gradients']=previous.gradient_rows(directory,config['alignment_mode']) if prior else gradient_rows(directory)
    for name in ('original.mp4','final.mp4','estimated_clean_before.mp4','estimated_clean_after.mp4'):
        frames=list(iio.imiter(directory/name,plugin='FFMPEG'))
        if len(frames)!=21 or any(f.shape!=(480,832,3) for f in frames): raise ValueError('Incomplete video: '+name)
    return report


def comparison_folders(root):
    root=Path(root)
    return {**{'previous_balanced_'+d:pairs.completed(root/'previous','balanced_'+d) for d in ('forward','reverse')},
            **{a:pairs.completed(root,a) for a in ARMS}}


def response_rows(root):
    root=Path(root); off=pairs.completed(root/'previous/previous','off'); rows=[]
    for arm,directory in comparison_folders(root).items():
        update=arrays(directory/'control/after_09.npz')['latent'].astype(float)-arrays(directory/'control/before_09.npz')['latent']
        for step in control.CHECKPOINTS:
            a=arrays(directory/'control'/f'denoise_{step:02d}.npz'); b=arrays(off/'control'/f'denoise_{step:02d}.npz')
            meta=read_json(directory/'control'/f'denoise_{step:02d}.json')
            delta=a['latent'].astype(float)-b['latent']; velocity=a['cfg_velocity'].astype(float)-b['cfg_velocity']
            next_delta=a['next_latent'].astype(float)-b['next_latent']
            rows.append(dict(arm=arm,step=step,guidance_update_rms=rms(update),cfg_velocity_difference_rms=rms(velocity),
                latent_difference_after_sampler_rms=rms(next_delta),difference_relative_to_off_rms=rms(next_delta)/rms(b['next_latent']),
                projection_on_initial_update=float(np.sum(next_delta*update)/np.sum(update**2)),
                paired_euler_residual_rms=rms(delta+(meta['sigma_next']-meta['sigma'])*velocity-next_delta)))
    return rows


def field_rows(root):
    root=Path(root); targets={d:arrays(root/'targets'/f'{d}.npz') for d in ('forward','reverse')}
    motion,selectivity=[],[]
    for arm,directory in comparison_folders(root).items():
        trace,_,events=load_trace(directory); fields=[e for e in events if e['kind']=='full_pair_target']
        for label,event in [('before',fields[0]),('after',fields[-1])]:
            flow=arrays(trace/event['file'])['flow'].astype(float); target=targets[arm.rsplit('_',1)[1]]
            region=target['subject']; pred=flow[region]; ref=target['flow'][region].astype(float)
            motion.append(dict(arm=arm,stage=label,subject_mse=float(np.square(pred-ref).mean()),
                subject_cosine=cosine(pred,ref),prediction_mean_dx=float(pred[:,0].mean()),prediction_mean_dy=float(pred[:,1].mean()),
                reference_mean_dx=float(ref[:,0].mean()),reference_mean_dy=float(ref[:,1].mean()),zero_field_mse=float(np.square(ref).mean())))
            for region in ('subject','background'):
                common=targets['forward'][region]&targets['reverse'][region]
                if not common.any(): raise ValueError('No common support for reference selectivity')
                error={d:float(np.square(flow-t['flow'])[common].mean()) for d,t in targets.items()}
                selectivity.append(dict(arm=arm,stage=label,region=region,common_positions=int(common.sum()),
                    forward_mse=error['forward'],reverse_mse=error['reverse'],forward_preference=error['reverse']-error['forward']))
    return motion,selectivity


def make_display(root, report):
    root=Path(root); off=pairs.completed(root/'previous/previous','off')
    videos=[(d+' reference',pairs.completed(root/'previous','balanced_'+d)/'original.mp4') for d in ('forward','reverse')]
    videos += [('Wan AMF off',off/'final.mp4')]
    videos += [(a.replace('_',' '),p/'final.mp4') for a,p in comparison_folders(root).items()]
    page=['<!doctype html><meta charset="utf-8"><title>Wan subject-only comparison</title>',
          '<style>body{font:16px system-ui;margin:24px;color:#17202a;background:#f7f8fa}section{display:flex;flex-wrap:wrap;gap:14px}figure{margin:0;flex:1 1 380px;max-width:832px}video{width:100%}figcaption{padding:10px 0}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style>',
          '<h1>Does removing background loss improve subject motion?</h1><p>Two new Wan 14B videos. '
          'Same mapped targets, head 30, seed 1, LR 0.001, five updates at index 9. '
          'Previous: 50/50 subject/background means. New: subject mean only.</p>',
          '<button onclick="document.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">Play all from start</button> '
          '<button onclick="document.querySelectorAll(\'video\').forEach(v=>v.pause())">Pause all</button><section>']
    for label,path in videos:
        page.append('<figure><figcaption>'+html.escape(label)+'</figcaption><video controls muted loop playsinline src="data:video/mp4;base64,'+base64.b64encode(path.read_bytes()).decode()+'"></video></figure>')
    page.append('</section><h2>Subject AMF after the five updates</h2><table><tr><th>Arm</th><th>MSE</th><th>Cosine</th><th>Mean dx</th><th>Target dx</th></tr>')
    for row in report['motion']:
        if row['stage']=='after':
            page.append('<tr><td>'+html.escape(row['arm'])+'</td>'+''.join('<td>'+('undefined' if row[k] is None else f'{row[k]:.4g}')+'</td>' for k in
                ('subject_mse','subject_cosine','prediction_mean_dx','reference_mean_dx'))+'</tr>')
    page.append('</table><p>These are index-9 correspondence metrics, not decoded motion scores. '
                'Check the final truck trajectory, size change, heading and appearance. The fixed approach/grow prompt conflicts with the reversed reference.</p>')
    (root/'subject_only_comparison.html').write_text('\n'.join(page),encoding='utf-8')


def summarize(root):
    root=Path(root); plan=checked_plan(root)
    audits={a:validate_result(pairs.completed(root,a),root,plan,a) for a in ARMS}
    # These prior arms were fully audited in setup; their files remain hash-frozen in checked_plan.
    prior=read_json(root/'previous_audit.json')['arms']
    combined={**{'previous_'+a:prior[a] for a in ('balanced_forward','balanced_reverse')},**audits}
    losses=[dict(arm=a,capture=i,**row) for a,r in combined.items() for i,row in enumerate(r['losses'])]
    gradients=[dict(arm=a,**row) for a,r in combined.items() for row in r['gradients']]
    motion,selectivity=field_rows(root); response=response_rows(root)
    for name,rows in [('region_losses',losses),('region_gradients',gradients),('subject_motion',motion),
                      ('subject_selectivity',selectivity),('sampler_response',response)]:
        control.write_csv(root/(name+'.csv'),rows)
    report=dict(arms=audits,motion=motion,selectivity=selectivity,response=response,gradients=gradients)
    head.write_json(root/'subject_only_audit.json',report)
    if not (root/'visual_review.csv').exists():
        control.write_csv(root/'visual_review.csv',[dict(arm=a,heading='',lateral_motion='',size_change='',turning='',background_motion='',artifacts='',notes='') for a in ARMS])
    make_display(root,report)
    return report


def run(root):
    root=Path(root).resolve(); plan=checked_plan(root)
    require_same_environment(read_json(root/'environment.json'),environment_snapshot())
    for arm in ARMS:
        marker=root/f'{arm}_done.json'
        if marker.is_file():
            validate_result(pairs.completed(root,arm),root,plan,arm)
            print('Already complete and revalidated:',arm,flush=True); continue
        directory=root/(arm+'_'+head.stamp()); log=directory.with_suffix('.log')
        command=[sys.executable,'-u','probe_wan_subject_only.py','--plan',str(root/'plan.json'),'--arm',arm,'--output_path',str(directory)]
        print('Generating:',arm,'| log:',log,flush=True)
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Failed attempt retained at {log}; inspect its error before retrying')
        audit=validate_result(directory,root,plan,arm)
        head.write_json(marker,dict(directory=directory.name,audit=audit))
    return summarize(root)
