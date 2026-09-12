"""Car-turn-only, staged 14B readout and sampled-latent response experiment."""
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from benchmark import wan_direction_pilot as direction
from benchmark import wan_timing_pilot as timing

CONTROLS = ['static', 'pan_right', 'pan_left', 'expand', 'contract']
MODEL = timing.MODEL_KEYS['14b']
EXTRA_SOURCES = ['probe_wan_affine.py', 'probe_wan_response.py', 'benchmark/wan_response_pilot.py']


def environment_snapshot():
    snapshot = direction.environment_snapshot()
    for name in EXTRA_SOURCES:
        snapshot['source_sha256'][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    return snapshot


def make_plan(inputs, root, archive=None):
    inputs, rows = direction.prepare_inputs(inputs, archive)
    row = next(r for r in rows if r['clip_id']=='car-turn')
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Setup needs a fresh directory; restore RESPONSE_OUT and run a stage to resume')
    root.mkdir(parents=True, exist_ok=True)
    video = str(inputs/row['video_path'])
    prompt = row['prompt'].rstrip().rstrip('.')+'. '+direction.DIRECTION_SUFFIXES['car-turn']
    plan = dict(schema_version=1, clip_id='car-turn', model=MODEL, inputs=str(inputs), prompt=prompt,
        readout_command=[sys.executable,'probe_wan_affine.py','-v',video,'--model','14b','--low_vram',
            '--mean_only','--blocks','10','--controls',*CONTROLS,'--noise_steps','0','9','29'],
        response_command=[sys.executable,'probe_wan_response.py','-v',video,'--prompt',prompt,
            '--branch_step','9','--updates','5'])
    (root/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (root/'environment.json').write_text(json.dumps(environment_snapshot(),indent=2),encoding='utf-8')
    (root/'input_manifest.csv').write_bytes((inputs/'manifest.csv').read_bytes())
    return plan


def audit_readout(root):
    import numpy as np
    root = Path(root)
    meta = json.loads((root/'metadata.json').read_text())
    rows = json.loads((root/'metrics.json').read_text())
    done = json.loads((root/'complete.json').read_text())
    if (meta['model']!=MODEL or meta['blocks']!=[10] or meta['controls']!=CONTROLS
            or meta.get('mean_only') is not True or meta.get('cpu_offload') is not True
            or meta['conditioning']!='' or meta['grid']!=[6,30,52]
            or [s['sampling_index'] for s in meta['noise_states']]!=[-1,0,9,29]):
        raise ValueError('Readout does not match the planned 14B baseline control suite')
    expected = {(c,s['noise_label'],o,support,field) for c in CONTROLS for s in meta['noise_states']
                for o in (-2,0,2) for support in ('geometry','textured') for field in ('hard','soft')}
    observed = {(r['control'],r['noise_label'],r['anchor_offset'],r['support'],r['field']) for r in rows}
    if observed!=expected or len(rows)!=len(expected) or done!={'forward_passes':20,'rows':len(rows)}:
        raise ValueError('Incomplete or duplicated affine observations')
    for row in rows:
        if row['variant']!='mean_logits' or row['block']!='block_10_attn1_processor' or row['finite_fraction']!=1:
            raise ValueError('Invalid affine readout row')
    pure = []
    for control in CONTROLS:
        for state in meta['noise_states']:
            with np.load(root/control/state['noise_label']/'block_10_attn1_processor/mean_logits.npz') as z:
                if z['soft'].shape!=(5,1560,2) or not all(np.isfinite(z[k]).all() for k in z.files):
                    raise ValueError('Invalid saved AMF array')
                if state['sampling_index']==0:
                    pure.append(z['soft'].copy())
    if not all(np.array_equal(pure[0],p) for p in pure[1:]):
        raise ValueError('Pure-noise controls differ despite identical inputs')
    return dict(model=MODEL, forward_passes=20, rows=len(rows), pure_noise_equal=True,
                note='Structural audit passed; read motion errors before deciding on a correction.')


def audit_response(root):
    import numpy as np
    from omegaconf import OmegaConf
    from probe_report import load_trace
    root = Path(root)
    meta = json.loads((root/'metadata.json').read_text())
    config = OmegaConf.to_container(OmegaConf.load(root/'suite_config.yaml'))
    expected_config = dict(model_key=MODEL, enable_model_cpu_offload=True, guidance_blocks=[10],
        injection_blocks=[], guidance_mode='latent', loss_type='flow', flow_loss='mse',
        flow_max_disp=100., motion_temp=2., scheduler='flowmatch', flow_shift=3., guidance_scale=5.,
        source_prompt='', seed=1, num_frames=21, height=480, width=832, threshloss=True,
        flow_min_conf=None, flow_region_masks=None, softmax_fp32=True, argmax_motion_flow=True)
    if any(config.get(k)!=v for k,v in expected_config.items()) or meta['model']!=MODEL:
        raise ValueError('Changed response configuration')
    step, updates = meta['branch_step'], meta['updates']
    if not 0<=step<10 or not 1<=updates<=5 or meta['scheduler_class']!='FlowMatchEulerDiscreteScheduler':
        raise ValueError('Unexpected intervention or scheduler')
    if [p['step'] for p in meta['prefix']]!=list(range(step)):
        raise ValueError('Missing unguided prefix')
    if len(meta['timesteps'])!=50 or len(meta['sigmas'])!=51:
        raise ValueError('Wrong schedule length')
    if meta['timestep']!=meta['timesteps'][step] or meta['sigma']!=meta['sigmas'][step]:
        raise ValueError('Intervention timestep/sigma mismatch')
    expected_lr = timing.learning_rates(range(10),[.002,.001],10)[step]
    if not math.isclose(meta['learning_rate'],expected_lr):
        raise ValueError('Wrong intervention learning rate')
    starts, reports, before_fields = [], [], []
    targets = {}
    for name in ('forward','reverse'):
        with np.load(root/f'target_{name}.npz') as z:
            targets[name] = (z['flow'].copy(),z['mask'].copy())
    if np.array_equal(targets['forward'][0],targets['reverse'][0]):
        raise ValueError('Direction control has no distinct target')
    for branch in ('off','forward','reverse'):
        folder = root/branch
        report = json.loads((folder/'response.json').read_text())
        path, recorded, events = load_trace(folder)
        if any(recorded['config'].get(k)!=v for k,v in expected_config.items()):
            raise ValueError(f'{branch}: changed branch configuration')
        opt = [e for e in events if e['kind']=='optimization']
        if [(e['step'],e['iteration']) for e in opt] != ([] if branch=='off' else [(step,i) for i in range(updates)]):
            raise ValueError(f'{branch}: incorrect actual optimization sequence')
        for e in opt:
            if (not math.isfinite(e['loss_before_update']) or not math.isclose(e['lr'],expected_lr)
                    or any(e[k].get('finite_fraction')!=1 or not (e[k].get('rms',0)>0) for k in ('gradient','update'))):
                raise ValueError(f'{branch}: nonfinite or ineffective optimizer update')
        sampled = [e for e in events if e['kind']=='sampling']
        if [e['step'] for e in sampled]!=list(range(step,50)) or report['sampled_indices']!=list(range(step,50)):
            raise ValueError(f'{branch}: missing/duplicated denoising steps')
        for e in sampled:
            if (e['sigma']!=meta['sigmas'][e['step']] or e['timestep']!=meta['timesteps'][e['step']]
                    or e['latent']['finite_fraction']!=1):
                raise ValueError(f'{branch}: invalid denoising schedule or latent')
            expected_nonzero = branch!='off' and e['step']==step
            if (e['guidance_update']['rms']>0)!=expected_nonzero:
                raise ValueError(f'{branch}: extra or missing intervention')
        if any(e.get('injected',False) for e in events):
            raise ValueError('KV injection occurred')
        refs = [e for e in events if e['kind']=='training_reference']
        name = 'reverse' if branch=='reverse' else 'forward'
        if len(refs)!=1:
            raise ValueError('Missing/duplicated loss target')
        with np.load(path/refs[0]['file']) as z:
            if not all(np.array_equal(z[k],v) for k,v in zip(('flow','mask'),targets[name])):
                raise ValueError('Actual loss target differs from saved direction control')
        fields = []
        recomputed = []
        common = targets['forward'][1] & targets['reverse'][1]
        if not common.any():
            raise ValueError('No shared reference support')
        for stage in ('before_update','after_update','final_latent'):
            with np.load(folder/f'{stage}.npz') as z:
                field = z['flow'].copy()
            if field.shape!=targets['forward'][0].shape or not np.isfinite(field).all():
                raise ValueError('Invalid response field')
            fields.append(field)
            native, shared = {}, {}
            for target_name, (truth, valid) in targets.items():
                errors = ((field-truth)**2).mean(-1)
                native[target_name] = float(errors[valid].mean())
                shared[target_name] = float(errors[common].mean())
            recomputed.append(dict(stage=stage,native_mask_mse=native,common_mask_mse=shared,
                forward_preference=shared['reverse']-shared['forward'],common_positions=int(common.sum())))
        if len(report['evaluations'])!=3:
            raise ValueError('Missing before/after/final evaluations')
        for actual, saved in zip(recomputed,report['evaluations']):
            if actual['stage']!=saved['stage'] or actual['common_positions']!=saved['common_positions']:
                raise ValueError('Incorrect response evaluation stages/support')
            for metric in ('native_mask_mse','common_mask_mse'):
                if any(not np.isclose(actual[metric][k],saved[metric][k],rtol=1e-5,atol=1e-5) for k in targets):
                    raise ValueError('Recorded response scores disagree with saved fields')
            if not np.isclose(actual['forward_preference'],saved['forward_preference'],rtol=1e-5,atol=1e-4):
                raise ValueError('Incorrect target-preference score')
        before_fields.append(fields[0])
        if branch=='off' and not np.array_equal(fields[0],fields[1]):
            raise ValueError('AMF changed without an update at the same latent/timestep')
        if not report['rope_unchanged'] or not (folder/'results.mp4').is_file():
            raise ValueError('Missing video or modified RoPE')
        starts.append(report['start_latent_sha256'])
        reports.append(dict(branch=branch, updates=len(opt), evaluations=recomputed))
    if any(s!=meta['shared_latent_sha256'] for s in starts) or not all(np.array_equal(before_fields[0],x) for x in before_fields[1:]):
        raise ValueError('Branches did not start from the same latent and motion readout')
    return dict(model=MODEL, branch_step=step, same_start=True, branches=reports,
                note='Impulse response validated, not a motion-quality pass. Inspect the three decoded videos.')


def run_stage(root, stage):
    if stage not in ('readout','response'):
        raise ValueError(stage)
    root = Path(root).resolve()
    plan = json.loads((root/'plan.json').read_text())
    expected = json.loads((root/'environment.json').read_text())
    timing.require_same_environment(expected, environment_snapshot())
    direction.prepare_inputs(plan['inputs'])
    if stage=='response':
        readout_marker = json.loads((root/'readout_done.json').read_text())
        audit_readout(root/readout_marker['directory'])
    marker = root/f'{stage}_done.json'
    validator = audit_readout if stage=='readout' else audit_response
    if marker.is_file():
        directory = root/json.loads(marker.read_text())['directory']
        validator(directory)
        print('Already complete and revalidated:',directory)
        return directory
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = root/f'{stage}_{stamp}'
    command = [*plan[f'{stage}_command'],'--output_path',str(directory)]
    with (root/f'{stage}_{stamp}.log').open('w',encoding='utf-8') as log:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                   encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            if process.wait()!=0:
                raise RuntimeError(f'{stage} failed; inspect {log.name}. Rerun this cell for a fresh stage attempt.')
        except BaseException:
            process.terminate(); process.wait()
            raise
    timing.require_same_environment(expected,environment_snapshot())
    audit = validator(directory)
    marker.write_text(json.dumps(dict(directory=directory.name,audit=audit),indent=2),encoding='utf-8')
    return directory


def archive_experiment(root):
    root = Path(root).resolve()
    archive = root.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob('*')):
            if path.is_file() and 'embeds' not in path.relative_to(root).parts and path.suffix not in ('.pt','.pth'):
                bundle.write(path,Path(root.name)/path.relative_to(root))
    return archive
