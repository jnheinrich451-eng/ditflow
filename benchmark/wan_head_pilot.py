"""Plan, resume and screen the fixed-temperature 14B head experiment."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark import wan_direction_pilot as direction
from benchmark import wan_timing_pilot as timing
from benchmark.wan_response_pilot import archive_experiment
from guidance_utils.wan_head_diagnostics import BLOCKS, HEADS, audit_heads, make_head_report
from guidance_utils.wan_amf_calibration import CONTROLS, MODEL, THRESHOLDS


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def environment_snapshot():
    snapshot = direction.environment_snapshot()
    for name in ('probe_wan_affine.py','benchmark/wan_head_pilot.py',
                 'benchmark/wan_head_visual_pilot.py','probe_wan_head_visual.py','probe_wan_response.py',
                 'benchmark/wan_head_crossover_pilot.py','benchmark/wan_noised_reference_pilot.py',
                 'probe_wan_noised_reference.py','benchmark/wan_pair_pilot.py','probe_wan_pairs.py'):
        snapshot['source_sha256'][name] = digest(name)
    return snapshot


def save_plan(root, plan):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh setup directory; restore the existing output path to resume run')
    environment = environment_snapshot()
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'plan.json',plan)
    (root/'plan.sha256').write_text(digest(root/'plan.json'),encoding='utf-8')
    write_json(root/'environment.json',environment)


def load_plan(root):
    root = Path(root)
    if digest(root/'plan.json') != (root/'plan.sha256').read_text(encoding='utf-8'):
        raise ValueError('Plan changed after setup; create a fresh experiment')
    return json.loads((root/'plan.json').read_text(encoding='utf-8'))


def command_for(video, blocks, heads, noise_seed):
    command = [sys.executable,'-u','probe_wan_affine.py','-v',str(video),'--model','14b','--low_vram',
        '--blocks',*map(str,blocks),'--controls',*CONTROLS,'--noise_steps','0','9','29','--noise_seed',str(noise_seed)]
    if heads is not None: command += ['--readout_heads',*map(str,heads)]
    return command


def validate_result(directory, plan):
    from omegaconf import OmegaConf
    config = OmegaConf.load(Path(directory)/'suite_config.yaml')
    if str(config.video_path) != plan['video']:
        raise ValueError('Head suite used a different input path')
    return audit_heads(directory,plan['blocks'],plan['heads'],plan['noise_seed'])


def summarize(root):
    root = Path(root).resolve(); plan = load_plan(root)
    directory = root/json.loads((root/'readout_done.json').read_text(encoding='utf-8'))['directory']
    audit = validate_result(directory,plan)
    report = make_head_report(directory)
    candidate = plan['fixed_candidate']
    report['fixed_candidate_confirmation'] = next((c for c in report['candidates'] if candidate and
        c['block']==candidate['block'] and c['variant']==candidate['variant']),None)
    write_json(root/'screening.json',report)
    write_json(root/'audit.json',audit)
    return directory,report


def candidate_receipt(root, candidate):
    directory, report = summarize(root)
    block,head = candidate
    matches = [c for c in report['candidates'] if c['block']==f'block_{block}_attn1_processor' and c['variant']==f'head_{head:02d}']
    if len(matches)!=1 or not matches[0]['screen_pass']:
        raise ValueError('The fixed head must pass every predeclared condition; no best-failing-head promotion')
    return dict(candidate=matches[0],plan_sha256=digest(Path(root)/'plan.json'),
        metadata_sha256=digest(directory/'metadata.json'),metrics_sha256=digest(directory/'metrics.json'))


def make_plan(inputs, root, archive=None, *, development=None, candidate=None):
    confirmation = development is not None
    receipt = None
    if confirmation:
        if (candidate is None or len(candidate)!=2 or type(candidate[0]) is not int or type(candidate[1]) is not int
                or candidate[0] not in BLOCKS or candidate[1] not in HEADS):
            raise ValueError('Freeze a passing development (block, head) before confirmation')
        dev_plan = load_plan(development)
        if dev_plan['stage']!='development': raise ValueError('Selection must use development, not confirmation')
        timing.require_same_environment(json.loads((Path(development)/'environment.json').read_text(encoding='utf-8')),
                                        environment_snapshot())
        receipt = candidate_receipt(development,candidate)
    elif candidate is not None:
        raise ValueError('Development tests all heads; selecting a head requires completed development')
    inputs, rows = direction.prepare_inputs(inputs,archive)
    clip,seed = ('camel',29) if confirmation else ('car-turn',17)
    row = next(r for r in rows if r['clip_id']==clip)
    blocks,heads = ([candidate[0]],[candidate[1]]) if confirmation else (BLOCKS,None)
    video = str(inputs/row['video_path'])
    plan = dict(schema_version=1,stage='confirmation' if confirmation else 'development',model=MODEL,
        inputs=str(inputs),video=video,clip_id=clip,noise_seed=seed,blocks=blocks,heads=heads,temperature=2.,
        thresholds=THRESHOLDS,forward_passes=20,expected_rows=20*len(blocks)*(2 if confirmation else 41)*12,
        fixed_candidate=receipt['candidate'] if receipt else None,development_receipt=receipt,
        command=command_for(video,blocks,heads,seed))
    save_plan(root,plan)
    (Path(root)/'input_manifest.csv').write_bytes((inputs/'manifest.csv').read_bytes())
    return plan


def checked_inputs(plan):
    inputs, rows = direction.prepare_inputs(plan['inputs'])
    row = next(r for r in rows if r['clip_id']==plan['clip_id'])
    if str(inputs/row['video_path']) != plan['video']:
        raise ValueError('Input manifest differs from plan')
    return inputs,rows


def run_process(root, label, command, validator):
    """Fresh attempts; success markers written only after exit and a trace audit."""
    root = Path(root).resolve()
    expected = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    timing.require_same_environment(expected,environment_snapshot())
    marker = root/f'{label}_done.json'
    if marker.is_file():
        directory = root/json.loads(marker.read_text(encoding='utf-8'))['directory']
        validator(directory)
        print('Already complete and revalidated:',directory)
        return directory
    attempt = f'{label}_{stamp()}'; directory = root/attempt
    with (root/f'{attempt}.log').open('w',encoding='utf-8') as log:
        process = subprocess.Popen([*command,'--output_path',str(directory)],stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            if process.wait()!=0: raise RuntimeError(f'Failed attempt retained at {log.name}; rerun for a fresh attempt')
        except BaseException:
            if process.poll() is None: process.terminate(); process.wait()
            raise
    timing.require_same_environment(expected,environment_snapshot())
    audit = validator(directory)
    write_json(marker,dict(directory=directory.name,audit=audit))
    return directory


def run(root):
    plan = load_plan(root)
    checked_inputs(plan)
    expected = command_for(plan['video'],plan['blocks'],plan['heads'],plan['noise_seed'])
    if plan['command'] != expected or plan['thresholds'] != THRESHOLDS:
        raise ValueError('Head protocol changed; make a new plan')
    return run_process(root,'readout',expected,lambda directory: validate_result(directory,plan))


def confirmed_candidate(development, confirmation):
    """Reaudit both stages; enforce the identity frozen before camel was evaluated."""
    dev,conf = load_plan(development),load_plan(confirmation)
    if dev['stage']!='development' or conf['stage']!='confirmation':
        raise ValueError('Visual validation requires development followed by confirmation')
    candidate = (conf['blocks'][0],conf['heads'][0])
    receipt = candidate_receipt(development,candidate)
    if receipt != conf['development_receipt']:
        raise ValueError('Development evidence or frozen candidate changed')
    confirmation_receipt = candidate_receipt(confirmation,candidate)
    dev_meta = json.loads((summarize(development)[0]/'metadata.json').read_text(encoding='utf-8'))
    conf_meta = json.loads((summarize(confirmation)[0]/'metadata.json').read_text(encoding='utf-8'))
    if dev_meta['input_base_sha256']==conf_meta['input_base_sha256'] or dev_meta['noise_sha256']==conf_meta['noise_sha256']:
        raise ValueError('Confirmation must use a different texture and noise tensor')
    for key in ('model','packages','model_dtype','source_sha256','model_revision'):
        if dev_meta[key] != conf_meta[key]: raise ValueError(f'Confirmation changed {key}')
    return dict(block=candidate[0],head=candidate[1],development=receipt,confirmation=confirmation_receipt)
