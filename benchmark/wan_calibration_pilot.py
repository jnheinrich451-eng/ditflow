"""Plan, resume and audit the 14B AMF sharpening diagnostic."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark import wan_direction_pilot as direction
from benchmark import wan_timing_pilot as timing
from benchmark.wan_response_pilot import archive_experiment
from guidance_utils.wan_amf_calibration import (
    BLOCKS, CONTROLS, MODEL, TEMPERATURES, THRESHOLDS, audit_calibration, screen_candidates,
)


def environment_snapshot():
    snapshot = direction.environment_snapshot()
    for name in ('probe_wan_affine.py', 'guidance_utils/wan_amf_calibration.py', 'benchmark/wan_calibration_pilot.py'):
        snapshot['source_sha256'][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    return snapshot


def command_for(video, noise_seed):
    return [sys.executable, 'probe_wan_affine.py', '-v', str(video), '--model', '14b', '--low_vram',
            '--mean_only', '--blocks', *map(str,BLOCKS), '--controls', *CONTROLS,
            '--noise_steps', '0', '9', '29', '--noise_seed', str(noise_seed),
            '--readout_temperatures', *map(str,TEMPERATURES)]


def make_plan(inputs, root, archive=None, *, clip_id='car-turn', noise_seed=17, candidate=None):
    if clip_id not in ('car-turn','camel') or type(noise_seed) is not int or not 0 <= noise_seed < 2**32:
        raise ValueError('Choose car-turn/camel and a nonnegative 32-bit noise seed')
    if candidate is not None:
        if len(candidate) != 2 or candidate[0] not in BLOCKS or candidate[1] not in TEMPERATURES:
            raise ValueError('Candidate must be a tested (block, sharpening) pair')
        if clip_id == 'car-turn' and noise_seed == 17:
            raise ValueError('Confirmation requires a different texture or noise seed')
        candidate = dict(block=f'block_{int(candidate[0])}_attn1_processor', temperature=float(candidate[1]))
    inputs, rows = direction.prepare_inputs(inputs, archive)
    row = next(r for r in rows if r['clip_id'] == clip_id)
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh setup directory; restore CAL_OUT and rerun run to resume')
    root.mkdir(parents=True, exist_ok=True)
    video = str(inputs/row['video_path'])
    plan = dict(schema_version=1, model=MODEL, inputs=str(inputs), video=video, clip_id=clip_id,
        noise_seed=noise_seed, blocks=BLOCKS, temperatures=TEMPERATURES, thresholds=THRESHOLDS,
        fixed_candidate=candidate, forward_passes=20, expected_rows=2880, readout_only=True,
        command=command_for(video,noise_seed))
    (root/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (root/'environment.json').write_text(json.dumps(environment_snapshot(),indent=2),encoding='utf-8')
    (root/'input_manifest.csv').write_bytes((inputs/'manifest.csv').read_bytes())
    return plan


def validate_result(directory, plan):
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(Path(directory)/'suite_config.yaml'))
    if Path(config['video_path']).resolve() != Path(plan['video']).resolve():
        raise ValueError('Calibration used a different input path')
    return audit_calibration(directory, plan['blocks'], plan['temperatures'], plan['noise_seed'])


def run(root):
    root = Path(root).resolve()
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    if (plan['model'] != MODEL or plan['blocks'] != BLOCKS or plan['temperatures'] != TEMPERATURES
            or plan['thresholds'] != THRESHOLDS or plan['readout_only'] is not True
            or plan['command'] != command_for(plan['video'],plan['noise_seed'])):
        raise ValueError('Plan changed; create a fresh experiment')
    environment = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    timing.require_same_environment(environment, environment_snapshot())
    inputs, rows = direction.prepare_inputs(plan['inputs'])
    row = next(r for r in rows if r['clip_id'] == plan['clip_id'])
    if str(inputs/row['video_path']) != plan['video']:
        raise ValueError('Input manifest does not match planned clip')
    marker = root/'readout_done.json'
    if marker.is_file():
        directory = root/json.loads(marker.read_text(encoding='utf-8'))['directory']
        validate_result(directory,plan)
        print('Already complete and revalidated:',directory)
        return directory
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = root/f'readout_{stamp}'
    command = [*plan['command'], '--output_path', str(directory)]
    with (root/f'readout_{stamp}.log').open('w',encoding='utf-8') as log:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            if process.wait() != 0:
                raise RuntimeError(f'Readout failed; retained log: {log.name}. Rerun run for a fresh attempt.')
        except BaseException:
            if process.poll() is None:
                process.terminate(); process.wait()
            raise
    timing.require_same_environment(environment,environment_snapshot())
    audit = validate_result(directory,plan)
    marker.write_text(json.dumps(dict(directory=directory.name,audit=audit),indent=2),encoding='utf-8')
    return directory


def summarize(root):
    root = Path(root).resolve()
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    directory = root/json.loads((root/'readout_done.json').read_text(encoding='utf-8'))['directory']
    validate_result(directory,plan)
    rows = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    report = screen_candidates(rows)
    candidate = plan.get('fixed_candidate')
    report['fixed_candidate_confirmation'] = next((c for c in report['candidates'] if candidate and
        c['block'] == candidate['block'] and c['temperature'] == candidate['temperature']),None)
    (root/'screening.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return directory, report
