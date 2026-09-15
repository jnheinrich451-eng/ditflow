"""Six readout forwards on saved RGB controls; no optimization or generation.

This diagnoses the block-20 target readout, with positive prompt conditioning
held fixed across clean/step-9 inputs. It does not change the AMF objective.
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from guidance_utils.wan_reference_diagnostics import reference_images, image_motion, comparison

STEP39_CRITERIA = dict(min_patches_per_pair=20, relative_cosine_min=.8,
                       relative_amplitude_range=[.5,1.5], static_epe_max=1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_plan(prepared):
    prepared=Path(prepared)
    plan=json.loads((prepared/'plan.json').read_text())
    expected_states = ['step_39'] if plan.get('protocol')=='step39' else ['clean','step_09']
    if (plan['controls']!=['forward','reverse','static'] or plan['states']!=expected_states
            or plan['noise_seed']!=29):
        raise ValueError('The fixed readout protocol changed; no additional controls/timings are allowed')
    if plan.get('protocol')=='step39' and plan.get('criteria')!=STEP39_CRITERIA:
        raise ValueError('Step-39 readout criteria changed')
    for name,expected in plan['input_sha256'].items():
        actual=digest(prepared/name)
        if actual!=expected:
            raise ValueError(f'Input changed: {name}; expected={expected}, actual={actual}')
    baseline_meta=prepared/'baseline/metadata.json'
    if plan.get('protocol')=='step39' and baseline_meta.is_file():
        core=json.loads(baseline_meta.read_text()).get('source_sha256',{})
        root=Path(__file__).resolve().parents[1]
        for name,expected in core.items():
            if name=='motion_guidance_wan.py' or name.startswith('guidance_utils/'):
                raw=(root/name).read_bytes()
                actual=hashlib.sha256(raw).hexdigest()
                normalized=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()
                if expected not in (actual,normalized):
                    raise ValueError(f'Core source changed beyond timing: {name}; expected={expected}, actual={actual}, normalized={normalized}')
    return plan


def controls(frames):
    return dict(forward=frames, reverse=frames[::-1], static=[frames[len(frames)//2]]*len(frames))


def prepare(saved_run, output):
    """CPU-only evidence preparation with explicit, reviewable RGB regions."""
    import cv2
    saved_run, output = Path(saved_run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Keep earlier diagnostics; use a fresh directory: {output}')
    meta = json.loads((saved_run/'manifest.json').read_text())
    if meta['token_grid'] != [6,30,52] or meta['config']['guidance_blocks'] != [20]:
        raise ValueError('This bounded protocol is for the saved 21-frame block-20 camel case')
    if not json.loads((saved_run/'parity.json').read_text())['passed']:
        raise ValueError('Resolve saved native parity before this diagnostic')
    expected_video='85b9184855ab21d738a78720600119c167fc93f59a9fa3d3c5bc2cc66b369f3b'
    actual_video=digest(saved_run/'off/final.mp4')
    if actual_video!=expected_video:
        raise ValueError(f'RGB regions were reviewed for a different video: expected={expected_video}, actual={actual_video}')
    output.mkdir(parents=True)
    frames = reference_images(saved_run/'off/final.mp4', 21, (832,480))
    # Conservative interior torso and left-fence ROIs in the reviewed off video.
    # These are manual RGB regions, not segmentation or latent-location proxies.
    boxes = dict(subject=[350,140,590,260], background=[0,200,140,350])
    yy, xx = np.meshgrid((np.arange(30)+.5)*16, (np.arange(52)+.5)*16, indexing='ij')
    regions = {name:(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]) for name,b in boxes.items()}
    sheet = Image.new('RGB',(416*5,266),'white')
    for col,index in enumerate((0,5,10,15,20)):
        panel = Image.fromarray(frames[index]); draw = ImageDraw.Draw(panel)
        for name,box in boxes.items(): draw.rectangle(box,outline='red' if name=='subject' else 'cyan',width=3)
        sheet.paste(panel.resize((416,240)),(416*col,26))
        ImageDraw.Draw(sheet).text((416*col+5,5),f'Off frame {index}: red torso, cyan fence',fill='black')
    sheet.save(output/'rgb_regions.jpg')
    arrays, rows = {}, []
    for name, sequence in controls(frames).items():
        folder = output/'controls'/name; folder.mkdir(parents=True)
        for i,frame in enumerate(sequence): Image.fromarray(frame).save(folder/f'{i:05d}.png')
        fields, validities = [], []
        for pair in range(5):
            flow, valid = image_motion(sequence[4*pair],sequence[4*(pair+1)],(30,52))
            fields.append(flow); validities.append(valid)
            for region, mask in regions.items():
                chosen = valid & mask
                rows.append(dict(control=name,pair=pair,region=region,patches=int(chosen.sum()),
                    rgb_dx=float(np.median(flow[chosen,0])) if chosen.any() else None,
                    rgb_dy=float(np.median(flow[chosen,1])) if chosen.any() else None))
        arrays[name+'_flow'], arrays[name+'_valid'] = np.asarray(fields), np.asarray(validities)
    arrays.update({name+'_region':mask for name,mask in regions.items()})
    np.savez_compressed(output/'rgb_motion.npz',**arrays)
    plan = dict(schema=1,saved_run=str(saved_run),saved_manifest_sha256=digest(saved_run/'manifest.json'),
        image_analysis_versions=dict(opencv=cv2.__version__,numpy=np.__version__),
        off_video_sha256=actual_video,grid=[6,30,52],block=20,noise_seed=29,
        checkpoint_revision=meta['checkpoint_revision'],conditioning_sha256=meta['conditioning_sha256'],
        sigmas=meta['sigmas'],timesteps=meta['timesteps'],
        states=['clean','step_09'],controls=['forward','reverse','static'],max_transformer_forwards=6,
        conditioning='positive target prompt for every readout; no CFG or source-prompt change',
        roi_method='Manual conservative torso and fence rectangles; inspect rgb_regions.jpg before GPU capture.',
        boxes=boxes,first_pair_caveat='Original frame 0 is corrupted: flag forward pair 0 and reverse pair 4; static uses frame 10.',
        rgb_metric='Farneback with forward/backward consistency and texture checks; estimate, not ground truth.',
        rgb_rows=rows)
    plan['input_sha256']={str(p.relative_to(output)):digest(p) for p in (output/'controls').rglob('*.png')}
    plan['input_sha256']['rgb_motion.npz']=digest(output/'rgb_motion.npz')
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    return output


def prepare_step39(previous, output):
    """Reuse verified RGB controls/measurements; old acceptance folder not needed."""
    previous, output = Path(previous).resolve(), Path(output).resolve()
    old = checked_plan(previous)
    if old['states']!=['clean','step_09']:
        raise ValueError('Use the completed clean/step-9 correspondence run as baseline')
    metadata=json.loads((previous/'readouts/metadata.json').read_text())
    expected=[(c,s) for c in old['controls'] for s in old['states']]
    if [(e['control'],e['state']) for e in metadata['events']]!=expected:
        raise ValueError('Baseline is incomplete')
    if (metadata['conditioning_sha256']!=old['conditioning_sha256'] or metadata['sigmas']!=old['sigmas']
            or metadata['timesteps']!=old['timesteps'] or Path(metadata['checkpoint']).name!=old['checkpoint_revision']):
        raise ValueError('Baseline checkpoint, conditioning, or schedule mismatch')
    if output.exists(): raise FileExistsError(f'Use a fresh diagnostic directory: {output}')
    output.mkdir(parents=True)
    for name in [*old['input_sha256'], 'rgb_regions.jpg']:
        dest=output/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(previous/name,dest)
    baseline=output/'baseline'; baseline.mkdir()
    for name in ('plan.json','correspondence_report.json','source_revision.json','readouts/metadata.json'):
        shutil.copyfile(previous/name,baseline/Path(name).name)
    plan=copy.deepcopy(old)
    plan.update(protocol='step39',states=['step_39'],max_transformer_forwards=3,
        criteria=copy.deepcopy(STEP39_CRITERIA),
        expected_noise_sha256=metadata['noise_sha256'],previous_plan_sha256=digest(previous/'plan.json'),
        target_prompt='A camel walking across a dusty paddock beside a metal fence.',
        model='Wan-AI/Wan2.1-T2V-14B-Diffusers')
    for name in ('saved_run','saved_manifest_sha256'): plan.pop(name,None)
    plan['input_sha256'].update({str(p.relative_to(output)):digest(p) for p in baseline.iterdir()})
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    return output


def load_model(prepared):
    """Reuse saved config/checkpoint, omitting unused reference setup forwards."""
    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from motion_guidance_wan import WanGuidance
    from probe_wan_response import tensor_hash
    plan = checked_plan(prepared)
    if (Path(prepared)/'readouts').exists():
        raise RuntimeError('Readout budget already started; reuse saved captures without loading a model')
    if plan.get('protocol')=='step39':
        from benchmark.wan_port_acceptance import acceptance_config
        meta=plan
        config=acceptance_config('',Path(prepared)/'controls/forward',Path(prepared)/'model_setup',plan['target_prompt'])
    else:
        saved = Path(plan['saved_run'])
        if digest(saved/'manifest.json') != plan['saved_manifest_sha256']:
            raise ValueError('Saved manifest changed')
        meta = json.loads((saved/'manifest.json').read_text())
        config = OmegaConf.create(copy.deepcopy(meta['config']))
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 35*2**30:
        raise RuntimeError('Use A100 40 GB/high RAM or 80 GB')
    budget=len(plan['controls'])*len(plan['states'])
    print(f'SETUP: one pinned model load; zero transformer forwards. Then at most {budget} readout forwards.',flush=True)
    config.model_key = snapshot_download(meta['model'],revision=meta['checkpoint_revision'])
    config.output_path = str(Path(prepared)/'model_setup')
    config.video_path = str(Path(prepared)/'controls/forward')
    config.enable_model_cpu_offload = torch.cuda.get_device_properties(0).total_memory < 70*2**30
    class ReadoutGuidance(WanGuidance):
        def load_latent(self):
            # Constructor placeholder only. capture() explicitly calls the base
            # VAE loader for all three RGB controls before their readouts.
            return torch.zeros_like(self.init_latents, dtype=self.dtype)
        def load_attn_features(self):
            return {}
    g = ReadoutGuidance(config)
    observed = tensor_hash(g.guidance_embeds)
    expected = meta['conditioning_sha256']
    if observed != expected:
        raise ValueError(f'Conditioning changed: expected={expected}, actual={observed}')
    if g.scheduler.sigmas.tolist() != meta['sigmas'] or g.timesteps.cpu().tolist() != meta['timesteps']:
        raise ValueError('Saved scheduler schedule differs from loaded diagnostic schedule')
    return g


def capture(g, prepared):
    import torch
    from importlib.metadata import version
    from motion_guidance_wan import WanGuidance
    from guidance_utils.wan_motion_flow_utils import compute_motion_flow
    from probe_wan_response import tensor_hash
    prepared = Path(prepared).resolve()
    plan = checked_plan(prepared)
    result = prepared/'readouts'
    if result.exists():
        raise RuntimeError('Readout budget already started; inspect its captures instead of rerunning')
    if (list(g.config.guidance_blocks)!=[plan['block']] or list(g.config.injection_blocks)
            or g.config.motion_temp!=2 or g.config.get('flow_head') is not None):
        raise ValueError('Keep block 20, all-head mean logits, temperature 2, and no injection')
    actual_conditioning=tensor_hash(g.guidance_embeds)
    if actual_conditioning!=plan['conditioning_sha256']:
        raise ValueError(f"Conditioning changed: expected={plan['conditioning_sha256']}, actual={actual_conditioning}")
    if Path(g.config.model_key).name!=plan['checkpoint_revision']:
        raise ValueError(f"Checkpoint changed: expected={plan['checkpoint_revision']}, actual={g.config.model_key}")
    if g.scheduler.sigmas.tolist()!=plan['sigmas'] or g.timesteps.cpu().tolist()!=plan['timesteps']:
        raise ValueError('Diagnostic schedule changed')
    budget=len(plan['controls'])*len(plan['states'])
    hypothesis=('Lower noise at step 39 restores useful block-20 correspondence.' if plan.get('protocol')=='step39'
                else 'Block-20 target AMF loses known subject direction at early noise.')
    print(f'HYPOTHESIS: {hypothesis} '
          'EXPECTED: opposite relative-motion signs and substantially reduced static error. '
          f'LIMIT: {budget} truncated transformer forwards, 3 VAE encodes, zero optimization/generation.',flush=True)
    old_path, old_output = g.config.video_path, g.output_path
    old_rope = g.transformer.trainable_rope
    noise = torch.randn(g.init_latents.shape,generator=torch.Generator(device=g.device).manual_seed(plan['noise_seed']),
                        device=g.device,dtype=torch.float32)
    if plan.get('expected_noise_sha256') and tensor_hash(noise)!=plan['expected_noise_sha256']:
        raise ValueError(f"Noise mismatch: expected={plan['expected_noise_sha256']}, actual={tensor_hash(noise)}")
    result.mkdir()
    g.transformer.trainable_rope = None
    events = []
    source_root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__),source_root/'motion_guidance_wan.py',*sorted((source_root/'guidance_utils').glob('*.py'))]
    metadata = dict(noise_sha256=tensor_hash(noise),noise_seed=plan['noise_seed'],
        packages={name:version(name) for name in ('torch','diffusers','transformers','huggingface-hub','ftfy','accelerate')},
        conditioning_sha256=tensor_hash(g.guidance_embeds),checkpoint=g.config.model_key,
        sigmas=g.scheduler.sigmas.tolist(),timesteps=g.timesteps.cpu().tolist(),
        source_sha256={str(p.relative_to(source_root)):digest(p) for p in sources},events=events)
    try:
        with torch.no_grad():
            for control in plan['controls']:
                g.config.video_path=str(prepared/'controls'/control)
                g.output_path=str(result/control); Path(g.output_path).mkdir()
                latent=WanGuidance.load_latent(g).float()
                for label in plan['states']:
                    index={'clean':None,'step_09':9,'step_39':39}[label]
                    sigma=0. if index is None else float(g.scheduler.sigmas[index])
                    time=torch.zeros_like(g.timesteps[:1]) if index is None else g.timesteps[index:index+1]
                    x=(1-sigma)*latent+sigma*noise
                    g._set_kv_mode(g.config.guidance_blocks,inject=False,copy=True)
                    try:
                        g._forward_transformer(x,g.guidance_embeds[1:2],time)
                        proc=g.transformer.blocks[plan['block']].attn1.processor
                        args=dict(h=g.patches_height,w=g.patches_width,nframes=g.latent_num_frames,
                                  temp=2.,softmax_fp32=True,head_dim=g.transformer.config.attention_head_dim)
                        hard, confidence=compute_motion_flow(proc.query,proc.key,**args,argmax=True,return_confidence=True)
                        soft=compute_motion_flow(proc.query,proc.key,**args,argmax=False)
                        # Retain post-RoPE Q/K once per readout for CPU re-analysis;
                        # FP32 represents the original BF16 values exactly.
                        np.savez_compressed(result/f'{control}_{label}.npz',
                            hard=hard.cpu().numpy(),soft=soft.cpu().numpy(),confidence=confidence.cpu().numpy(),
                            query=proc.query.float().cpu().numpy(),key=proc.key.float().cpu().numpy())
                        events.append(dict(control=control,state=label,sigma=sigma,timestep=float(time[0]),
                                           input_sha256=tensor_hash(x),qk_shape=list(proc.query.shape)))
                        (result/'metadata.json').write_text(json.dumps(metadata,indent=2))
                    finally:
                        g._set_kv_mode(g.config.guidance_blocks,inject=False,copy=False)
                        g._clear_kv(g.config.guidance_blocks)
    finally:
        g.config.video_path, g.output_path = old_path, old_output
        g.transformer.trainable_rope=old_rope
    if len(events)!=budget: raise RuntimeError('Incomplete readout budget')
    return analyze(prepared)


def analyze(prepared):
    prepared=Path(prepared)
    plan=json.loads((prepared/'plan.json').read_text())
    f,h,w=plan['grid']; pair_indices=np.arange(f-1)*(f+1)+1
    rows=[]
    with np.load(prepared/'rgb_motion.npz') as rgb:
        for control in plan['controls']:
            for state in plan['states']:
                with np.load(prepared/'readouts'/f'{control}_{state}.npz') as data:
                    for field in ('hard','soft'):
                        amf=data[field][pair_indices].reshape(f-1,h,w,2)
                        for pair in range(f-1):
                            measured,valid=rgb[control+'_flow'][pair],rgb[control+'_valid'][pair]
                            bg=valid&rgb['background_region']
                            for region in ('subject','background','subject_relative_to_background'):
                                chosen=valid&rgb[('subject' if region.startswith('subject') else 'background')+'_region']
                                a,b=amf[pair],measured
                                if region=='subject_relative_to_background':
                                    if not bg.any(): chosen=np.zeros_like(chosen)
                                    else: a,b=a-np.median(a[bg],axis=0),b-np.median(b[bg],axis=0)
                                item=comparison(a,b,chosen)
                                item['epe']=float(np.linalg.norm(a[chosen]-b[chosen],axis=-1).mean()) if chosen.any() else None
                                rows.append(dict(control=control,state=state,field=field,pair=pair,
                                                 region=region,includes_corrupt_endpoint=(control=='forward' and pair==0) or
                                                 (control=='reverse' and pair==f-2),**item))
    report=dict(status='Readout diagnostic only; decoded motion transfer remains unproven.',
                caveats=[plan['roi_method'],plan['rgb_metric'],plan['first_pair_caveat']],rows=rows)
    (prepared/'correspondence_report.json').write_text(json.dumps(report,indent=2))
    if plan.get('protocol')=='step39':
        comparison_report=timing_report(prepared)
        print('Step-39 readout screen:',comparison_report['ready_for_decoded_comparison'])
        for reason in comparison_report['failures']: print(' ',reason)
    print('Saved correspondence_report.json; inspect per-pair support and ROI overlay before interpreting signs.')
    return report


def timing_report(prepared):
    prepared=Path(prepared)
    plan=json.loads((prepared/'plan.json').read_text())
    previous=json.loads((prepared/'baseline/correspondence_report.json').read_text())
    current=json.loads((prepared/'correspondence_report.json').read_text())
    rows=previous['rows']+current['rows']
    summary=[]; failures=[]
    for control in plan['controls']:
        for state in ('clean','step_09','step_39'):
            for region in ('subject','background','subject_relative_to_background'):
                chosen=[r for r in rows if r['control']==control and r['state']==state and r['field']=='soft'
                        and r['region']==region and not r['includes_corrupt_endpoint']]
                values=dict(control=control,state=state,region=region,pairs=len(chosen))
                for key in ('amf_dx','image_dx','cosine','epe'):
                    numbers=[r[key] for r in chosen if r[key] is not None]
                    values[key]=float(np.mean(numbers)) if numbers else None
                summary.append(values)
                if state!='step_39': continue
                expected_pairs=plan['grid'][0]-1-(control!='static')
                if len(chosen)!=expected_pairs or any(r['patches']<20 for r in chosen):
                    failures.append(f'{control}/{region}: insufficient pair or image-flow support')
                if values['epe'] is None or not np.isfinite(values['epe']):
                    failures.append(f'{control}/{region}: invalid endpoint error')
                if control=='static':
                    if values['epe'] is None or values['epe']>STEP39_CRITERIA['static_epe_max']:
                        failures.append(f'static/{region}: mean error exceeds one patch')
                elif region=='subject_relative_to_background':
                    if not chosen or any(r['amf_dx'] is None or r['image_dx'] is None or
                                         not np.isfinite(r['amf_dx']*r['image_dx']) or
                                         r['amf_dx']*r['image_dx']<=0 for r in chosen):
                        failures.append(f'{control}: relative-motion sign incorrect in one or more pairs')
                    if values['cosine'] is None or not np.isfinite(values['cosine']) or values['cosine']<STEP39_CRITERIA['relative_cosine_min']:
                        failures.append(f'{control}: mean relative-motion cosine below 0.8')
                    gain=abs(values['amf_dx']/values['image_dx']) if values['image_dx'] and values['amf_dx'] is not None else 0.
                    if not .5<=gain<=1.5:
                        failures.append(f'{control}: relative horizontal amplitude outside [0.5,1.5] of RGB estimate')
    result=dict(ready_for_decoded_comparison=not failures,port_success=False,failures=failures,
                criteria=STEP39_CRITERIA,summary=summary,
                meaning='Readout screening only. Passing does not demonstrate guided video motion transfer.')
    (prepared/'timing_comparison.json').write_text(json.dumps(result,indent=2))
    return result
