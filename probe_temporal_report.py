"""Audit video duration and temporal AMF coverage from saved runs; no model loading."""
import argparse
import csv
import html
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from probe_report import load_trace


def video_summary(path):
    meta = iio.immeta(path, plugin='FFMPEG')
    previous, changes, count = None, [], 0
    for frame in iio.imiter(path, plugin='FFMPEG'):
        if previous is not None:
            changes.append(float(np.abs(frame.astype(np.float32)-previous.astype(np.float32)).mean()))
        previous = frame
        count += 1
    fps = float(meta['fps'])
    return dict(file=Path(path).name, frames=count, fps=fps,
                nominal_duration_s=count/fps, container_duration_s=meta.get('duration'),
                last_frame_time_s=(count-1)/fps if count else None,
                pixel_change_per_transition=changes,
                exact_duplicate_transitions=sum(v==0 for v in changes))


def audit_run(run):
    run = Path(run)
    trace, meta, events = load_trace(run)
    f, h, w = meta['grid']
    result = dict(run=str(run.resolve()), grid=meta['grid'], videos=[], blocks={})
    for path in sorted(run.glob('*.mp4')):
        result['videos'].append(video_summary(path))
    for event in events:
        if event['kind'] != 'training_reference':
            continue
        block = event['block']
        with np.load(trace/event['file']) as data:
            flow, mask = data['flow'], data['mask'].astype(bool)
        if flow.shape != (f*f,h*w,2) or mask.shape != (f*f,h*w):
            raise ValueError(f'Unexpected all-pair reference shape: {flow.shape}, {mask.shape}')
        masks = mask.reshape(f,f,h*w)
        pair_count = masks.sum(-1)
        total = int(mask.sum())
        late = np.arange(f) >= (f+1)//2
        involves_late = late[:,None] | late[None,:]
        item = dict(reference_shape=list(flow.shape), retained_counts=pair_count.tolist(),
                    total_retained=total,
                    late_frame_indices=np.flatnonzero(late).tolist(),
                    retained_fraction_involving_late_frames=float(pair_count[involves_late].sum()/total) if total else None,
                    source_frame_retained=pair_count.sum(1).tolist(),
                    target_frame_retained=pair_count.sum(0).tolist(),
                    actual_loss_captures=[], within_step_changes=[])
        # Saved loss-path arrays cover all five forward-adjacent pairs; these are not the full loss.
        for capture in events:
            if capture['kind']!='training_flow' or capture['block']!=block:
                continue
            with np.load(trace/capture['file']) as data:
                pred, target, selected = data['flow'], data['reference'], data['mask'].astype(bool)
            errors=((pred-target)**2).mean(-1)
            pair_mse=[float(errors[i][selected[i]].mean()) if selected[i].any() else None for i in range(f-1)]
            item['actual_loss_captures'].append(dict(step=capture['step'], iteration=capture['iteration'],
                all_pairs_loss=capture['loss'], adjacent_pair_mse=pair_mse))
        for step in sorted({e['step'] for e in item['actual_loss_captures']}):
            selected=sorted([e for e in item['actual_loss_captures'] if e['step']==step],key=lambda e:e['iteration'])
            first,last=selected[0],selected[-1]
            if first['iteration']==last['iteration']:
                continue
            item['within_step_changes'].append(dict(step=step,
                before_iteration=first['iteration'],after_iteration=last['iteration'],
                mse_before=first['adjacent_pair_mse'],mse_after=last['adjacent_pair_mse']))
        result['blocks'][block]=item
    return result


def make_temporal_report(runs, output_dir, packed_reference=None):
    from probe_report import _inline_figure
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    results=[audit_run(run) for run in runs]
    packing=json.loads(Path(packed_reference).read_text(encoding='utf-8')) if packed_reference else None
    payload=dict(runs=results,packed_reference=packing)
    (output/'temporal_audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>Temporal AMF audit</title>',
          '<style>body{font:16px system-ui;max-width:1450px;margin:24px auto}img{max-width:100%}table{border-collapse:collapse}th,td{padding:6px;border:1px solid #ccc}</style>',
          '<h1>Video duration and temporal AMF coverage</h1>',
          '<p>Video frames and denoising iterations are different axes. All latent video frames coexist at every sampling step. '
          'Ending guidance after sampling step 9 does not mean only video frames 0-9 were guided. '
          'Latent frame i is approximately associated with decoded anchor 4*i, not one exact frame.</p>']
    if packing:
        page.append('<p>Packed source: '+html.escape(str(packing['frames']))+' frames at '+html.escape(str(packing['effective_fps']))+
                    ' FPS. This pilot consumes the first 21 and exports both preview and generation at 16 FPS. '
                    'The packed source time and the exported playback time therefore differ.</p>')
    table_rows=[]
    for result in results:
        page.append('<h2>'+html.escape(result['run'])+'</h2><table><tr><th>Video</th><th>Frames</th><th>FPS</th><th>Frames/FPS duration</th><th>Exact duplicate transitions</th></tr>')
        for video in result['videos']:
            page.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (video['file'],video['frames'],video['fps'],video['nominal_duration_s'],video['exact_duplicate_transitions']))+'</tr>')
        page.append('</table>')
        fig,ax=plt.subplots(figsize=(10,3))
        for video in result['videos']:
            ax.plot(np.arange(1,video['frames']),video['pixel_change_per_transition'],label=video['file'][:40])
        ax.set(xlabel='Decoded destination frame',ylabel='Mean absolute pixel change (0-255)',title='Frame activity, not object speed or camera motion')
        ax.legend(fontsize=8); fig.tight_layout(); page.append(_inline_figure(fig))
        for block,item in result['blocks'].items():
            late_share = item['retained_fraction_involving_late_frames']
            page.append('<h3>'+html.escape(block)+'</h3><p>Raw reference shape: '+html.escape(str(item['reference_shape']))+
                        '. Retained fraction involving late latent frames '+html.escape(str(item['late_frame_indices']))+': '+
                        (f'{late_share:.1%}' if late_share is not None else 'no retained entries')+'. Counts describe actual reference mask entries, not gradient importance.</p>')
            counts=np.asarray(item['retained_counts'])
            fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(counts,cmap='Blues',vmin=0)
            for i in range(len(counts)):
                for j in range(len(counts)):ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=9)
            ax.set(xlabel='Target latent frame',ylabel='Source latent frame',title='Retained patches per frame pair')
            fig.colorbar(im,ax=ax); fig.tight_layout(); page.append(_inline_figure(fig))
            page.append('<table><tr><th>Sampling index</th><th>Latent pair</th><th>Before MSE</th><th>After MSE</th></tr>')
            for change in item['within_step_changes']:
                for i,(before,after) in enumerate(zip(change['mse_before'],change['mse_after'])):
                    row=dict(run=result['run'],block=block,step=change['step'],latent_source=i,latent_target=i+1,
                             before_iteration=change['before_iteration'],after_iteration=change['after_iteration'],mse_before=before,mse_after=after)
                    table_rows.append(row)
                    if change['step'] in (0,9):
                        page.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (change['step'],f'{i}->{i+1}',before,after))+'</tr>')
            page.append('</table><p>Existing captures are before optimizer iteration 0 and before iteration 4: this compares four updates at a fixed sampling timestep, not all five updates. '
                        'Adjacent-pair MSE is diagnostic; actual optimization uses the retained entries across all frame pairs. These arrays do not directly measure per-frame gradients.</p>')
    if table_rows:
        with (output/'within_step_temporal_mse.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(table_rows[0])); writer.writeheader(); writer.writerows(table_rows)
    report=output/'temporal_audit.html'; report.write_text('\n'.join(page),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output_dir',type=Path,default=Path('probe_comparison/temporal_audit'))
    parser.add_argument('--packed_reference',type=Path,help='Optional packed clip meta.json')
    args=parser.parse_args()
    print(make_temporal_report(args.runs,args.output_dir,args.packed_reference))
