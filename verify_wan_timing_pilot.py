"""Controlled timing plan and actual-trace audit tests, without model inference."""
import ast
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.wan_timing_pilot import build_jobs, require_same_environment, validate_run, audit_experiment


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(clip_id=c,video_path=f'clips/{c}',prompt=f'Prompt for {c}') for c in ('car-turn','camel')]

    def test_pairs_differ_only_in_window_and_output(self):
        jobs=build_jobs(self.rows,'inputs','outputs')
        self.assertEqual(len(jobs),4)
        def comparable(command):
            values=[]; index=0
            while index<len(command):
                if command[index]=='--guidance_timestep_range':index+=3; continue
                if command[index]=='--output_path':index+=2; continue
                values.append(command[index]); index+=1
            return values
        for early,later in (jobs[:2],jobs[2:]):
            self.assertEqual(comparable(early['command']),comparable(later['command']))
            self.assertEqual(early['sampling_indices'],list(range(10)))
            self.assertEqual(later['sampling_indices'],list(range(20,30)))
            self.assertEqual(early['learning_rates'],later['learning_rates'])
            self.assertEqual(early['expected_updates'],50)
            self.assertIn('--no_injection',early['command']); self.assertNotIn('--flow_region_masks',early['command'])

    def test_commands_parse_against_actual_cli(self):
        from guidance_utils.motion_probe import add_probe_arguments
        from motion_guidance_wan import MODEL_IDS,WAN_NEGATIVE_PROMPT
        tree=ast.parse(Path('motion_guidance_wan.py').read_text(encoding='utf-8'))
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        nodes=[]
        for node in main.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='opt' for t in node.targets):break
            nodes.append(node)
        context=dict(argparse=argparse,add_probe_arguments=add_probe_arguments,MODEL_IDS=MODEL_IDS,WAN_NEGATIVE_PROMPT=WAN_NEGATIVE_PROMPT)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'parser','exec'),context)
        for job in build_jobs(self.rows,'inputs','outputs'):
            args=context['parser'].parse_args(job['command'][3:])
            self.assertEqual(args.guidance_timestep_range,job['window'])
            self.assertEqual(args.optimization_steps,5); self.assertEqual(args.lr_decay_steps,10)
            self.assertIsNone(args.flow_region_masks); self.assertTrue(args.no_injection)
            self.assertTrue(set(job['sampling_indices'][i] for i in (0,1,4,9))<=set(args.probe_steps))

    def test_environment_changes_are_rejected(self):
        expected=dict(packages={'torch':'2.6'},source_sha256={'model':'abc'},gpu=['A100'])
        require_same_environment(expected,copy.deepcopy(expected))
        for key,value in [('packages',{'torch':'2.11'}),('source_sha256',{'model':'new'}),('gpu',['other'])]:
            changed=copy.deepcopy(expected); changed[key]=value
            with self.assertRaises(RuntimeError):require_same_environment(expected,changed)

    def test_actual_trace_audit_rejects_wrong_updates_and_mixed_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            jobs=build_jobs(self.rows,'inputs',root)
            environment=dict(packages={'torch':'test'},source_sha256={'model.py':'abc'},gpu=['test GPU'])
            (root/'plan.json').write_text(json.dumps(jobs)); (root/'environment.json').write_text(json.dumps(environment))
            for job in jobs:
                run=Path(job['output']); trace=run/'probes/test'; trace.mkdir(parents=True)
                (run/'original.mp4').write_bytes(b'reference'); (run/'generated.mp4').write_bytes(b'generated')
                config=dict(model_key='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',opt_mode='latent',guidance_mode='latent',loss_type='flow',flow_loss='mse',
                    flow_region_masks=None,guidance_blocks=[10],injection_blocks=[],optimization_steps=5,
                    guidance_timestep_range=job['window'],lr=[.002,.001],lr_decay_steps=10,
                    num_inference_steps=50,seed=1,num_frames=21,height=480,width=832,scheduler='flowmatch',
                    flow_shift=3.,guidance_scale=5.,motion_temp=2.,flow_max_disp=100.,threshloss=True,
                    argmax_motion_flow=True,flow_min_conf=None,softmax_fp32=True,source_prompt='',target_prompt=job['prompt'])
                meta=dict(config=config,packages=environment['packages'],source_sha256=environment['source_sha256'],gpu='test GPU')
                (trace/'metadata.json').write_text(json.dumps(meta))
                events=[dict(kind='optimization',step=step,iteration=i,lr=rate,gradient={'finite_fraction':1},loss_before_update=1.)
                        for step,rate in zip(job['sampling_indices'],job['learning_rates']) for i in range(5)]
                events += [dict(kind='sampling',step=step,sigma=1-step/50) for step in range(50)]
                (trace/'events.jsonl').write_text('\n'.join(map(json.dumps,events)))
            with patch('benchmark.wan_timing_pilot.decoded_digest',return_value='same decoded input'):
                report=audit_experiment(root); self.assertEqual([r['updates'] for r in report],[50]*4)
                bad=copy.deepcopy(environment); bad['packages']['torch']='changed'
                with self.assertRaises(ValueError):validate_run(jobs[0],expected_environment=bad)
                trace=Path(jobs[0]['output'])/'probes/test/events.jsonl'
                lines=trace.read_text().splitlines(); trace.write_text('\n'.join(lines[1:]))
                with self.assertRaises(ValueError):validate_run(jobs[0])


if __name__=='__main__':
    unittest.main(verbosity=2)
