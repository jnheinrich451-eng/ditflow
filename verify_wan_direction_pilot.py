"""Validate factorial controls, actual-trace rejection and exports without Wan inference."""
import argparse
import ast
import copy
import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from benchmark import wan_direction_pilot as pilot


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(clip_id=c, video_path=f'clips/{c}', prompt=f'Original prompt for {c}') for c in pilot.CASES]

    def test_factorial_controls_and_cli(self):
        from guidance_utils.motion_probe import add_probe_arguments
        from motion_guidance_wan import MODEL_IDS, WAN_NEGATIVE_PROMPT
        tree = ast.parse(Path('motion_guidance_wan.py').read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        nodes = []
        for node in main.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'opt' for t in node.targets):
                break
            nodes.append(node)
        context = dict(argparse=argparse, add_probe_arguments=add_probe_arguments,
                       MODEL_IDS=MODEL_IDS, WAN_NEGATIVE_PROMPT=WAN_NEGATIVE_PROMPT)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'parser', 'exec'), context)
        for model, offload in [('1.3b', False), ('14b', True)]:
            jobs = pilot.build_jobs(self.rows, 'inputs', 'outputs', model=model, cpu_offload=offload)
            self.assertEqual(len(jobs), 8)
            self.assertEqual(len({j['output'] for j in jobs}), 8)
            for clip in pilot.CASES:
                comparable = []
                for job in [j for j in jobs if j['clip_id'] == clip]:
                    args = context['parser'].parse_args(job['command'][3:])
                    self.assertEqual(args.model, model)
                    self.assertEqual(args.low_vram, offload)
                    self.assertEqual(MODEL_IDS[args.model], pilot.timing.MODEL_KEYS[model])
                    self.assertEqual(args.no_guidance, not job['amf_enabled'])
                    self.assertTrue(args.no_injection)
                    self.assertEqual(args.guidance_timestep_range, [50, 40])
                    self.assertEqual(args.guidance_blocks, [10])
                    self.assertEqual(job['expected_updates'], 0 if args.no_guidance else 50)
                    if job['prompt_condition'] == 'original':
                        self.assertEqual(job['prompt'], job['original_prompt'])
                    else:
                        self.assertTrue(job['prompt'].startswith(job['original_prompt'] + '. '))
                    for key in ('prompt', 'output_path', 'no_guidance'):
                        delattr(args, key)
                    comparable.append(vars(args))
                self.assertTrue(all(a == comparable[0] for a in comparable))

    def make_runs(self, root, model='1.3b', cpu_offload=False):
        jobs = pilot.build_jobs(self.rows, 'inputs', root, model=model, cpu_offload=cpu_offload)
        env = dict(packages={'torch': 'test'}, source_sha256={'model.py': 'abc'}, gpu=['test GPU'])
        (root/'plan.json').write_text(json.dumps(jobs))
        (root/'environment.json').write_text(json.dumps(env))
        with (root/'input_manifest.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)
        for job in jobs:
            run = Path(job['output']); trace = run/'probes/test'; trace.mkdir(parents=True)
            (run/'original.mp4').write_bytes(b'reference'); (run/'generated.mp4').write_bytes(b'generated')
            config = dict(model_key=pilot.timing.MODEL_KEYS[model], enable_model_cpu_offload=cpu_offload,
                opt_mode='latent', guidance_mode='latent',
                loss_type='flow', flow_loss='mse', flow_region_masks=None,
                guidance_blocks=[10] if job['amf_enabled'] else [], injection_blocks=[], optimization_steps=5,
                guidance_timestep_range=[50, 40], lr=[.002, .001], lr_decay_steps=10,
                num_inference_steps=50, seed=1, num_frames=21, height=480, width=832,
                scheduler='flowmatch', flow_shift=3., guidance_scale=5., motion_temp=2., flow_max_disp=100.,
                threshloss=True, argmax_motion_flow=True, flow_min_conf=None, softmax_fp32=True,
                source_prompt='', target_prompt=job['prompt'], output_path=str(run))
            meta = dict(config=config, packages=env['packages'], source_sha256=env['source_sha256'],
                        gpu='test GPU', python='test', scheduler='FlowMatchEulerDiscreteScheduler',
                        scheduler_config={'shift': 3}, grid=[6, 30, 52])
            (trace/'metadata.json').write_text(json.dumps(meta))
            events = [dict(kind='optimization', step=step, iteration=i, lr=rate,
                      gradient={'finite_fraction': 1}, update={'finite_fraction': 1, 'rms': .001}, loss_before_update=1.)
                      for step, rate in zip(job['sampling_indices'], job['learning_rates']) for i in range(5)]
            events += [dict(kind='sampling', step=step, sigma=1-step/50,
                           guidance_update={'finite_fraction': 1, 'rms': .001 if step in job['sampling_indices'] else 0})
                       for step in range(50)]
            if job['amf_enabled']:
                np.savez(trace/'reference.npz', flow=np.ones((36, 8, 2)), mask=np.ones((36, 8), dtype=bool))
                events.append(dict(kind='training_reference', block='block_10_attn1_processor', file='reference.npz'))
            (trace/'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
        return jobs, env

    def test_14b_changes_only_checkpoint_and_memory_placement(self):
        small = pilot.build_jobs(self.rows, 'inputs', 'outputs')
        large = pilot.build_jobs(self.rows, 'inputs', 'outputs', model='14b', cpu_offload=True)
        for a, b in zip(small, large):
            self.assertEqual(a['prompt'], b['prompt'])
            self.assertEqual(a['learning_rates'], b['learning_rates'])
            command = b['command'].copy()
            command[command.index('--model')+1] = '1.3b'
            command.remove('--low_vram')
            self.assertEqual(a['command'], command)

    def test_14b_audit_rejects_wrong_checkpoint_or_offload_and_mixed_plan(self):
        for mutation in (None, 'checkpoint', 'offload', 'mixed'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp, patch(
                    'benchmark.wan_timing_pilot.decoded_digest', return_value='same'):
                root = Path(temp); jobs, _ = self.make_runs(root, model='14b', cpu_offload=True)
                if mutation is None:
                    report = pilot.audit_experiment(root)
                    self.assertEqual({r['model'] for r in report}, {'14b'})
                    self.assertEqual([r['updates'] for r in report], [0, 50, 0, 50]*2)
                    continue
                if mutation == 'mixed':
                    jobs[-1]['model'] = '1.3b'
                    (root/'plan.json').write_text(json.dumps(jobs))
                else:
                    path = Path(jobs[0]['output'])/'probes/test/metadata.json'
                    meta = json.loads(path.read_text())
                    if mutation == 'checkpoint':
                        meta['config']['model_key'] = pilot.timing.MODEL_KEYS['1.3b']
                    else:
                        meta['config']['enable_model_cpu_offload'] = False
                    path.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    pilot.audit_experiment(root)

    def test_offload_does_not_first_load_full_pipeline_to_cuda(self):
        # Execute the real initialization placement branch without loading weights.
        tree = ast.parse(Path('motion_guidance_wan.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanGuidance')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Attribute) and n.test.attr == 'enable_model_cpu_offload')
        # Include the preceding statement to catch an unconditional pipe.to(cuda).
        nodes = [init.body[start]]
        if isinstance(init.body[start-1], ast.Expr):
            nodes.insert(0, init.body[start-1])
        branch = ast.Module(body=nodes, type_ignores=[])
        for enabled in (False, True):
            pipe = Mock()
            owner = SimpleNamespace(pipe=pipe, device='cuda:0')
            config = SimpleNamespace(enable_model_cpu_offload=enabled, scheduler='unchanged')
            exec(compile(branch, 'placement', 'exec'), dict(self=owner, config=config))
            if enabled:
                pipe.to.assert_not_called()
                pipe.enable_model_cpu_offload.assert_called_once_with(device='cuda:0')
            else:
                pipe.to.assert_called_once_with('cuda:0')
                pipe.enable_model_cpu_offload.assert_not_called()

    def test_cached_rope_uses_execution_device_after_cpu_construction(self):
        import torch
        tree = ast.parse(Path('motion_guidance_wan.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanGuidance')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        node = next(n for n in init.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == 'init_rope' for t in n.targets))
        # Meta device exercises an actual device move without requiring a GPU.
        frequencies = torch.ones(2, 1, 4, 1, 8)
        transformer = SimpleNamespace(default_rope=lambda _: frequencies)
        owner = SimpleNamespace(transformer=transformer, device=torch.device('meta'), init_latents=torch.zeros(1))
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'rope_placement', 'exec'), {'self': owner})
        self.assertEqual(transformer.init_rope.device.type, 'meta')
        self.assertEqual(transformer.init_rope.dtype, frequencies.dtype)
        self.assertEqual(frequencies.device.type, 'cpu')

    def test_valid_eight_runs_and_relocated_archive(self):
        with tempfile.TemporaryDirectory() as temp, patch('benchmark.wan_timing_pilot.decoded_digest', return_value='same'):
            root = Path(temp); jobs, _ = self.make_runs(root)
            # Stored Colab paths need not exist on the machine reviewing the archive.
            for job in jobs:
                job['output'] = '/content/old/' + job['clip_id'] + '/' + job['method']
            (root/'plan.json').write_text(json.dumps(jobs))
            reports = pilot.audit_experiment(root)
            self.assertEqual([r['updates'] for r in reports], [0, 50, 0, 50]*2)

    def test_scheduler_default_name_order_is_not_a_configuration_change(self):
        with tempfile.TemporaryDirectory() as temp, patch('benchmark.wan_timing_pilot.decoded_digest', return_value='same'):
            root = Path(temp); jobs, _ = self.make_runs(root)
            originals = {}
            for i, job in enumerate(jobs):
                metadata = Path(job['output'])/'probes/test/metadata.json'
                meta = json.loads(metadata.read_text())
                names = ['shift_terminal', 'base_shift', 'invert_sigmas']
                meta['scheduler_config']['_use_default_values'] = names if i % 2 else names[::-1]
                metadata.write_text(json.dumps(meta))
                originals[metadata] = metadata.read_bytes()
            reports = pilot.audit_experiment(root)
            self.assertEqual(len(reports), 8)
            self.assertTrue(all(path.read_bytes() == data for path, data in originals.items()))

    def test_scheduler_real_changes_still_fail_with_details(self):
        changes = [('shift', 4), ('_use_default_values', ['base_shift']),
                   ('disable_corrector', [1, 0]), ('_diffusers_version', 'changed')]
        for key, value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp, patch(
                    'benchmark.wan_timing_pilot.decoded_digest', return_value='same'):
                root = Path(temp); jobs, _ = self.make_runs(root)
                for job in jobs:
                    metadata = Path(job['output'])/'probes/test/metadata.json'
                    meta = json.loads(metadata.read_text())
                    meta['scheduler_config'].update(_use_default_values=['base_shift', 'shift_terminal'],
                                                     disable_corrector=[0, 1], _diffusers_version='original')
                    if job == jobs[1]:
                        meta['scheduler_config'][key] = value
                    metadata.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, 'scheduler_config.*' + key):
                    pilot.audit_experiment(root)

    def test_reject_control_update_missing_guidance_and_wrong_prompt(self):
        for mutation in ('control_delta', 'rogue_optimizer', 'missing_guidance', 'wrong_prompt', 'nonfinite_update', 'injection'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp, patch(
                    'benchmark.wan_timing_pilot.decoded_digest', return_value='same'):
                root = Path(temp); jobs, _ = self.make_runs(root)
                job = jobs[0] if mutation in ('control_delta', 'rogue_optimizer') else jobs[1]
                trace = Path(job['output'])/'probes/test'
                events = [json.loads(s) for s in (trace/'events.jsonl').read_text().splitlines()]
                if mutation == 'control_delta':
                    events[0]['guidance_update']['rms'] = .1
                elif mutation == 'rogue_optimizer':
                    events.insert(0, dict(kind='optimization', step=0, iteration=0))
                elif mutation == 'missing_guidance':
                    events.pop(0)
                elif mutation == 'wrong_prompt':
                    meta = json.loads((trace/'metadata.json').read_text())
                    meta['config']['target_prompt'] = 'different'
                    (trace/'metadata.json').write_text(json.dumps(meta))
                elif mutation == 'nonfinite_update':
                    events[0]['update']['finite_fraction'] = 0
                elif mutation == 'injection':
                    events.append(dict(kind='attention', injected=True))
                (trace/'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
                with self.assertRaises(ValueError):
                    pilot.audit_experiment(root)

    def test_reject_reference_and_environment_mismatch(self):
        for mutation in ('amf', 'environment', 'config'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp, patch(
                    'benchmark.wan_timing_pilot.decoded_digest', return_value='same'):
                root = Path(temp); jobs, _ = self.make_runs(root)
                trace = Path(jobs[3]['output'])/'probes/test'
                if mutation == 'amf':
                    np.savez(trace/'reference.npz', flow=np.zeros((36, 8, 2)), mask=np.ones((36, 8), dtype=bool))
                else:
                    meta = json.loads((trace/'metadata.json').read_text())
                    if mutation == 'environment':
                        meta['packages']['torch'] = 'changed'
                    else:
                        meta['config']['negative_prompt'] = 'extra text'
                    (trace/'metadata.json').write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    pilot.audit_experiment(root)

    def test_completed_jobs_resume_without_generation(self):
        with tempfile.TemporaryDirectory() as temp, patch('benchmark.wan_timing_pilot.decoded_digest', return_value='same'):
            root = Path(temp); jobs, env = self.make_runs(root)
            for job in jobs:
                (Path(job['output'])/'direction_done.json').write_text(json.dumps(job))
            with patch.object(pilot, 'environment_snapshot', return_value=env), patch.object(pilot.subprocess, 'Popen') as popen:
                pilot.run_jobs(root)
                popen.assert_not_called()

    def test_archive_includes_videos_and_drive_copy_excludes_embeds(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); root = parent/'run'; root.mkdir()
            (root/'original.mp4').write_bytes(b'ref')
            (root/'generated.mp4').write_bytes(b'video')
            (root/'embeds').mkdir(); (root/'embeds/latent.pt').write_bytes(b'large')
            (root/'trace.json').write_text('{}')
            drive = parent/'MyDrive'; drive.mkdir()
            archive, copied = pilot.archive_experiment(root, drive/'ditflow_probes')
            self.assertEqual(archive.read_bytes(), copied.read_bytes())
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), {'run/original.mp4', 'run/generated.mp4', 'run/trace.json'})
            _, second = pilot.archive_experiment(root, drive/'ditflow_probes')
            self.assertNotEqual(copied, second)


if __name__ == '__main__':
    unittest.main(verbosity=2)
