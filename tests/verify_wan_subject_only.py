"""Subject-only objective, archive/provenance gates, and real tiny BF16 sampling."""
import base64
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import wan_subject_only_pilot as pilot
from benchmark import wan_head_pilot as head
from benchmark import wan_control_pilot as control
from benchmark import wan_pair_pilot as pairs
from probe_wan_subject_only import SubjectOnlyLossMixin, subject_only_loss
from probe_wan_control import ControlGuidanceMixin
from probe_wan_noised_reference import NoisedReferenceMixin
from probe_wan_pairs import ForwardAdjacentMixin
from guidance_utils.wan_control_trace import ControlRecorder
from probe_report import load_trace


class ObjectiveTests(unittest.TestCase):
    def test_only_subject_mean_enters_backward(self):
        prediction = torch.tensor([[[2., 4.], [100., 200.], [3., 5.]]], requires_grad=True)
        target = torch.zeros_like(prediction)
        subject = torch.tensor([[True, False, True]])
        background = ~subject
        loss, fg, bg, weight = subject_only_loss(prediction, target, subject, background)
        self.assertIs(loss, fg)
        self.assertEqual(weight, 1.)
        self.assertEqual(loss.item(), 13.5)
        loss.backward()
        torch.testing.assert_close(prediction.grad[background], torch.zeros((1, 2)))
        torch.testing.assert_close(prediction.grad[subject], prediction.detach()[subject]/2)
        self.assertGreater(bg.item(), loss.item())


class ArchiveTests(unittest.TestCase):
    def fixture(self, archive):
        with zipfile.ZipFile(archive, 'w') as z:
            for prefix, stage, arms in [('subject', pilot.previous.STAGE, pilot.previous.ARMS),
                                         ('subject/previous', control.STAGE, control.ARMS)]:
                z.writestr(prefix+'/plan.json', json.dumps(dict(stage=stage)))
                for name in ['plan.sha256', 'environment.json', *[a+'_done.json' for a in arms]]:
                    z.writestr(prefix+'/'+name, '{}')
            z.writestr('subject/targets/forward.npz', b'raw-target')

    def test_selects_outer_subject_preserves_existing_and_reuses_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); archive=root/'result.zip'; self.fixture(archive)
            destination=root/'existing'; destination.mkdir(); (destination/'keep').write_text('unchanged')
            restored=pilot.restore_archive(archive,destination)
            self.assertNotEqual(restored,destination)
            self.assertEqual((destination/'keep').read_text(),'unchanged')
            self.assertEqual(pilot.read_json(restored/'plan.json')['stage'],pilot.previous.STAGE)
            self.assertEqual((restored/'targets/forward.npz').read_bytes(),b'raw-target')
            self.assertEqual(pilot.restore_archive(archive,destination),restored)
            self.assertEqual(pilot.restore_archive(archive,restored),restored)

    def test_rejects_traversal_and_review_only_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('../outside','bad')
            with self.assertRaisesRegex(ValueError,'Unsafe'): pilot.restore_archive(archive,root/'out')
            self.assertFalse((root/'outside').exists())
            with zipfile.ZipFile(archive,'w') as z: z.writestr('diagnosis.md','review only')
            with self.assertRaisesRegex(ValueError,'full completed SUBJECT'): pilot.restore_archive(archive,root/'out')


class EnvironmentAndDisplayTests(unittest.TestCase):
    def test_text_processing_mismatch_reports_both_values(self):
        expected=dict(text_processing=dict(packages=dict(ftfy='6.3.1', tokenizers='0.22.1')))
        current=dict(text_processing=dict(packages=dict(ftfy=None, tokenizers='0.23.0')))
        with self.assertRaises(RuntimeError) as error: pilot.require_same_environment(expected,current)
        self.assertIn("text_processing.packages.ftfy: recorded='6.3.1', current=None",str(error.exception))
        self.assertIn("tokenizers: recorded='0.22.1', current='0.23.0'",str(error.exception))
        pilot.require_same_environment(expected,expected)

    def test_display_embeds_seven_correct_videos(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for base,arms in [(root/'previous/previous',['off']),
                              (root/'previous',['balanced_forward','balanced_reverse']), (root,pilot.ARMS)]:
                for arm in arms:
                    folder=base/arm; folder.mkdir(parents=True)
                    head.write_json(base/(arm+'_done.json'),dict(directory=arm))
                    (folder/'final.mp4').write_bytes(('final/'+arm).encode())
                    (folder/'original.mp4').write_bytes(('reference/'+arm).encode())
            pilot.make_display(root,dict(motion=[]))
            page=(root/'subject_only_comparison.html').read_text()
            videos=re.findall(r'src="data:video/mp4;base64,([A-Za-z0-9+/=]+)"',page)
            expected=['reference/balanced_forward','reference/balanced_reverse','final/off',
                      'final/balanced_forward','final/balanced_reverse',
                      'final/subject_only_forward','final/subject_only_reverse']
            self.assertEqual([base64.b64decode(v).decode() for v in videos],expected)


class SamplingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by production Wan')
    def test_background_target_and_diagnostic_gradients_cannot_change_sampling(self):
        from motion_guidance_wan import WanGuidance
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.motion_probe import MotionProbe

        class SubjectWan(ControlGuidanceMixin, SubjectOnlyLossMixin, ForwardAdjacentMixin, NoisedReferenceMixin, WanGuidance):
            pass

        with tempfile.TemporaryDirectory() as temp:
            finals, losses={},{}
            for name, recorded, background_shift in [('plain',False,0.),('recorded',True,0.),('changed_bg',True,100.)]:
                folder=Path(temp)/name; folder.mkdir(); torch.manual_seed(17)
                g=SubjectWan.__new__(SubjectWan); torch.nn.Module.__init__(g)
                g.config=OmegaConf.create(dict(probe=True,probe_blocks=[1],probe_steps=[9],probe_rope=False,
                    guidance_blocks=[1],injection_blocks=[],loss_type='flow',flow_head=30,
                    motion_temp=2.,softmax_fp32=True,argmax_motion_flow=True,threshloss=True,flow_max_disp=100.,
                    optimization_steps=5,verbose=False,save_embeds=False,flow_loss='mse',
                    reference_noise_step=9,reference_noise_seed=29,flow_pair_mode=pairs.PAIR_MODE,
                    alignment_mode='subject_only',record_region_gradients=recorded))
                g.device,g.dtype=torch.device('cuda'),torch.bfloat16
                g.transformer=ControlledWanTransformer(patch_size=(1,2,2),num_attention_heads=40,
                    attention_head_dim=8,in_channels=4,out_channels=4,text_dim=16,freq_dim=16,
                    ffn_dim=32,num_layers=3,cross_attn_norm=True,qk_norm='rms_norm_across_heads',
                    eps=1e-6,rope_max_seq_len=64).to(device=g.device,dtype=g.dtype).eval().requires_grad_(False)
                g.transformer.enable_gradient_checkpointing()
                g.latent_height,g.latent_width,g.patch_size=4,6,2
                g.patches_height,g.patches_width,g.latent_num_frames=2,3,3
                g.checkpoint_amf,g._guidance_scale=True,5
                g.scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); g.scheduler.set_timesteps(50,device=g.device)
                g.timesteps=g.scheduler.timesteps; g.lr_by_step={9:.001}; g.output_path=str(folder)
                g.register_guidance([1]); g.register_attention_processor([0,1,2]); g.probe=MotionProbe(g,'wan')
                g.motion_latent=torch.randn(1,4,3,4,6,device=g.device,dtype=g.dtype)*10
                g.transformer.init_rope=g.transformer.default_rope(g.motion_latent).to(g.device)
                g.source_embeds=torch.randn(1,5,16,device=g.device,dtype=g.dtype)
                g.guidance_embeds=torch.randn(2,5,16,device=g.device,dtype=g.dtype)
                g.motion_timestep=torch.tensor([0],device=g.device)
                g.motion_attn_features=ForwardAdjacentMixin.load_attn_features(g)
                block='block_1_attn1_processor'; valid=g.motion_attn_masks[block].cpu().numpy()
                subject=np.zeros_like(valid); positions=np.argwhere(valid)
                for pair,pos in positions[:max(1,len(positions)//3)]: subject[pair,pos]=True
                background=valid & ~subject
                field=g.motion_attn_features[block].float().cpu().numpy().copy()
                field[subject]+=np.array([1.,-.5],dtype=np.float32); field[background]+=background_shift
                bundle=dict(flow=field,mask=valid,subject=subject,background=background)
                g.alignment={k:torch.as_tensor(bundle[k],device=g.device) for k in ('flow','subject','background')}
                target_path=folder/'target.npz'; np.savez_compressed(target_path,**bundle)
                np.savez_compressed(g.probe.path/'aligned_reference.npz',**bundle)
                g.probe.emit('aligned_reference',block=block,file='aligned_reference.npz',mode='subject_only')
                x=torch.randn_like(g.motion_latent).float(); g.control=ControlRecorder(g,block=1,head=30)
                try:
                    with torch.no_grad(),patch('probe_wan_head_visual.decode'):
                        for i,t in enumerate(g.timesteps):
                            before=x.clone()
                            if i==9:
                                with torch.enable_grad(): x,_=g.guidance_step(x,i,t,'latent','flow')
                            optimized=x
                            x=g.denoise_step(x,i,g.guidance_embeds)
                            g.probe.sampling(i,t,before,optimized,x,g.scheduler)
                finally: g.control.close()
                finals[name]=x.clone()
                _,_,events=load_trace(folder)
                control.noised.audit_trace(events,OmegaConf.to_container(g.config),'candidate')
                losses[name]=pilot.audit_losses(folder,target_path)
                for path in (folder/'control').glob('*.npz'):
                    control.audit_capture(path,shape=(1,4,3,4,6),expected_block=1)
                if recorded:
                    rows=pilot.gradient_rows(folder)
                    self.assertEqual(len(rows),5)
                    self.assertTrue(all(r['subject_gradient_rms']>0 and r['background_gradient_rms']>0 and
                                        r['weighted_background_gradient_rms']==0. and r['update_rms']>0 for r in rows))
                    path=folder/'region_gradients/iteration_02.npz'; saved=pilot.arrays(path)
                    np.savez_compressed(path,**{**saved,'latent':saved['latent']+1})
                    with self.assertRaisesRegex(ValueError,'trajectory'): pilot.gradient_rows(folder)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in g.transformer.parameters()))
            torch.testing.assert_close(finals['plain'],finals['recorded'],rtol=0,atol=0)
            torch.testing.assert_close(finals['recorded'],finals['changed_bg'],rtol=0,atol=0)
            self.assertEqual(losses['plain'],losses['recorded'])
            self.assertEqual([r['total'] for r in losses['recorded']],[r['total'] for r in losses['changed_bg']])
            self.assertGreater(losses['changed_bg'][-1]['background_mse'],losses['recorded'][-1]['background_mse']+100)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
