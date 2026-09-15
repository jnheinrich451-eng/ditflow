"""Weight-free tests for portable step-39 setup, report gate, and small archives."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from benchmark.wan_camel_correspondence import digest, prepare_step39, checked_plan, load_model, timing_report
from benchmark.wan_acceptance_runtime import restore_correspondence_inputs, save_diagnostic_archives
from probe_wan_response import tensor_hash


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); previous=root/'previous'; previous.mkdir()
        embeddings=torch.zeros(2,1,1)
        plan=dict(states=['clean','step_09'],controls=['forward','reverse','static'],noise_seed=29,
            checkpoint_revision='pinned-test',conditioning_sha256=tensor_hash(embeddings),
            sigmas=torch.linspace(1,0,51).tolist(),timesteps=torch.arange(50).tolist(),
            input_sha256={},grid=[6,30,52],block=20,saved_run='/gone/old/acceptance',saved_manifest_sha256='gone')
        for name in ('controls/forward/00000.png','controls/reverse/00000.png','controls/static/00000.png','rgb_motion.npz'):
            path=previous/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'fixture')
            plan['input_sha256'][name]=digest(path)
        (previous/'rgb_regions.jpg').write_bytes(b'roi')
        (previous/'plan.json').write_text(json.dumps(plan))
        (previous/'source_revision.json').write_text('{}')
        (previous/'correspondence_report.json').write_text('{"rows":[]}')
        (previous/'readouts').mkdir()
        (previous/'readouts/metadata.json').write_text(json.dumps(dict(
            checkpoint='/snapshots/pinned-test',conditioning_sha256=plan['conditioning_sha256'],
            sigmas=plan['sigmas'],timesteps=plan['timesteps'],noise_sha256='old-noise',
            events=[dict(control=c,state=s) for c in plan['controls'] for s in plan['states']])))
        (previous/'readouts/forward_clean.npz').write_bytes(b'large-QK-not-needed')
        archive=root/'previous.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for file in previous.rglob('*'):
                if file.is_file(): z.write(file,file.relative_to(previous))
        restored=restore_correspondence_inputs(archive,root/'restored')
        assert restore_correspondence_inputs(archive,root/'restored')==restored
        assert not (restored/'readouts/forward_clean.npz').exists()
        candidate=prepare_step39(restored,root/'step39')
        updated=checked_plan(candidate)
        assert updated['states']==['step_39'] and updated['max_transformer_forwards']==3
        assert 'saved_run' not in updated and updated['expected_noise_sha256']=='old-noise'
        assert updated['input_sha256']['rgb_motion.npz']==plan['input_sha256']['rgb_motion.npz']
        constructed=[]
        class FakeGuidance:
            def __init__(self,config):
                assert config.loss_type=='flow' and config.guidance_blocks==[20]
                self.config=config; self.dtype=torch.bfloat16; self.init_latents=torch.zeros(1)
                self.guidance_embeds=embeddings
                self.scheduler=SimpleNamespace(sigmas=torch.tensor(plan['sigmas']))
                self.timesteps=torch.tensor(plan['timesteps'])
                # Derived readout initializer must suppress both expensive paths.
                assert torch.equal(self.load_latent(),torch.zeros(1,dtype=torch.bfloat16))
                assert self.load_attn_features()=={}
                constructed.append(config)
        with (patch('motion_guidance_wan.WanGuidance',FakeGuidance),
              patch('huggingface_hub.snapshot_download',return_value='/snapshots/pinned-test') as download,
              patch('torch.cuda.is_available',return_value=True),
              patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=40*2**30))):
            loaded=load_model(candidate)
            assert len(constructed)==1 and loaded.config.enable_model_cpu_offload
            download.assert_called_once_with('Wan-AI/Wan2.1-T2V-14B-Diffusers',revision='pinned-test')
        rows=[]
        for control in plan['controls']:
            for pair in range(5):
                for region in ('subject','background','subject_relative_to_background'):
                    dx=-1. if control=='forward' else 1. if control=='reverse' else 0.
                    rows.append(dict(control=control,state='step_39',field='soft',region=region,pair=pair,
                        patches=30,amf_dx=dx,image_dx=dx,cosine=1.,epe=0.,
                        includes_corrupt_endpoint=(control=='forward' and pair==0) or (control=='reverse' and pair==4)))
        report=candidate/'correspondence_report.json'; report.write_text(json.dumps(dict(rows=rows)))
        assert timing_report(candidate)['ready_for_decoded_comparison'] is True
        assert timing_report(candidate)['port_success'] is False
        rows[5]['amf_dx']=1. # Break a forward relative-motion pair's sign.
        report.write_text(json.dumps(dict(rows=rows)))
        assert timing_report(candidate)['ready_for_decoded_comparison'] is False
        rows[5]['amf_dx']=None
        report.write_text(json.dumps(dict(rows=rows)))
        assert timing_report(candidate)['ready_for_decoded_comparison'] is False
        (candidate/'readouts').mkdir()
        (candidate/'readouts/forward_step_39.npz').write_bytes(b'raw tensors')
        (candidate/'readouts/metadata.json').write_text('{}')
        (candidate/'source_revision.json').write_text('{}')
        archives=save_diagnostic_archives(candidate,root/'drive')
        with zipfile.ZipFile(archives['review']) as small:
            assert 'timing_comparison.json' in small.namelist()
            assert 'readouts/forward_step_39.npz' not in small.namelist()
        with zipfile.ZipFile(archives['full']) as full:
            assert 'readouts/forward_step_39.npz' in full.namelist()
        print('PASS: selective restore, unchanged inputs, no old acceptance dependency, zero-forward setup, pinned snapshot, screening gate, split archives.')


if __name__=='__main__': main()
