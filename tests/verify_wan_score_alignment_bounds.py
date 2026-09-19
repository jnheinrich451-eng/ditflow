"""Validate saved-summary bounds against explicit probability-gradient matrices."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmark.review_wan_centered_torso import gradient_alignment_bounds


def cosine(a,b):
    return float((a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b))


def verify():
    rng=np.random.default_rng(47)
    for case in range(80):
        pairs,n=3,32
        logits=rng.normal(size=(pairs,n,n))*(.5+case%8)
        logits-=logits.max(-1,keepdims=True)
        p=np.exp(logits);p/=p.sum(-1,keepdims=True)
        support=rng.random((pairs,n)) < (.1+.1*(case%8))
        support[0,0]=True
        ids={a:rng.integers(0,n,size=(pairs,n)) for a in ('forward','reverse')}
        if case%5==0:
            ids['reverse']=ids['forward'].copy()
        logs={a:np.log(np.take_along_axis(p,ix[...,None],axis=-1)[...,0]) for a,ix in ids.items()}
        gradients={}
        for a in ids:
            target=np.eye(n)[ids[a]]
            gradients[a]=(p-target)*support[...,None]
        bound=gradient_alignment_bounds(p.max(-1),logs,ids,support)
        c=cosine(gradients['forward'],gradients['reverse'])
        lo,hi=bound['centered_score_gradient_cosine_bounds']
        assert lo-1e-12 <= c <= hi+1e-12
        raw={a:g-g.mean(1,keepdims=True) for a,g in gradients.items()}
        for a in raw:
            removed=(gradients[a]-raw[a])
            fraction=float((removed**2).sum()/(gradients[a]**2).sum())
            assert fraction <= bound['max_centering_removed_energy_fraction']+1e-12
        assert cosine(raw['forward'],raw['reverse']) <= bound['raw_score_gradient_cosine_upper_bound']+1e-12
    print('PASS: 80 explicit softmax-gradient comparisons, identical/different targets, sparse/dense masks, centering projection energy and cosine bounds.')


if __name__=='__main__':
    verify()
