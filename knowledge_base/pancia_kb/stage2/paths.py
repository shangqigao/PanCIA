"""Data roots. Order: environment variable PANCIA_<NAME> (e.g. PANCIA_R2SEG), then ~/mnt/<name> (Cowork VM), then the Mac
defaults below. Names: PanCIA, clinical, TCGA_Seg, TCGA_NIFTI, R2Seg, Totalsegmentator."""
import os

_MAC = {
    'PanCIA': '/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA',
    'clinical': '/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/clinical',
    'TCGA_Seg': '/Users/sg2162/Datasets/CancerDatasets/PanCIA/TCGA-TCIA/TCGA_Seg',
    'TCGA_NIFTI': '/Users/sg2162/Datasets/CancerDatasets/PanCIA/TCGA-TCIA/TCGA_NIFTI',
    'R2Seg': '/Users/sg2162/Datasets/CancerDatasets/R2Seg',
    'Totalsegmentator': '/Users/sg2162/Datasets/Totalsegmentator',
}


def root(name):
    env = os.environ.get('PANCIA_' + name.upper())
    if env:
        return env
    vm = os.path.join(os.path.expanduser('~'), 'mnt', name)
    if os.path.isdir(vm):
        return vm
    return _MAC[name]


def kb_dir():
    return os.path.join(root('PanCIA'), 'knowledge_base')


def fit_path(tag='stage2_fit.json'):
    """fitted stage-2 values live with the code (knowledge_base/knowledge_base/stage2/), not in an output folder"""
    return os.path.join(kb_dir(), 'knowledge_base', 'stage2', tag)


def ledger_dir():
    return os.path.join(root('clinical'), 'PanCIA_outputs', 'ledger')


def out_dir(*sub):
    d = os.path.join(root('clinical'), 'PanCIA_outputs', 'stage2', *sub)
    os.makedirs(d, exist_ok=True)
    return d
