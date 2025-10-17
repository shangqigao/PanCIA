import pathlib
import shutil

def resplit_multiphase_breast_tumor(singlephase_dataset, multiphase_dataset):
    single_splits = pathlib.Path(singlephase_dataset).glob('*_mask')
    multi_train = pathlib.Path(f'{multiphase_dataset}/train_mask')
    multi_test = pathlib.Path(f'{multiphase_dataset}/test_mask')
    multi_paths = list(multi_train.glob('*.png')) + list(multi_test.glob('*.png'))
    for split_path in single_splits:
        split = split_path.name
        msk_paths = split_path.glob('*.png')
        case_names = [p.name.split('_')[0:2] for p in msk_paths]
        case_names = ['_'.join(n) for n in case_names]
        case_names = list(set(case_names))
        for p in multi_paths:
            name = '_'.join(p.name.split('_')[0:2])
            p = f'{p}'
            if name in case_names:
                print(f"Moving image and mask of {name}...")
                target_dir = f'{multiphase_dataset}/{split}'
                p_name = pathlib.Path(p).name
                if not pathlib.Path(f'{target_dir}/{p_name}').exists():
                    shutil.move(p, target_dir)
                p = p.replace('_mask', '')
                p = p.replace('_breast+tumor.png', '.png')
                target_dir = f'{multiphase_dataset}/{split}'.replace('_mask', '')
                p_name = pathlib.Path(p).name
                if not pathlib.Path(f'{target_dir}/{p_name}').exists():
                    shutil.move(p, f'{multiphase_dataset}/{split}'.replace('_mask', ''))

if __name__ == "__main__":
    singlephase_dataset = '/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/BiomedParse_TumorSegmentation/Breast_Tumor'
    multiphase_dataset = '/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/BiomedParse_TumorSegmentation/Multiphase_Breast_Tumor'
    # resplit_multiphase_breast_tumor(singlephase_dataset, multiphase_dataset)

    # check consistency between two datasets
    singlephase_splits = sorted(pathlib.Path(singlephase_dataset).glob('*'))
    singlephase_splits = [p for p in singlephase_splits if p.is_dir()]
    multiphase_splits = sorted(pathlib.Path(multiphase_dataset).glob('*'))
    multiphase_splits = [p for p in multiphase_splits if p.is_dir()]
    for s_split, m_split in zip(singlephase_splits, multiphase_splits):
        s_paths = pathlib.Path(s_split).glob('*.png')
        s_names = sorted([p.name.replace('_0001', '') for p in s_paths])
        m_paths = pathlib.Path(m_split).glob('*.png')
        m_names = sorted([p.name for p in m_paths])
        for s_name, m_name in zip(s_names, m_names):
            if s_name != m_name:
                print(s_name, m_name)
                raise ValueError('Found inconsistent case')