#!/bin/bash
# scans finished so far (of 423) and the running shards
cd "$(dirname "$0")"; source env.sh
python - <<P
import glob, pandas as pd
fs = glob.glob('$PANCIA_CLINICAL/PanCIA_outputs/stage2/validation_agents_v3/rows*.csv')
d = pd.concat([pd.read_csv(f, low_memory=False, usecols=['ds', 'case', 'kind']) for f in fs]) if fs else None
print('finished scans:', 0 if d is None else d[d.kind != 'error'][['ds', 'case']].drop_duplicates().shape[0], 'of 423;',
      'errors:', 0 if d is None else int((d.kind == 'error').sum()))
P
echo "running shards: $(pgrep -fc '^python[0-9.]* s2_validate_agents\.py .*validation_agents_v3')"
