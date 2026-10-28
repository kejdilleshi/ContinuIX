#!/bin/bash
# Family-level S01 scores (tools/check_s01.py) of the original submission and of SMB-phase reruns.
#   bash tools/s01_family.sh EXP01 tagA tagB ...
cd "$(dirname "$0")/.."
e=$1; shift
row() { python tools/check_s01.py $1 --src input_data/$e/${e}_S01.nc 2>/dev/null | python -c "
import sys,re
t=sys.stdin.read()
p=re.search(r'admissible mean profile: bias (\S+), RMSE (\S+) m i.e./yr; ELA range of admissible members \[(\S+), (\S+)\]',t)
s=re.search(r'pixel SMB vs law: n=\d+  bias (\S+)  RMSE (\S+)',t)
d=re.search(r'pixel FDIV vs law \(steady state: FDIV = SMB\): bias \S+  RMSE (\S+)',t)
print('profile RMSE %5.2f bias %+5.2f | ELA %5.0f | pixel SMB RMSE %5.2f | FDIV RMSE %5.2f'%(float(p.group(2)),float(p.group(1)),(float(p.group(3))+float(p.group(4)))/2,float(s.group(2)),float(d.group(1))))"; }
[ -z "$NO_ORIG" ] && printf "%-16s %s\n" original "$(row GROUP_UNIL/$e/${e}_S01_method01.nc)"
for t in "$@"; do f=results/${e}_S01_cnn/smb_$t/export/${e}_S01_method01.nc; [ -f $f ] && printf "%-16s %s\n" $t "$(row $f)"; done
