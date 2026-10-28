#!/usr/bin/env python3
"""
igm-smb-inference's DA L-curve analysis (pipeline/analyze_da_lcurve.py, unmodified) for the dev stack: the
reg.thk of each multirun member is read from the dev override name (assimilations.data_assimilation.*).

    python tools/analyze_lcurve_dev.py <multirun dir> --input data/<igm input>.nc
"""
import importlib.util
import sys

SCRIPT = "/home/klleshi/Documents/igm-smb-inference/pipeline/analyze_da_lcurve.py"

spec = importlib.util.spec_from_file_location("analyze_da_lcurve", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.REG_PARAM = "assimilations.data_assimilation.regularization.thk"
sys.argv = [SCRIPT] + sys.argv[1:]
mod.main()
