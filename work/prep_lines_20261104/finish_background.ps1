$ErrorActionPreference='Stop'
Wait-Process -Id 4704 -ErrorAction SilentlyContinue
Set-Location -LiteralPath 'C:/Users/Administrator/Desktop/数据集处理'
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
if (-not (Test-Path -LiteralPath 'work/prep_lines_20261104/verification.json')) { throw 'Main analysis has not produced verification.json' }
& D:/anaconda/python.exe -I work/prep_lines_20261104/refine_validation.py *> work/prep_lines_20261104/refine.log
if ($LASTEXITCODE -ne 0) { throw 'Refinement failed; inspect refine.log' }
& D:/anaconda/python.exe -I work/prep_lines_20261104/finalize_review.py *> work/prep_lines_20261104/finalize.log
if ($LASTEXITCODE -ne 0) { throw 'Finalization failed; inspect finalize.log' }
