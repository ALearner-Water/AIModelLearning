$scripts = @(
    'Python_basic/python_to_numpy/python_numpy_readiness_challenge.py'
    'Numpy_basic/numpy_final_challenge.py'
    'Pytorch_basic/pytorch_class/tensor/tensor_phase2_challenge.py'
    'Pytorch_basic/pytorch_class/autograd/autograd.py'
    'Pytorch_basic/pytorch_class/dataset_dataloader/dataset_basics.py'
    'Pytorch_basic/pytorch_class/dataset_dataloader/dataloader_training_loop.py'
)

foreach ($script in $scripts) {
    Write-Host "`n==> $script"
    conda run -n dl-study python $script
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n全部核心脚本运行通过。"
