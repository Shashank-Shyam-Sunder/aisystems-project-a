$docStem = "07_promotional_events"

$files = @(
    @{
        Persona = "standard"
        Dataset = "synthetic_data/synthetic_questions_processed_2026-04-02_00-13-19_07_promotional_events_standard.json"
        Tag = "${docStem}_standard"
    },
    @{
        Persona = "frustrated"
        Dataset = "synthetic_data/synthetic_questions_processed_2026-04-02_00-13-38_07_promotional_events_frustrated.json"
        Tag = "${docStem}_frustrated"
    },
    @{
        Persona = "mismatch"
        Dataset = "synthetic_data/synthetic_questions_processed_2026-04-02_00-14-13_07_promotional_events_mismatch.json"
        Tag = "${docStem}_mismatch"
    }
)

foreach ($item in $files) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Running eval for persona: $($item.Persona)"
    Write-Host "Dataset: $($item.Dataset)"
    Write-Host "========================================"

    python scripts/eval_harness.py `
        --dataset $item.Dataset `
        --save-baseline `
        --output-tag $item.Tag

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Eval failed for $($item.Persona)" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "All persona evals completed." -ForegroundColor Green