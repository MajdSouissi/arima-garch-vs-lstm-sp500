$src = 'c:\Users\LENOVO\Bureau\mastere'
$dst = 'c:\Users\LENOVO\Bureau\mastere_results_only'

# Create destination
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# --- 1. Reports & Summaries ---
$reports = @(
    '04_garch_report.txt',
    '07_evaluation_report.txt',
    '08_dm_report.txt',
    '09_final_research_summary.md',
    'diagnostic_log.txt',
    'diagnostic_report.md',
    'mean_variance_log.txt',
    'mean_variance_report.md',
    'validation_report.txt'
)
foreach ($file in $reports) {
    $s = Join-Path $src $file
    $d = Join-Path $dst $file
    if (Test-Path $s) { Copy-Item -Path $s -Destination $d -Force }
}

# --- 2. CSV result files ---
$csvFiles = @(
    'diagnostic_results.csv',
    'dm_tests.csv',
    'evaluation_merged.csv',
    'garch_forecasts.csv',
    'lstm_forecasts.csv',
    'lstm_hyperparameters.csv',
    'mean_variance_results.csv',
    'metrics_summary.csv'
)
foreach ($file in $csvFiles) {
    $s = Join-Path $src $file
    $d = Join-Path $dst $file
    if (Test-Path $s) { Copy-Item -Path $s -Destination $d -Force }
}

# --- 3. Plot directories ---
$plotDirs = @(
    'diagnostic_plots',
    'final_results_plots',
    'images',
    'lstm_plots',
    'mean_variance_plots',
    'plots'
)
foreach ($dir in $plotDirs) {
    $s = Join-Path $src $dir
    $d = Join-Path $dst $dir
    if (Test-Path $s) { Copy-Item -Path $s -Destination $d -Recurse -Force }
}

# --- 4. Paper / LaTeX ---
Copy-Item -Path (Join-Path $src 'paper') -Destination (Join-Path $dst 'paper') -Recurse -Force

# --- 5. LaTeX chapters at root ---
Copy-Item -Path (Join-Path $src 'chapitre1_cadre_etat_art.tex') -Destination $dst -Force
Copy-Item -Path (Join-Path $src 'chapitre2_fondements_modeles.tex') -Destination $dst -Force

# --- 6. Original data (small xlsx) ---
Copy-Item -Path (Join-Path $src 'SP500.xlsx') -Destination $dst -Force

Write-Host "`n=== mastere_results_only created successfully ==="
Write-Host "`nContents:"
Get-ChildItem -Path $dst | Format-Table Name, @{N='Size(KB)';E={if($_.PSIsContainer){'[DIR]'}else{[math]::Round($_.Length/1KB,1)}}}, LastWriteTime -AutoSize
