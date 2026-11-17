$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$githubCli = Get-Command gh -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $githubCli -and (Test-Path 'C:\Program Files\GitHub CLI\gh.exe')) {
    $githubCli = 'C:\Program Files\GitHub CLI\gh.exe'
}
if (-not $githubCli) {
    throw 'Install GitHub CLI first: winget install --id GitHub.cli --exact. Then reopen PowerShell and run gh auth login.'
}
$githubAccount = & $githubCli api user --jq .login
if ($LASTEXITCODE -ne 0 -or $githubAccount -ne 'Danm2810') {
    throw 'Run gh auth login and authenticate as Danm2810 before publishing.'
}
& $githubCli repo create Danm2810/neural-network-quantization-lab --private --source . --remote origin --push
if ($LASTEXITCODE -ne 0) { throw 'Publishing failed. Review the GitHub CLI output before retrying.' }
