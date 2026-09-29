# Double-click start.cmd to run this: setup.ps1, then run.ps1 --parity.
# Everything shown in the window is also written to log.txt next to this file, and the PC is kept
# awake until the window closes.
$ErrorActionPreference = "Continue"
$here = $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
$env:TQDM_MININTERVAL = "20"        # progress bars update every 20 s, so the log stays readable
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Log file that other programs can read while it's being written.
$stream = New-Object System.IO.FileStream((Join-Path $here "log.txt"), "Create", "Write", "ReadWrite")
$log = New-Object System.IO.StreamWriter($stream, (New-Object System.Text.UTF8Encoding($false)))
$log.AutoFlush = $true
function Show {
  process {
    $text = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" }
    foreach ($line in ($text -split "`r")) {
      if ($line.Trim()) {
        Write-Host $line
        $stamped = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $line
        $log.WriteLine($stamped)
      }
    }
  }
}

# Keep the PC from sleeping while this window is open (released automatically when it closes).
try {
  Add-Type -Namespace ConceptLens -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
  [ConceptLens.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
} catch { }

$ok = $true
foreach ($stage in @(@{script = "setup.ps1"; args = @()}, @{script = "run.ps1"; args = @("--parity")})) {
  "`n######## $($stage.script) $($stage.args -join ' ')" | Show
  try {
    & (Join-Path $here $stage.script) @($stage.args) *>&1 | Show
    if ($LASTEXITCODE) { throw "$($stage.script) stopped with exit code $LASTEXITCODE" }
  } catch {
    "FAILED: $_" | Show
    $ok = $false
    break
  }
}
if ($ok) { "`nALL DONE. The bundle is in $(Resolve-Path (Join-Path $here '..\bundle'))" | Show }
$log.Close()
if (-not $ok) { exit 1 }
