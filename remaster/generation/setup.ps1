# One-time setup for generating ConceptLens data on this PC. Needs an NVIDIA GPU (6 GB or more).
#
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# Creates .venv here, installs PyTorch with CUDA and the other packages, fetches the concept-lens
# repository, the StyleGAN2 checkpoint and the CLIP weights: about 8 GB of downloads the first time.
# Safe to run again; finished steps are skipped.
param(
  [string]$Cuda = "",              # force a PyTorch build: cu130, cu128 or cu126 (default: picked from the driver)
  [string]$Python = "",            # a python.exe (3.10 to 3.13) to build .venv from (default: newest suitable one found)
  [string]$Domain = "s2_ffhq256"   # which StyleGAN checkpoint to download
)
$ErrorActionPreference = "Continue"      # native tools write progress to stderr; exit codes are checked instead
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is very slow with its progress bar
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$here = $PSScriptRoot

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }
function Quietly([scriptblock]$block) { & $block *> $null; return ($LASTEXITCODE -eq 0) }

# 1. GPU ------------------------------------------------------------------------------------------
Step "Checking the NVIDIA GPU"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  throw "nvidia-smi wasn't found. Feature extraction needs an NVIDIA GPU with its driver installed."
}
$smiLines = @(& nvidia-smi 2>&1 | ForEach-Object { "$_" })
$driverCuda = $null
if (($smiLines -join "`n") -match "CUDA Version:\s*(\d+)\.(\d+)") {
  $driverCuda = [version]"$($Matches[1]).$($Matches[2])"
} else {
  # No "CUDA Version" in the header: go by the driver version instead.
  $driver = "$(& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | Select-Object -First 1)".Trim()
  if ($driver -match "^(\d+)\.") {
    $major = [int]$Matches[1]
    if ($major -ge 580) { $driverCuda = [version]"13.0" } elseif ($major -ge 570) { $driverCuda = [version]"12.8" }
    elseif ($major -ge 560) { $driverCuda = [version]"12.6" } else { $driverCuda = [version]"12.0" }
    Write-Host "driver $driver"
  }
}
if (-not $driverCuda) {
  $smiLines | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }
  throw "Couldn't read the driver's CUDA version from nvidia-smi (its output is above)."
}
$gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) -join "; "
Write-Host "$gpu (driver supports CUDA $driverCuda)"
if ($Cuda) { $builds = @($Cuda) }
elseif ($driverCuda -ge [version]"13.0") { $builds = @("cu130", "cu128", "cu126") }
elseif ($driverCuda -ge [version]"12.8") { $builds = @("cu128", "cu126") }
elseif ($driverCuda -ge [version]"12.6") { $builds = @("cu126") }
else { throw "This driver supports CUDA $driverCuda, and current PyTorch needs 12.6 or newer. Update the NVIDIA driver, then run this again." }

# 2. Python environment ---------------------------------------------------------------------------
Step "Python environment (.venv)"
$py = Join-Path $here ".venv\Scripts\python.exe"
if (Test-Path $py) {
  Write-Host "already there"
} else {
  $candidates = @()
  if ($Python) { $candidates += , @($Python) }
  if (Get-Command py -ErrorAction SilentlyContinue) { foreach ($v in "3.12", "3.11", "3.13", "3.10") { $candidates += , @("py", "-$v") } }
  if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += , @("python") }
  $base = $null
  foreach ($c in $candidates) {
    $exe = $c[0]; $pre = @($c | Select-Object -Skip 1)
    if (Quietly { & $exe @pre -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" }) { $base = $c; break }
  }
  if (-not $base) { throw "No Python 3.10 to 3.13 found. Install Python 3.12 from python.org, then run this again (or pass -Python C:\path\to\python.exe)." }
  $exe = $base[0]; $pre = @($base | Select-Object -Skip 1)
  & $exe @pre -m venv (Join-Path $here ".venv"); Check "Creating .venv"
  Write-Host "created from: $($base -join ' ')"
}
& $py -m pip install --upgrade pip --quiet; Check "Updating pip"

# 3. PyTorch with CUDA ----------------------------------------------------------------------------
Step "PyTorch with CUDA"
$ready = $false
if (Quietly { & $py -c "import torch, torchvision, sys; sys.exit(0 if torch.cuda.is_available() else 1)" }) { $ready = $true }
foreach ($build in $builds) {
  if ($ready) { break }
  Write-Host "installing the $build build"
  & $py -m pip install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$build"
  if ($LASTEXITCODE -eq 0 -and (Quietly { & $py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" })) { $ready = $true }
  else { Write-Host "the $build build didn't work with this GPU; trying the next one" -ForegroundColor Yellow }
}
if (-not $ready) { throw "Couldn't install a PyTorch build that can use the GPU." }
& $py -c "import torch; print('PyTorch', torch.__version__, 'on', torch.cuda.get_device_name())"

# 4. Other packages -------------------------------------------------------------------------------
Step "Other packages"
& $py -m pip install -r (Join-Path $here "requirements.txt") --quiet; Check "Installing requirements.txt"
Write-Host "done"

# 5. The concept-lens repository ------------------------------------------------------------------
Step "concept-lens repository"
$repo = Join-Path $here "concept-lens"
$enclosing = Split-Path (Split-Path $here -Parent) -Parent  # remaster branch: this folder is <repo>\remaster\generation
if ((Test-Path (Join-Path $enclosing "data_generation")) -and (Test-Path (Join-Path $enclosing "server"))) { $repo = $enclosing }
if (Test-Path (Join-Path $repo "data_generation")) {
  Write-Host "already at $repo"
} elseif (Get-Command git -ErrorAction SilentlyContinue) {
  & git clone --depth 1 https://github.com/swj0418/concept-lens.git $repo; Check "git clone"
} else {
  $zip = Join-Path $env:TEMP "concept-lens-main.zip"
  Invoke-WebRequest "https://github.com/swj0418/concept-lens/archive/refs/heads/main.zip" -OutFile $zip -ErrorAction Stop
  Expand-Archive $zip -DestinationPath $here -Force -ErrorAction Stop
  Rename-Item (Join-Path $here "concept-lens-main") "concept-lens" -ErrorAction Stop
  Remove-Item $zip
  Write-Host "downloaded to $repo"
}

# 6. StyleGAN checkpoint (NVIDIA's official pickles; names as in data_generation/_helpers.py) -----
Step "StyleGAN checkpoint for $Domain"
$sg2 = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/"
$sg3 = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"
$checkpoints = @{
  s2_ffhq256 = "stylegan2-ffhq-256x256.pkl"; s2_celeba256 = "stylegan2-celebahq-256x256.pkl"
  s2_ffhq1024 = "stylegan2-ffhq-1024x1024.pkl"; s2_metfaces1024 = "stylegan2-metfaces-1024x1024.pkl"
  s2_cat512 = "stylegan2-afhqcat-512x512.pkl"; s2_wild512 = "stylegan2-afhqwild-512x512.pkl"
  s3t_ffhq1024 = "stylegan3-t-ffhq-1024x1024.pkl"; s3t_wild512 = "stylegan3-t-afhqv2-512x512.pkl"
  s3t_metfaces1024 = "stylegan3-t-metfaces-1024x1024.pkl"; s3r_ffhq1024 = "stylegan3-r-ffhq-1024x1024.pkl"
  s3r_wild512 = "stylegan3-r-afhqv2-512x512.pkl"; s3r_metfaces1024 = "stylegan3-r-metfaces-1024x1024.pkl"
}
if (-not $checkpoints.ContainsKey($Domain)) { throw "No official checkpoint known for $Domain. Put it in concept-lens\resources\checkpoints yourself." }
$name = $checkpoints[$Domain]
$folder = Join-Path $repo "resources\checkpoints"
New-Item -ItemType Directory -Force $folder | Out-Null
$target = Join-Path $folder $name
if ((Test-Path $target) -and (Get-Item $target).Length -gt 10MB) {
  Write-Host "already at $target"
} else {
  $url = $(if ($name.StartsWith("stylegan3")) { $sg3 } else { $sg2 }) + $name
  Write-Host "downloading $url"
  & curl.exe -L --fail --retry 3 -o "$target.part" $url
  if ($LASTEXITCODE -ne 0) { Invoke-WebRequest $url -OutFile "$target.part" -ErrorAction Stop }
  Move-Item -Force "$target.part" $target
  Write-Host ("{0:N0} MB" -f ((Get-Item $target).Length / 1MB))
}

# 7. CLIP weights (ViT-H-14 DFN5B for features; OpenAI ViT-B-32 and ViT-B-16 for the svm and clip methods) --
Step "CLIP weights (about 5 GB the first time)"
& $py -c "import open_clip; [open_clip.create_model_and_transforms(m, pretrained=p) for m, p in [('ViT-H-14-quickgelu', 'dfn5b'), ('ViT-B-32-quickgelu', 'openai'), ('ViT-B-16-quickgelu', 'openai')]]; print('ready')"
Check "Downloading CLIP weights"

# 8. A one-image test of the generator -----------------------------------------------------------
Step "Test render"
& $py (Join-Path $here "generate.py") --domain $Domain --check; Check "Test render"

if (-not (Get-Command node -ErrorAction SilentlyContinue) -and -not (Get-Command deno -ErrorAction SilentlyContinue)) {
  Write-Host "`nOptional: install Node.js (nodejs.org) or Deno (deno.com) to run the parity check (run.ps1 --parity)." -ForegroundColor Yellow
}
Write-Host "`nSetup finished. Next: powershell -ExecutionPolicy Bypass -File .\run.ps1" -ForegroundColor Green
exit 0
