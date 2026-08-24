[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$isAdministrator = ([Security.Principal.WindowsPrincipal]::new(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdministrator) {
    Write-Host '[py-launcher] Requesting administrator permission...'
    $arguments = @(
        '-NoProfile'
        '-ExecutionPolicy', 'Bypass'
        '-File', $PSCommandPath
    )
    $process = Start-Process powershell.exe -Verb RunAs -Wait -PassThru -ArgumentList $arguments
    exit $process.ExitCode
}

$pythonVersion = '3.13.15'
$installerName = "python-$pythonVersion-amd64.exe"
$installerPath = Join-Path $env:TEMP $installerName
$downloadUrl = "https://www.python.org/ftp/python/$pythonVersion/$installerName"

try {
    Write-Host "[py-launcher] Downloading $downloadUrl ..."
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath

    Write-Host '[py-launcher] Installing py.exe silently...'
    $process = Start-Process $installerPath -Wait -PassThru -ArgumentList @(
        '/quiet'
        'LauncherOnly=1'
        'InstallLauncherAllUsers=1'
    )

    if ($process.ExitCode -ne 0) {
        throw "Python launcher installer exited with code $($process.ExitCode)."
    }

    $launcherPath = Join-Path $env:WINDIR 'py.exe'
    if (-not (Test-Path $launcherPath)) {
        throw "Installation completed, but $launcherPath was not found."
    }

    Write-Host "[py-launcher] Installed: $launcherPath"
    & $launcherPath --version
    & $launcherPath -0p
}
finally {
    if (Test-Path $installerPath) {
        Remove-Item -LiteralPath $installerPath -Force
    }
}
