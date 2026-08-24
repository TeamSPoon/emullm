# ensure_winget.ps1
# Ensures winget (Windows Package Manager) is installed and resolvable,
# installing/updating the Microsoft App Installer package over the network
# if needed. Used by install_python.cmd and update_windows_tools.cmd so the
# same fix only has to live in one place.
#
# On success, prints the full path to winget.exe as the ONLY line on
# stdout (all human-readable progress/diagnostics go to stderr instead, so
# callers can safely capture stdout with `for /f` without picking up noise).
# Exits 0 on success, 1 on failure.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Write-Status {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
}

function Get-WingetPath {
    $cmd = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    # winget.exe lives in a per-user folder that Windows normally puts on
    # PATH automatically. Fall back to checking there directly in case
    # that never happened (see the PATH fix-up below).
    $candidate = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\winget.exe'
    if (Test-Path $candidate) { return $candidate }
    return $null
}

$winget = Get-WingetPath
if (-not $winget) {
    Write-Status '[ensure_winget] winget was not found. Installing Microsoft App Installer...'
    $bundle = Join-Path $env:TEMP 'Microsoft.DesktopAppInstaller.msixbundle'
    try {
        Invoke-WebRequest -Uri 'https://github.com/microsoft/winget-cli/releases/latest/download/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle' -OutFile $bundle
        Add-AppxPackage -Path $bundle
    } catch {
        # Add-AppxPackage throws if e.g. an equal-or-newer version is
        # already deployed (HRESULT 0x80073D06). That is not a real
        # failure -- App Installer is already present -- so only treat
        # this as fatal if the package genuinely isn't installed.
        $installed = Get-AppxPackage -Name Microsoft.DesktopAppInstaller -ErrorAction SilentlyContinue
        if (-not $installed) {
            Write-Status "[ensure_winget] Add-AppxPackage failed: $($_.Exception.Message)"
            throw
        }
        Write-Status "[ensure_winget] App Installer is already up to date ($($installed.Version)); continuing."
    } finally {
        Remove-Item -LiteralPath $bundle -Force -ErrorAction SilentlyContinue
    }
    $winget = Get-WingetPath
}

if (-not $winget) {
    Write-Status '[ensure_winget] App Installer was installed, but winget.exe still cannot be found.'
    Write-Status '[ensure_winget] Install or update "App Installer" from the Microsoft Store, then try again.'
    exit 1
}

# winget.exe's folder is normally added to the user's PATH by Windows when
# App Installer is registered. On some systems that never happens, so
# winget works only via a full path. Fix that permanently (one time) so
# future terminals can just run `winget` directly.
$wingetDir = Split-Path -Parent $winget
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$pathEntries = @()
if ($userPath) { $pathEntries = $userPath -split ';' }
if ($pathEntries -notcontains $wingetDir) {
    $joined = if ($userPath) { $userPath.TrimEnd(';') + ';' + $wingetDir } else { $wingetDir }
    [Environment]::SetEnvironmentVariable('Path', $joined, 'User')
    Write-Status "[ensure_winget] Added '$wingetDir' to your permanent PATH (new terminals will see 'winget' directly)."
}

Write-Output $winget
exit 0
