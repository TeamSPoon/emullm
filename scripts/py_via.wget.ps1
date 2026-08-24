$installer = "$env:TEMP\python-3.13.15-amd64.exe"

Invoke-WebRequest `
  -Uri "https://www.python.org/ftp/python/3.13.15/python-3.13.15-amd64.exe" `
  -OutFile $installer

Start-Process $installer -Wait -ArgumentList `
  "/quiet", `
  "LauncherOnly=1", `
  "InstallLauncherAllUsers=1"