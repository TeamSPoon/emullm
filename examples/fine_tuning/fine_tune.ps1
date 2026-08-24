[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Model,

    [string]$TrainingFile = "$PSScriptRoot\training.jsonl",

    [switch]$Wait,

    [ValidateRange(5, 3600)]
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$baseUrl = 'https://api.openai.com/v1'
$apiKey = $env:OPENAI_API_KEY

if (-not $apiKey) {
    throw 'Set OPENAI_API_KEY before running this example.'
}

$trainingPath = (Resolve-Path -LiteralPath $TrainingFile).Path
if ([IO.Path]::GetExtension($trainingPath) -ne '.jsonl') {
    throw 'The training file must have a .jsonl extension.'
}

Write-Host "Uploading $trainingPath ..."
$uploadJson = & curl.exe --fail-with-body --silent --show-error `
    "$baseUrl/files" `
    -H "Authorization: Bearer $apiKey" `
    -F 'purpose=fine-tune' `
    -F "file=@$trainingPath"

if ($LASTEXITCODE -ne 0) {
    throw "File upload failed with curl exit code $LASTEXITCODE."
}

$uploaded = $uploadJson | ConvertFrom-Json
Write-Host "Uploaded file ID: $($uploaded.id)"

$headers = @{
    Authorization = "Bearer $apiKey"
    'Content-Type' = 'application/json'
}
$body = @{
    model = $Model
    training_file = $uploaded.id
    method = @{ type = 'supervised' }
} | ConvertTo-Json -Depth 5

Write-Host 'Creating supervised fine-tuning job...'
$job = Invoke-RestMethod -Method Post -Uri "$baseUrl/fine_tuning/jobs" -Headers $headers -Body $body
$job | ConvertTo-Json -Depth 10

if (-not $Wait) {
    Write-Host "Retrieve it later with:"
    Write-Host "Invoke-RestMethod -Headers @{ Authorization = 'Bearer `$env:OPENAI_API_KEY' } -Uri '$baseUrl/fine_tuning/jobs/$($job.id)'"
    exit 0
}

do {
    Start-Sleep -Seconds $PollSeconds
    $job = Invoke-RestMethod -Method Get -Uri "$baseUrl/fine_tuning/jobs/$($job.id)" -Headers $headers
    Write-Host "$($job.id): $($job.status)"
} while ($job.status -notin @('succeeded', 'failed', 'cancelled'))

$job | ConvertTo-Json -Depth 10
if ($job.status -ne 'succeeded') {
    exit 1
}

Write-Host "Fine-tuned model: $($job.fine_tuned_model)"
