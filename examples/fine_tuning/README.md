# Fine-tuning API examples

These examples target the real OpenAI API and can incur API charges. Against
emullm, file upload and job creation are useful for integration testing: the
server validates UTF-8 JSONL, persists the file and job, and exposes job events
and checkpoints. It then returns `status: "failed"` with
`error.code: "training_not_available"` because emullm does not train models.

The examples deliberately require a model argument because model support and
availability can change. Choose a base model that the current fine-tuning
documentation and your API project both identify as fine-tunable.

## Training data

`training.jsonl` is a deliberately tiny format demonstration, not a useful
production dataset. Every line is one JSON object containing a chat example,
and the desired answer is the assistant message. Build a larger, reviewed
dataset and keep separate evaluation examples before spending money on a job.

## Python SDK example

Install the current SDK and set your key in PowerShell:

```powershell
python -m pip install --upgrade openai
$env:OPENAI_API_KEY = "your-key"
```

Upload the sample and create a supervised job:

```powershell
python .\examples\fine_tuning\fine_tune.py create `
  --model YOUR_SUPPORTED_BASE_MODEL `
  --training-file .\examples\fine_tuning\training.jsonl
```

Copy the returned `ftjob-...` ID, then inspect or wait for it:

```powershell
python .\examples\fine_tuning\fine_tune.py status ftjob-YOUR_JOB_ID
python .\examples\fine_tuning\fine_tune.py wait ftjob-YOUR_JOB_ID
```

After the job succeeds, copy its `ft:...` model ID and use it:

```powershell
python .\examples\fine_tuning\fine_tune.py chat `
  --model "ft:YOUR_FINE_TUNED_MODEL_ID" `
  "My delivery is a week late"
```

Cancel an unwanted running job with:

```powershell
python .\examples\fine_tuning\fine_tune.py cancel ftjob-YOUR_JOB_ID
```

## PowerShell REST example

The PowerShell version uses `curl.exe` for the multipart file upload and
`Invoke-RestMethod` for job creation and polling:

```powershell
$env:OPENAI_API_KEY = "your-key"
.\examples\fine_tuning\fine_tune.ps1 `
  -Model YOUR_SUPPORTED_BASE_MODEL `
  -Wait
```

If local policy blocks scripts, use a process-scoped bypass:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\examples\fine_tuning\fine_tune.ps1 `
  -Model YOUR_SUPPORTED_BASE_MODEL `
  -Wait
```

Official references:

- [Upload a file](https://developers.openai.com/api/reference/typescript/resources/files/methods/create)
- [Fine-tuning API](https://developers.openai.com/api/reference/resources/fine_tuning)
- [List job checkpoints](https://developers.openai.com/api/reference/python/resources/fine_tuning/subresources/jobs/subresources/checkpoints/methods/list)
