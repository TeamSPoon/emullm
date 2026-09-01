import fs from "node:fs";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

const configPath = process.argv[2];
if (!configPath) {
  throw new Error("copilot_sdk_bridge requires a servant config path");
}
const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
delete process.env.COPILOT_AGENT_SESSION_ID;

const sdk = await import(pathToFileURL(config.copilot_sdk_path).href);
const { CopilotClient, RuntimeConnection, approveAll } = sdk;
const runtimeArgs = [];
if (!config.enable_builtin_mcps) runtimeArgs.push("--disable-builtin-mcps");
if (config.max_ai_credits != null) {
  runtimeArgs.push("--max-ai-credits", String(config.max_ai_credits));
}

const runtimeEnv = { ...process.env };
delete runtimeEnv.COPILOT_AGENT_SESSION_ID;
const connection = RuntimeConnection.forStdio({
  path: config.copilot_runtime_path,
  args: runtimeArgs,
  env: runtimeEnv,
});
const client = new CopilotClient({
  connection,
  workingDirectory: config.resolved_cwd,
  useLoggedInUser: true,
  logLevel: "none",
  enableRemoteSessions: false,
});
const sessionConfig = {
  model: config.selected_model,
  reasoningEffort: config.reasoning_effort ?? undefined,
  contextTier: config.context,
  workingDirectory: config.resolved_cwd,
  skipCustomInstructions: !config.load_custom_instructions,
  availableTools: config.allow_all ? undefined : [],
  onPermissionRequest: config.allow_all ? approveAll : undefined,
};

function emit(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

await client.start();
let session;
let resumed = false;
try {
  session = await client.resumeSession(config.session_id, sessionConfig);
  resumed = true;
} catch (resumeError) {
  try {
    session = await client.createSession({
      ...sessionConfig,
      sessionId: config.session_id,
    });
  } catch (createError) {
    throw new Error(
      `could not resume or create Copilot session ${config.session_id}: ` +
      `resume=${resumeError}; create=${createError}`,
    );
  }
}

emit({
  type: "ready",
  session_id: session.sessionId,
  model: config.selected_model,
  resumed,
  pid: process.pid,
  runtime_pid: client.cliProcess?.pid ?? null,
});

let active = null;
let switchingModel = false;
let currentModel = config.selected_model;
let shuttingDown = false;

async function handleRequest(message) {
  if (active !== null || switchingModel) {
    emit({
      type: "error",
      id: message.id,
      error: `Copilot session is busy with request ${active.id}`,
    });
    return;
  }
  const current = {
    id: String(message.id),
    cancelled: false,
    startedAt: Date.now(),
  };
  active = current;
  try {
    const response = await session.sendAndWait(
      {
        prompt: String(message.prompt ?? ""),
        attachments: Array.isArray(message.attachments) ? message.attachments : [],
      },
      Number(message.timeout_ms ?? 60000),
    );
    if (current.cancelled) return;
    const content = response?.data?.content;
    if (typeof content !== "string" || content.length === 0) {
      throw new Error("Copilot SDK returned no assistant message");
    }
    emit({
      type: "response",
      id: current.id,
      content,
      duration_ms: Date.now() - current.startedAt,
    });
  } catch (error) {
    if (!current.cancelled) {
      emit({
        type: "error",
        id: current.id,
        error: error instanceof Error ? error.message : String(error),
        duration_ms: Date.now() - current.startedAt,
      });
    }
  } finally {
    if (active === current) active = null;
  }
}

async function handleSetModel(message) {
  const id = String(message.id ?? "");
  const model = String(message.model ?? "").trim();
  if (!model) {
    emit({ type: "model_change_error", id, error: "model is required" });
    return;
  }
  if (active !== null || switchingModel) {
    emit({ type: "model_change_error", id, error: "Copilot session is busy" });
    return;
  }
  if (model === currentModel) {
    emit({ type: "model_changed", id, model, unchanged: true });
    return;
  }
  switchingModel = true;
  try {
    await session.setModel(model, {
      reasoningEffort: message.reasoning_effort ?? undefined,
      contextTier: message.context ?? config.context,
    });
    currentModel = model;
    emit({ type: "model_changed", id, model });
  } catch (error) {
    emit({
      type: "model_change_error",
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    switchingModel = false;
  }
}

async function handleCancel(message) {
  const requestId = String(message.id ?? "");
  const cancelled = active !== null && active.id === requestId;
  if (cancelled) {
    active.cancelled = true;
    await session.abort();
  }
  emit({ type: "cancelled", id: requestId, cancelled });
}

async function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  if (active !== null) {
    active.cancelled = true;
    try {
      await session.abort();
    } catch {
      // Continue shutting down the owned runtime.
    }
  }
  try {
    await session.disconnect();
  } finally {
    const errors = await client.stop();
    emit({ type: "stopped", errors: errors.map((error) => String(error)) });
  }
}

const input = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});
input.on("line", (line) => {
  if (!line.trim()) return;
  let message;
  try {
    message = JSON.parse(line);
  } catch (error) {
    emit({ type: "error", id: null, error: `invalid JSON request: ${error}` });
    return;
  }
  if (message.type === "request") {
    void handleRequest(message);
  } else if (message.type === "set_model") {
    void handleSetModel(message);
  } else if (message.type === "cancel") {
    void handleCancel(message);
  } else if (message.type === "shutdown") {
    void shutdown().finally(() => process.exit(0));
  } else {
    emit({ type: "error", id: message.id ?? null, error: `unsupported message type ${message.type}` });
  }
});
input.on("close", () => {
  void shutdown().finally(() => process.exit(0));
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    void shutdown().finally(() => process.exit(0));
  });
}
if (process.platform === "win32") {
  process.on("SIGBREAK", () => {
    void shutdown().finally(() => process.exit(0));
  });
}
