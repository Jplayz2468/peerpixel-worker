/* Thin wrapper over the PeerPixel HTTP API. */
import { API, readConfig } from "./config.mjs";

export class ApiError extends Error {
  constructor(status, code, body) {
    super(`${code} (${status})`);
    this.status = status;
    this.code = code;
    this.body = body;
  }
}

async function request(path, { method = "GET", json, body, auth = true, headers = {} } = {}) {
  const config = readConfig();
  const response = await fetch(`${API}${path}`, {
    method,
    headers: {
      ...(json ? { "content-type": "application/json" } : {}),
      ...(auth && config.token ? { authorization: `Bearer ${config.token}` } : {}),
      ...headers,
    },
    body: json ? JSON.stringify(json) : body,
  });
  const text = await response.text();
  let parsed;
  try { parsed = text ? JSON.parse(text) : {}; } catch { parsed = { raw: text }; }
  if (!response.ok) throw new ApiError(response.status, parsed.error || "http_error", parsed);
  return parsed;
}

export const pair = (code, info) =>
  request("/api/pair/claim", { method: "POST", json: { code, ...info }, auth: false });

export const submitBench = (ms, accelerator) =>
  request("/api/device/bench", { method: "POST", json: { ms, accelerator } });

export const submitResult = (jobId, jpeg) =>
  request(`/api/device/job/${jobId}/result`, {
    method: "POST",
    body: jpeg,
    headers: { "content-type": "image/jpeg" },
  });

export const pool = () => request("/api/pool", { auth: false });
