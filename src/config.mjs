/* Where this install keeps its identity.

   The device token is the only secret here and it is written 0600. It is not a
   password: it authorises one machine to render, nothing else. Losing it costs
   a re-pair, and revoking it is a row in the devices table. */

import { mkdirSync, readFileSync, writeFileSync, existsSync, chmodSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const DIR = process.env.PEERPIXEL_HOME || join(homedir(), ".peerpixel");
const FILE = join(DIR, "config.json");

export const API = (process.env.PEERPIXEL_API || "https://peerpixel.cc").replace(/\/+$/, "");

export function readConfig() {
  if (!existsSync(FILE)) return {};
  try { return JSON.parse(readFileSync(FILE, "utf8")); } catch { return {}; }
}

export function writeConfig(patch) {
  mkdirSync(DIR, { recursive: true });
  const next = { ...readConfig(), ...patch };
  writeFileSync(FILE, JSON.stringify(next, null, 2));
  try { chmodSync(FILE, 0o600); } catch { /* windows */ }
  return next;
}

export const configPath = () => FILE;
