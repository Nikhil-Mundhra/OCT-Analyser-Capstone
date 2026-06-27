const queryApiBase = globalThis.location
  ? new URLSearchParams(globalThis.location.search).get("apiBase")
  : "";
const DEFAULT_API_BASE =
  globalThis.location?.port && globalThis.location.port !== "8000"
    ? `${globalThis.location.protocol}//${globalThis.location.hostname}:8000`
    : "";

export const OCT_ANALYZER_API_BASE = (
  globalThis.OCT_ANALYZER_API_BASE || queryApiBase || DEFAULT_API_BASE
).replace(/\/$/, "");

/**
 * Uploads an OCT/OCTA scan through the backend API contract.
 *
 * @param {File} file
 * @returns {Promise<object>}
 */
export async function createScan(file) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(apiUrl("/api/scans"), {
    method: "POST",
    body: form,
  });
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Upload failed");
  }

  return normalizeScanResult(payload);
}

export function normalizeScanResult(scan) {
  if (!scan || typeof scan !== "object") {
    return scan;
  }

  return {
    ...scan,
    previews: normalizePreviewMap(scan.previews),
    ipnv2: scan.ipnv2
      ? {
          ...scan.ipnv2,
          previews: normalizePreviewMap(scan.ipnv2.previews),
        }
      : scan.ipnv2,
  };
}

function normalizePreviewMap(previews = {}) {
  return Object.fromEntries(
    Object.entries(previews).map(([key, value]) => [
      key, 
      Array.isArray(value) ? value.map(apiUrl) : apiUrl(value)
    ])
  );
}

function apiUrl(path) {
  if (!path || /^https?:\/\//i.test(path)) {
    return path;
  }
  return `${OCT_ANALYZER_API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}
