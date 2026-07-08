const queryApiBase = globalThis.location
  ? new URLSearchParams(globalThis.location.search).get("apiBase")
  : "";
const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export const OCT_ANALYZER_API_BASE = (
  globalThis.OCT_ANALYZER_API_BASE || queryApiBase || DEFAULT_API_BASE
).replace(/\/$/, "");

// Segmentation endpoint:
//   - Local dev:  set NEXT_PUBLIC_SEGMENTATION_API_URL=http://127.0.0.1:8000 in frontend/.env.local
//   - Production: set NEXT_PUBLIC_SEGMENTATION_API_URL=https://nmundhra-oct-segmentation-model.hf.space in frontend/.env.production
//   - Fallback:   same host as the backend API (local /predict endpoint)
export const SEGMENTATION_API_BASE = (
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_SEGMENTATION_API_URL) ||
  globalThis.SEGMENTATION_API_BASE ||
  OCT_ANALYZER_API_BASE
).replace(/\/$/, "");

/**
 * Uploads an OCT/OCTA scan and polls for completion using the background job queue.
 *
 * @param {File} file
 * @returns {Promise<object>}
 */
export async function createScan(file) {
  const form = new FormData();
  form.append("file", file);

  const localImageUrl = await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });

  // Step 1: Submit scan to background queue
  let scanRecord = await fetch(apiUrl("/api/scans"), {
    method: "POST",
    body: form,
  }).then(res => {
    if (!res.ok) throw new Error("Failed to upload scan");
    return res.json();
  });

  // Step 2: Poll for completion
  const scanId = scanRecord.scan_id;
  while (scanRecord.status === "pending" || scanRecord.status === "processing") {
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
    scanRecord = await fetch(apiUrl(`/api/scans/${scanId}`)).then(res => res.json());
  }

  if (scanRecord.status === "failed") {
    throw new Error(scanRecord.detail || "Scan processing failed");
  }

  // Segment via API for visualization
  const segmentReq = fetch(`${SEGMENTATION_API_BASE}/predict`, {
    method: "POST",
    body: form,
  }).then(res => res.ok ? res.json() : null).catch(() => null);

  const segmentPayload = await segmentReq;

  // The local MVP pipeline already formats mostly to the expected structure
  const normalized = normalizeScanResult(scanRecord, segmentPayload);
  normalized.localImageUrl = localImageUrl;
  return normalized;
}

export function normalizeScanResult(scan, segmentation = null) {
  if (!scan || typeof scan !== "object") {
    return scan;
  }

  // Local /api/scans returns the full MVP payload which might be a bit different from HF
  // But we adapt it gracefully:
  const classification = scan.classification || {};
  const diagnosis = classification.diagnosis || scan.diagnosis || "NORMAL";
  
  return {
    status: scan.status || "completed",
    diagnosis: diagnosis,
    confidence: scan.confidence || classification.confidence || 0.0,
    level1: classification.level1 || scan.level1 || {},
    level2: classification.level2 || scan.level2 || {},
    level3: classification.level3 || scan.level3 || {},
    gradcams: scan.gradcams || {},
    previews: normalizePreviewMap(scan.previews),
    segmentation: segmentation || null,
    ipnv2: null
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
