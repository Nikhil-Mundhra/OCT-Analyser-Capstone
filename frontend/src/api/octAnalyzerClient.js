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

/** Maximum time to wait for a scan to move from pending/processing to a terminal state. */
const MAX_POLL_DURATION_MS = 120_000; // 2 minutes

/**
 * Uploads an OCT/OCTA scan and polls for completion using the background job queue.
 *
 * @param {File} file
 * @param {function|null} onProgress - callback receiving a detail string each poll tick
 * @returns {Promise<object>}
 * @throws {Error} if upload fails, processing fails, or polling exceeds MAX_POLL_DURATION_MS
 */
export async function createScan(file, onProgress = null) {
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
    if (!res.ok) throw new Error(`Failed to upload scan (HTTP ${res.status})`);
    return res.json();
  });

  // Step 2: Poll for completion with exponential backoff and a hard timeout
  const scanId = scanRecord.scan_id;
  let delay = 1000;
  const pollDeadline = Date.now() + MAX_POLL_DURATION_MS;
  const controller = new AbortController();

  while (scanRecord.status === "pending" || scanRecord.status === "processing") {
    if (Date.now() >= pollDeadline) {
      controller.abort();
      throw new Error(`Scan processing timed out after ${MAX_POLL_DURATION_MS / 1000}s. Please try again.`);
    }

    if (onProgress && scanRecord.detail) {
      onProgress(scanRecord.detail);
    }

    await new Promise(resolve => setTimeout(resolve, delay));

    const pollRes = await fetch(apiUrl(`/api/scans/${scanId}`), { signal: controller.signal });
    if (!pollRes.ok) {
      throw new Error(`Failed to poll scan status (HTTP ${pollRes.status})`);
    }
    scanRecord = await pollRes.json();
    delay = Math.min(delay * 1.5, 5000); // max delay of 5 seconds between polls
  }

  if (scanRecord.status === "failed") {
    throw new Error(scanRecord.detail || "Scan processing failed");
  }

  // The local MVP pipeline already formats mostly to the expected structure
  // We no longer need a separate segmentReq since analyze_volume does it all!
  const normalized = normalizeScanResult(scanRecord, scanRecord.segmentation);
  normalized.localImageUrl = localImageUrl;
  return normalized;
}

/**
 * Runs single or multiple segmentation & detection models from the 5-Model Suite.
 *
 * @param {File} file
 * @param {string} modelId - "all" | "model1" | "model2" | "model3" | "model4" | "model5"
 * @param {number} scoreThreshold - Confidence threshold for Model 5 detector (default 0.5)
 * @returns {Promise<object>}
 */
export async function runModelSuite(file, modelId = "all", scoreThreshold = 0.5) {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  form.append("score_threshold", scoreThreshold.toString());

  const response = await fetch(apiUrl("/api/segment_suite"), {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    throw new Error(`Failed to execute Segmentation 5-Model Suite (HTTP ${response.status})`);
  }

  return await response.json();
}

export function normalizeScanResult(scan, segmentation = null) {
  if (!scan || typeof scan !== "object") {
    return scan;
  }

  // Local /api/scans returns the full MVP payload which might be a bit different from HF
  // But we adapt it gracefully:
  const classification = scan.classification || {};
  const diagnosis = classification.diagnosis || scan.diagnosis || "NORMAL";

  // Use crypto.randomUUID() for a reliable fallback — never Math.random()
  const fallbackId = typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

  return {
    scan_id: scan.scan_id || scan.id || fallbackId,
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

