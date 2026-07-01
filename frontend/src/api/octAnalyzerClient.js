const queryApiBase = globalThis.location
  ? new URLSearchParams(globalThis.location.search).get("apiBase")
  : "";
const DEFAULT_API_BASE = "https://nmundhra-oct-image-classifier-model.hf.space";

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

  const response = await fetch(apiUrl("/predict"), {
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

  // Map Hugging Face Pipeline output to the legacy React frontend schema
  const isAbnormal = scan.level1_prediction === "ABNORMAL";
  
  let diagnosis = "NORMAL";
  if (isAbnormal && scan.level2_prediction) {
    if (scan.level2_prediction === "Macular_Degeneration") diagnosis = "AMD";
    else if (scan.level2_prediction === "Diabetic_Complications") diagnosis = "DR";
    else diagnosis = scan.level2_prediction;
  }

  return {
    status: "completed",
    diagnosis: diagnosis,
    confidence: isAbnormal ? scan.level2_confidence : scan.level1_confidence,
    level1: {
      prediction: scan.level1_prediction,
      confidence: scan.level1_confidence
    },
    level2: {
      prediction: scan.level2_prediction,
      confidence: scan.level2_confidence
    },
    previews: {},
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
