/**
 * Table-driven triage map covering all supported disease classes.
 * Any diagnosis not in either set is treated as "low risk" by default.
 */
const HIGH_RISK_DIAGNOSES = new Set(["DR", "CRVO", "BRVO", "CRAO", "AMD_WET", "CNV"]);
const MODERATE_RISK_DIAGNOSES = new Set(["AMD_DRY", "CSR", "MH", "ERM", "PED", "VMT", "MNV"]);

/**
 * Derives a human-readable triage result from a scan object.
 *
 * @param {object|null} scan
 * @returns {{ label: string, tone: string, confidence: string, action: string }}
 */
export function riskFromScan(scan) {
  if (!scan || scan.status !== "completed") {
    return { label: "No active scan", tone: "neutral", confidence: "N/A", action: "Upload scan" };
  }
  const confidencePct = `${Math.round(scan.confidence * 100)}%`;
  if (HIGH_RISK_DIAGNOSES.has(scan.diagnosis)) {
    return { label: "High risk", tone: "danger", confidence: confidencePct, action: "Specialist review required" };
  }
  if (MODERATE_RISK_DIAGNOSES.has(scan.diagnosis) || scan.confidence < 0.7) {
    return { label: "Ambiguous", tone: "warning", confidence: confidencePct, action: "Send to human audit" };
  }
  return { label: "Low risk", tone: "safe", confidence: confidencePct, action: "Clinician sample review" };
}
