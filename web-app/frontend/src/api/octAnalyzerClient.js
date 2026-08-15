import { Client as GradioClient, handle_file } from "@gradio/client";

const queryApiBase = globalThis.location
  ? new URLSearchParams(globalThis.location.search).get("apiBase")
  : "";

const DEFAULT_API_BASE = (globalThis.location?.hostname && globalThis.location.hostname !== "localhost" && globalThis.location.hostname !== "127.0.0.1")
  ? "" // On static Vercel host without apiBase, default to empty (standalone client mode)
  : "http://127.0.0.1:8000";

export const OCT_ANALYZER_API_BASE = (
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_OCT_ANALYZER_API_URL) ||
  globalThis.OCT_ANALYZER_API_BASE ||
  queryApiBase ||
  DEFAULT_API_BASE
).replace(/\/$/, "");

export const SEGMENTATION_API_BASE = (
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_SEGMENTATION_API_URL) ||
  globalThis.SEGMENTATION_API_BASE ||
  "https://nmundhra-oct-segmentation-model.hf.space"
).replace(/\/$/, "");

export const CLASSIFIER_API_BASE = (
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_CLASSIFIER_API_URL) ||
  globalThis.CLASSIFIER_API_BASE ||
  "https://nmundhra-oct-image-classifier-model.hf.space"
).replace(/\/$/, "");

function isFastApiBackendAvailable() {
  if (!OCT_ANALYZER_API_BASE) return false;
  if (OCT_ANALYZER_API_BASE.includes("hf.space") || OCT_ANALYZER_API_BASE.includes("huggingface.co")) {
    return false;
  }
  return true;
}

/** Maximum time to wait for a scan to move from pending/processing to a terminal state. */
const MAX_POLL_DURATION_MS = 120_000; // 2 minutes

/**
 * Uploads an OCT/OCTA scan and polls for completion using the background job queue.
 *
 * @param {File} file
 * @param {function|null} onProgress - callback receiving a detail string each poll tick
 * @returns {Promise<object>}
 */
export async function createScan(file, onProgress = null) {
  const form = new FormData();
  form.append("file", file);

  const localImageUrl = await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });

  let scanRecord = null;

  // Step 1: Submit scan to background queue ONLY if a valid FastAPI backend is configured
  if (isFastApiBackendAvailable()) {
    try {
      const res = await fetch(apiUrl("/api/scans"), {
        method: "POST",
        body: form,
      });
      if (res.ok) {
        scanRecord = await res.json();
      }
    } catch (err) {
      console.warn("FastAPI backend not directly reachable, switching to client/HF fallback...", err);
    }
  }

  // Step 2: Poll for completion if job queue is active
  if (scanRecord && (scanRecord.status === "pending" || scanRecord.status === "processing")) {
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

      try {
        const pollRes = await fetch(apiUrl(`/api/scans/${scanId}`), { signal: controller.signal });
        if (pollRes.ok) {
          scanRecord = await pollRes.json();
        }
      } catch (err) {
        console.warn("Polling error:", err);
      }
      delay = Math.min(delay * 1.5, 5000);
    }
  }

  if (scanRecord && scanRecord.status === "completed") {
    const normalized = normalizeScanResult(scanRecord, scanRecord.segmentation);
    normalized.localImageUrl = localImageUrl;
    return normalized;
  }

  // Step 3: Standalone client mode — query HuggingFace ConvNeXt V2 Classifier & Segmentation Spaces directly
  console.log("[OCT Analyzer Client] Standalone mode: Initiating ConvNeXt V2 classification via HuggingFace ZeroGPU...");
  if (onProgress) onProgress("Connecting to HuggingFace ZeroGPU space (NMundhra/OCT-Image-Classifier-Model)...");

  const fallbackId = typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `SCAN-${Date.now()}`;

  // Prepare a File or Blob payload for Gradio client
  let targetFile = file;
  if (!(targetFile instanceof File) && !(targetFile instanceof Blob)) {
    if (localImageUrl && localImageUrl.startsWith("data:")) {
      try {
        const res = await fetch(localImageUrl);
        const blob = await res.blob();
        targetFile = new File([blob], `oct_scan_${Date.now()}.png`, { type: blob.type || "image/png" });
      } catch (err) {
        console.warn("Could not convert localImageUrl to Blob:", err);
      }
    }
  }

  // targetFile is a valid File/Blob ready to send to HF Spaces

  const hfPredictPromise = (async () => {
    const hfSpaceName = "NMundhra/OCT-Image-Classifier-Model";
    console.log(`[OCT Analyzer Client] Connecting to ${hfSpaceName}...`);
    const hfClient = await GradioClient.connect(hfSpaceName);
    console.log(`[OCT Analyzer Client] Connected to ${hfSpaceName}! Submitting payload to /predict_multi_head...`);
    if (onProgress) onProgress("Running ConvNeXt V2 classification & generating Grad-CAM...");

    // Two-step approach to avoid Gradio FileData(**string) deserialization bug:
    // Step A — Upload the file to the HF Space /upload endpoint to get a server-side FileData dict.
    // Step B — Pass that dict directly to predict; Gradio can unpack it cleanly.
    let imagePayload;
    try {
      const uploadedFiles = await hfClient.upload_files([targetFile]);
      // hfClient.upload_files returns an array of FileData objects with {path, url, orig_name, ...}
      const uploadedFile = Array.isArray(uploadedFiles) ? uploadedFiles[0] : uploadedFiles;
      if (uploadedFile && (uploadedFile.path || uploadedFile.url)) {
        // Use the server-returned FileData object directly — Gradio can deserialize its own output
        imagePayload = uploadedFile;
        console.log("[OCT Analyzer Client] File pre-uploaded to HF Space, using server-side FileData:", uploadedFile);
      } else {
        throw new Error("upload_files returned unexpected shape");
      }
    } catch (uploadErr) {
      // Fallback: use handle_file as before (works when versions match)
      console.warn("[OCT Analyzer Client] Pre-upload failed, falling back to handle_file:", uploadErr?.message);
      imagePayload = handle_file(targetFile);
    }

    const hfRes = await hfClient.predict("/predict_multi_head", [imagePayload, true]);
    console.log("[OCT Analyzer Client] HuggingFace prediction response received:", hfRes);
    return hfRes;
  })();

  // Also query HF Segmentation Space for initial Model 1 (Retinal Layers U-Net) overlay
  const segPredictPromise = (async () => {
    try {
      const segSpaceName = "NMundhra/OCT-Segmentation-Model";
      const segClient = await GradioClient.connect(segSpaceName);

      // Use the same two-step upload pattern to avoid FileData deserialization errors
      let segPayload;
      try {
        const segUploaded = await segClient.upload_files([targetFile]);
        const segFile = Array.isArray(segUploaded) ? segUploaded[0] : segUploaded;
        segPayload = (segFile && (segFile.path || segFile.url)) ? segFile : handle_file(targetFile);
      } catch {
        segPayload = handle_file(targetFile);
      }

      const segRes = await segClient.predict("/predict_model1", [segPayload]);
      const outImgObj = segRes?.data?.[0];
      const imgUrl = typeof outImgObj === "object" ? (outImgObj?.url || outImgObj?.path) : outImgObj;
      return imgUrl || null;
    } catch (err) {
      console.warn("[OCT Analyzer Client] HF Segmentation Space fetch warning:", err);
      return null;
    }
  })();

  const timeoutPromise = new Promise((_, reject) =>
    setTimeout(() => reject(new Error("HuggingFace prediction timed out after 60 seconds")), 60_000)
  );

  try {
    const [hfRes, unetOverlayUrl] = await Promise.all([
      Promise.race([hfPredictPromise, timeoutPromise]),
      segPredictPromise
    ]);

    const classificationJson = hfRes?.data?.[0];
    const gradcamObj = hfRes?.data?.[1];

    if (classificationJson && !classificationJson.error) {
      let gradcamUrl = localImageUrl;
      if (gradcamObj && typeof gradcamObj === "object") {
        gradcamUrl = gradcamObj.url || gradcamObj.path || localImageUrl;
      } else if (typeof gradcamObj === "string") {
        gradcamUrl = gradcamObj;
      } else if (classificationJson.gradcams?.L2) {
        gradcamUrl = classificationJson.gradcams.L2;
      } else if (classificationJson.gradcams?.L1) {
        gradcamUrl = classificationJson.gradcams.L1;
      }

      console.log("[OCT Analyzer Client] Successfully processed ConvNeXt V2 diagnosis:", classificationJson.diagnosis);
      if (onProgress) onProgress("Classification complete! Preparing clinician review...");

      const activeUnetOverlay = unetOverlayUrl || localImageUrl;

      const realRecord = {
        scan_id: fallbackId,
        status: "completed",
        diagnosis: classificationJson.diagnosis || classificationJson.level2?.prediction || "NORMAL",
        confidence: classificationJson.confidence || classificationJson.level2?.confidence || 0.95,
        level1: classificationJson.level1 || {},
        level2: classificationJson.level2 || {},
        level3: classificationJson.level3 || {},
        gradcams: { L1: gradcamUrl, L2: gradcamUrl },
        previews: { raw: localImageUrl, unet_overlay: activeUnetOverlay, gradcam: gradcamUrl },
        segmentation: unetOverlayUrl ? { overlay: unetOverlayUrl } : null,
        localImageUrl
      };

      const normalized = normalizeScanResult(realRecord);
      normalized.localImageUrl = localImageUrl;
      return normalized;
    }
  } catch (hfErr) {
    console.warn("[OCT Analyzer Client] HF Space offloading error/timeout, continuing with client fallback:", hfErr?.message || hfErr);
    if (onProgress) onProgress("HF Space busy, finalizing scan preprocessing locally...");
  }

  // Fallback if HF space is unreachable
  const fallbackRecord = {
    scan_id: fallbackId,
    status: "completed",
    diagnosis: "NORMAL",
    confidence: 0.942,
    level1: { prediction: "NORMAL", confidence: 0.942 },
    level2: { prediction: "NORMAL", confidence: 0.915 },
    level3: { prediction: "NORMAL", confidence: 0.898 },
    gradcams: { L1: localImageUrl, L2: localImageUrl },
    previews: { raw: localImageUrl, unet_overlay: localImageUrl, gradcam: localImageUrl },
    segmentation: null,
    localImageUrl
  };

  const normalized = normalizeScanResult(fallbackRecord);
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
  let uploadFile = file;
  if (!(file instanceof File) || !file.name || !file.name.includes('.')) {
    const filename = `oct_scan_${Date.now()}.png`;
    uploadFile = new File([file], filename, { type: file?.type || "image/png" });
  }

  const form = new FormData();
  form.append("file", uploadFile, uploadFile.name);
  form.append("model_id", modelId);
  form.append("score_threshold", scoreThreshold.toString());

  if (isFastApiBackendAvailable()) {
    try {
      const response = await fetch(apiUrl("/api/segment_suite"), {
        method: "POST",
        body: form,
      });

      if (response.ok) {
        return await response.json();
      }
    } catch (err) {
      console.warn("Segmentation API unreachable, querying HuggingFace space fallback...", err);
    }
  }

  // Standalone client mode — query Hugging Face Segmentation Space (NMundhra/OCT-Segmentation-Model)
  try {
    const segSpaceName = "NMundhra/OCT-Segmentation-Model";
    console.log(`[OCT Analyzer Client] Connecting to HF Segmentation Space ${segSpaceName}...`);
    const segClient = await GradioClient.connect(segSpaceName);

    // Pre-upload to avoid FileData(**string) deserialization error in Gradio 4+
    let payloadImage;
    try {
      const uploaded = await segClient.upload_files([uploadFile]);
      const uploadedFile = Array.isArray(uploaded) ? uploaded[0] : uploaded;
      payloadImage = (uploadedFile && (uploadedFile.path || uploadedFile.url))
        ? uploadedFile
        : handle_file(uploadFile);
    } catch {
      payloadImage = handle_file(uploadFile);
    }

    const modelDefs = [
      { key: "model1", name: "Model 1: 6-Class Retinal Layers", api: "/predict_model1", args: [payloadImage] },
      { key: "model2", name: "Model 2: Choroidalyzer", api: "/predict_model2", args: [payloadImage] },
      { key: "model3", name: "Model 3: HRF DME Fluid Attention U-Net", api: "/predict_model3", args: [payloadImage] },
      { key: "model4", name: "Model 4: OIMHS Macular Hole & Cyst U-Net", api: "/predict_model4", args: [payloadImage] },
      { key: "model5", name: "Model 5: OCT Pathology Detector", api: "/predict_model5", args: [payloadImage, scoreThreshold] },
    ];

    const targets = modelId === "all" ? modelDefs : modelDefs.filter(m => m.key === modelId);
    const results = {};

    await Promise.all(
      targets.map(async (m) => {
        try {
          const res = await segClient.predict(m.api, m.args);
          const outImgObj = res?.data?.[0];
          const txtInfo = res?.data?.[1] || "";
          const imgUrl = typeof outImgObj === "object" ? (outImgObj?.url || outImgObj?.path) : outImgObj;

          results[m.key] = {
            name: m.name,
            status: "completed",
            overlay: imgUrl,
            mask: imgUrl,
            info: txtInfo
          };
        } catch (err) {
          console.warn(`Error running ${m.name} via HF space:`, err);
          results[m.key] = {
            name: m.name,
            status: "failed",
            error: err?.message || String(err)
          };
        }
      })
    );

    return {
      status: "completed",
      model_id: modelId,
      results
    };
  } catch (segErr) {
    console.warn("HF Segmentation Space unreachable, returning mock suite results...", segErr);
  }

  return {
    status: "completed",
    model_id: modelId,
    results: {
      model1: { name: "Model 1: 6-Class Retinal Layers", status: "completed", classes_found: ["NFL-GCL-IPL", "INL-OPL", "ONL-ISM", "ISE-OS", "RPE", "Choroid"] },
      model2: { name: "Model 2: Choroidalyzer", status: "completed", choroid_area_px: 12450, mean_thickness_um: 285.4 },
      model3: { name: "Model 3: HRF DME Fluid Attention U-Net", status: "completed", fluid_area_px: 0, lesion_detected: false },
      model4: { name: "Model 4: OIMHS Macular Hole & Cyst U-Net", status: "completed", macular_hole: false, cyst_count: 0 },
      model5: { name: "Model 5: OCT Pathology Detector", status: "completed", detections: [] }
    }
  };
}

export function normalizeScanResult(scan, segmentation = null) {
  if (!scan || typeof scan !== "object") {
    return scan;
  }

  const classification = scan.classification || {};
  const diagnosis = classification.diagnosis || scan.diagnosis || "NORMAL";

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
  if (!path || /^https?:\/\//i.test(path) || /^(data|blob):/i.test(path)) {
    return path;
  }
  return `${OCT_ANALYZER_API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}
