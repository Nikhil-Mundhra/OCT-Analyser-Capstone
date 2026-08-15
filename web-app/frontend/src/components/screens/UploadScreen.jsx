"use client";
import React, { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Activity, AlertTriangle, CheckCircle2, Upload } from "lucide-react";
import { useAppContext } from "../../AppContext";
import { createScan } from "../../api/octAnalyzerClient";
import { Card } from "../ui/Card";
import { Metric } from "../ui/Metric";
import { Spinner } from "../ui/Spinner";

export function UploadScreen() {
  const { scan, uploadState, setUploadState, setScan, addScanToHistory, resetUpload } = useAppContext();
  const router = useRouter();
  const [patientId, setPatientId] = useState("");
  const [modality, setModality] = useState("Structural OCT");
  const [target, setTarget] = useState("Macula");
  const [pattern, setPattern] = useState("Cube / Volume (3D)");
  const [applyArtifactMask, setApplyArtifactMask] = useState(true);
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const onUpload = useCallback(async (file) => {
    const finalPatientId = patientId.trim() || `ANON-${Math.floor(Math.random() * 10000)}`;
    setUploadState({ status: "Uploading", progress: 20, fileName: file.name, error: "" });

    try {
      setUploadState({ status: "Processing", progress: 55, detail: "Initializing...", fileName: file.name, error: "" });
      const payload = await createScan(file, (detail) => {
        let progress = 55;
        if (detail.includes("Preprocessing")) progress = 60;
        else if (detail.includes("Flattening")) progress = 65;
        else if (detail.includes("inference")) progress = 75;
        else if (detail.includes("Extracting")) progress = 85;
        else if (detail.includes("GradCAM")) progress = 95;
        setUploadState(prev => ({ ...prev, detail, progress: Math.max(prev.progress, progress) }));
      });
      const combinedScanType = `${modality} - ${target} - ${pattern}`;
      const aiSupported = modality === "Structural OCT" && target === "Macula";
      const enrichedPayload = { ...payload, file, patient_id: finalPatientId, scan_type: combinedScanType, modality, target, pattern, ai_supported: aiSupported };
      setScan(enrichedPayload);
      addScanToHistory(enrichedPayload);
      setUploadState({ status: "Completed", progress: 100, detail: "", fileName: file.name, error: "" });
      router.push("/review");
    } catch (error) {
      setScan(null);
      setUploadState({
        status: "Failed",
        progress: 0,
        fileName: file.name,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }, [patientId, modality, target, pattern, setUploadState, setScan, addScanToHistory, router]);

  function handleFiles(files) {
    const file = files?.[0];
    if (file) onUpload(file);
  }

  const qcWarnings = scan?.qc?.warnings || [];
  const signalRange = scan?.qc?.signal_range || [];
  const completed = scan?.status === "completed";

  const handleReset = useCallback(() => {
    resetUpload();
    setPatientId("");
  }, [resetUpload]);

  const aiDisabled = !(modality === "Structural OCT" && target === "Macula");

  return (
    <div className="flex flex-col gap-5">
      {uploadState.status !== "Waiting" && (
        <div className="flex justify-end">
          <button onClick={handleReset} className="rounded-2xl bg-rose-100 px-4 py-2 text-sm font-bold text-rose-700 hover:bg-rose-200 transition-colors">
            Cancel &amp; Start New Upload
          </button>
        </div>
      )}
      <div className="flex flex-col gap-5">
        <Card title="Patient &amp; Scan Intake" subtitle="Complete details before upload" icon={Upload}>
          <div className="mb-4 space-y-3">
            <div>
              <label className="mb-1 block text-xs font-bold text-slate-300">Patient ID</label>
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                disabled={uploadState.status !== "Waiting"}
                placeholder="Optional (e.g. PT-10294)"
                className="w-full rounded-xl border border-slate-800 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-500 focus:outline-none disabled:bg-slate-900 disabled:text-slate-500"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-bold text-slate-300">Imaging Modality</label>
              <select
                value={modality}
                onChange={(e) => {
                  const newModality = e.target.value;
                  setModality(newModality);
                  if (newModality === "OCTA") setPattern("Cube / Volume (3D)");
                }}
                disabled={uploadState.status !== "Waiting"}
                className="w-full rounded-xl border border-slate-800 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 focus:border-sky-500 focus:outline-none disabled:bg-slate-900 disabled:text-slate-500"
              >
                <option value="Structural OCT" className="bg-slate-900 text-slate-100">Structural OCT</option>
                <option value="OCTA" className="bg-slate-900 text-slate-100">OCTA</option>
                <option value="EDI-OCT" className="bg-slate-900 text-slate-100">EDI-OCT</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-bold text-slate-300">Anatomical Target</label>
              <select
                value={target}
                onChange={(e) => {
                  const newTarget = e.target.value;
                  setTarget(newTarget);
                  if (newTarget === "Optic Disc / ONH") setPattern("Circle / Ring");
                  else if (newTarget === "Anterior Segment") setPattern("Line / Single B-Scan");
                }}
                disabled={uploadState.status !== "Waiting"}
                className="w-full rounded-xl border border-slate-800 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 focus:border-sky-500 focus:outline-none disabled:bg-slate-900 disabled:text-slate-500"
              >
                <option value="Macula" className="bg-slate-900 text-slate-100">Macula</option>
                <option value="Optic Disc / ONH" className="bg-slate-900 text-slate-100">Optic Disc / ONH</option>
                <option value="Anterior Segment" className="bg-slate-900 text-slate-100">Anterior Segment</option>
                <option value="Widefield" className="bg-slate-900 text-slate-100">Widefield</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-bold text-slate-300">Scan Pattern</label>
              <select
                value={pattern}
                onChange={(e) => setPattern(e.target.value)}
                disabled={uploadState.status !== "Waiting"}
                className="w-full rounded-xl border border-slate-800 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 focus:border-sky-500 focus:outline-none disabled:bg-slate-900 disabled:text-slate-500"
              >
                <option value="Cube / Volume (3D)" className="bg-slate-900 text-slate-100">Cube / Volume (3D)</option>
                <option value="Raster" className="bg-slate-900 text-slate-100">Raster</option>
                <option value="Line / Single B-Scan" className="bg-slate-900 text-slate-100">Line / Single B-Scan</option>
                <option value="Radial / Star" className="bg-slate-900 text-slate-100">Radial / Star</option>
                <option value="Circle / Ring" className="bg-slate-900 text-slate-100">Circle / Ring</option>
                <option value="Mesh / Grid" className="bg-slate-900 text-slate-100">Mesh / Grid</option>
              </select>
            </div>
            
            {/* Optional Bottom-Left Artifact Masking Toggle */}
            <div className="pt-1">
              <label className="flex items-start gap-2.5 cursor-pointer p-3 rounded-xl bg-slate-950 border border-slate-800 hover:border-slate-700 transition-all">
                <input
                  type="checkbox"
                  checked={applyArtifactMask}
                  onChange={(e) => setApplyArtifactMask(e.target.checked)}
                  disabled={uploadState.status !== "Waiting"}
                  className="mt-0.5 accent-sky-500 h-4 w-4 cursor-pointer"
                />
                <div>
                  <span className="text-xs font-bold text-slate-200 block">Apply Artifact Mask (Bottom-Left Zeroing)</span>
                  <span className="text-[11px] text-slate-400 block mt-0.5 leading-snug">
                    Zeroes out manufacturer logo/compass artifacts. Uncheck if scan is clean or tissue extends into corner.
                  </span>
                </div>
              </label>
            </div>
          </div>

          {aiDisabled && (
            <div className="mt-4 rounded-2xl bg-amber-950/40 p-4 border border-amber-500/30 text-sm text-amber-300 shadow-sm">
              <div className="font-bold flex items-center gap-2 mb-1 text-amber-200">
                <AlertTriangle className="h-4 w-4" /> AI Inference Disabled
              </div>
              The AI models are only validated for <b>Structural OCT</b> of the <b>Macula</b>. This scan can be uploaded for manual review, but no AI insights will be generated.
            </div>
          )}

          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
            className={`flex h-40 flex-col items-center justify-center rounded-2xl border-2 border-dashed p-4 text-center transition-colors ${
              dragging ? "border-sky-400 bg-sky-950/40 text-sky-300" : "border-slate-800 bg-slate-950/60 hover:border-slate-700 text-slate-400"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".vol,.dcm,.zip,.tif,.tiff,image/png,image/jpeg,image/webp,image/tiff"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <Upload className="mb-2 h-7 w-7 text-sky-400" />
            <p className="text-sm font-bold text-slate-200">Drag OCT/OCTA volume here</p>
            <button onClick={() => inputRef.current?.click()} className="mt-2 rounded-xl bg-sky-500 border border-sky-400 px-4 py-1.5 text-xs font-bold text-slate-950 hover:bg-sky-400 transition-all">
              Browse Files
            </button>
          </div>

          <div className="mt-4 grid gap-3 text-xs">
            <div className="rounded-xl bg-slate-950 border border-slate-800 p-3 text-slate-300"><b className="text-slate-400">File:</b> {scan?.filename || uploadState.fileName || "No scan uploaded"}</div>
            <div className="rounded-xl bg-slate-950 border border-slate-800 p-3 text-slate-300"><b className="text-slate-400">Format:</b> {scan?.source_format || "Awaiting upload"}</div>
            <div className="rounded-xl bg-slate-950 border border-slate-800 p-3 text-slate-300"><b className="text-slate-400">Shape:</b> {scan?.volume_shape?.join(" x ") || "N/A"}</div>
          </div>
        </Card>

        <Card title="Quality Control Gate" subtitle="The model can be blocked before inference if scan quality is unsafe." icon={CheckCircle2}>
          <div className="space-y-3">
            <Metric label="Signal range" value={signalRange.length ? `${signalRange[0].toFixed(1)}-${signalRange[1].toFixed(1)}` : "N/A"} tone={completed ? "safe" : "neutral"} />
            <Metric label="Crop applied" value={scan?.qc?.crop_applied ? "Yes" : completed ? "No" : "N/A"} tone={scan?.qc?.crop_applied ? "info" : completed ? "safe" : "neutral"} />
            <Metric label="Warnings" value={qcWarnings.length} tone={qcWarnings.length ? "warning" : completed ? "safe" : "neutral"} />
            <Metric label="QC decision" value={uploadState.error ? "Blocked" : completed ? "Proceed" : uploadState.status} tone={uploadState.error ? "danger" : completed ? "info" : "neutral"} />
          </div>
          {uploadState.error ? <div className="mt-4 rounded-2xl bg-rose-950/40 border border-rose-500/30 p-4 text-sm text-rose-300">{uploadState.error}</div> : null}
          {qcWarnings.length ? <div className="mt-4 rounded-2xl bg-amber-950/40 border border-amber-500/30 p-4 text-sm text-amber-300">{qcWarnings.join(" ")}</div> : null}
        </Card>

        <Card title="Pipeline Status" subtitle="Clinician sees where the case is in the system." icon={Activity}>
          <div className="space-y-3">
            {[
              { label: "Upload received", doneThreshold: 20 },
              { label: "Preprocessing complete", doneThreshold: 55 },
              { label: "QC passed", doneThreshold: 60 },
              { label: "Inference", doneThreshold: 99 },
              { label: "Report ready", doneThreshold: 100 }
            ].map((stepObj, index) => {
              const isDone = completed || uploadState.progress >= stepObj.doneThreshold;
              const isActive = !completed && !isDone && (
                index === 0 ? uploadState.progress < 20 :
                index === 1 ? uploadState.progress >= 20 && uploadState.progress < 55 :
                index === 2 ? uploadState.progress >= 55 && uploadState.progress < 60 :
                index === 3 ? uploadState.progress >= 60 && uploadState.progress < 99 :
                uploadState.progress >= 99
              );

              let stepTitle = stepObj.label;
              if (index === 3) {
                stepTitle = isDone ? "Inference complete" : isActive ? "Inference in progress" : "Inference complete";
              }

              return (
                <div
                  key={stepObj.label}
                  className={`flex flex-col gap-1 rounded-xl p-3.5 border transition-all ${
                    isDone
                      ? "bg-emerald-950/30 border-emerald-500/30 text-emerald-300"
                      : isActive
                      ? "bg-sky-950/40 border-sky-500/50 text-sky-200 shadow-sm shadow-sky-950"
                      : "bg-slate-950 border-slate-800 text-slate-400"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold font-mono transition-colors ${
                        isDone
                          ? "bg-emerald-500 text-slate-950"
                          : isActive
                          ? "bg-sky-500 text-slate-950"
                          : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {isActive ? <Spinner className="h-3.5 w-3.5 text-slate-950" /> : index + 1}
                    </div>
                    <span className="font-semibold text-xs">{stepTitle}</span>
                  </div>

                  {/* Inline active detail string for current processing step */}
                  {isActive && uploadState.detail && (
                    <div className="ml-10 text-[11px] font-medium text-sky-300/90 leading-tight">
                      {uploadState.detail}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Card>
      </div>
    </div>
  );
}
