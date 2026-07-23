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
      <div className="grid gap-5 lg:grid-cols-12">
        <Card title="Patient &amp; Scan Intake" subtitle="Complete details before upload" icon={Upload} className="lg:col-span-4">
          <div className="mb-4 space-y-3">
            <div>
              <label className="mb-1 block text-sm font-bold text-slate-700">Patient ID</label>
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                disabled={uploadState.status !== "Waiting"}
                placeholder="Optional (e.g. PT-10294)"
                className="w-full rounded-xl border border-slate-300 px-4 py-2 text-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:bg-slate-100"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-bold text-slate-700">Imaging Modality</label>
              <select
                value={modality}
                onChange={(e) => {
                  const newModality = e.target.value;
                  setModality(newModality);
                  if (newModality === "OCTA") setPattern("Cube / Volume (3D)");
                }}
                disabled={uploadState.status !== "Waiting"}
                className="w-full rounded-xl border border-slate-300 px-4 py-2 text-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:bg-slate-100"
              >
                <option value="Structural OCT">Structural OCT</option>
                <option value="OCTA">OCTA</option>
                <option value="EDI-OCT">EDI-OCT</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-sm font-bold text-slate-700">Anatomical Target</label>
              <select
                value={target}
                onChange={(e) => {
                  const newTarget = e.target.value;
                  setTarget(newTarget);
                  if (newTarget === "Optic Disc / ONH") setPattern("Circle / Ring");
                  else if (newTarget === "Anterior Segment") setPattern("Line / Single B-Scan");
                }}
                disabled={uploadState.status !== "Waiting"}
                className="w-full rounded-xl border border-slate-300 px-4 py-2 text-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:bg-slate-100"
              >
                <option value="Macula">Macula</option>
                <option value="Optic Disc / ONH">Optic Disc / ONH</option>
                <option value="Anterior Segment">Anterior Segment</option>
                <option value="Widefield">Widefield</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-sm font-bold text-slate-700">Scan Pattern</label>
              <select
                value={pattern}
                onChange={(e) => setPattern(e.target.value)}
                disabled={uploadState.status !== "Waiting"}
                className="w-full rounded-xl border border-slate-300 px-4 py-2 text-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:bg-slate-100"
              >
                <option value="Cube / Volume (3D)">Cube / Volume (3D)</option>
                <option value="Raster">Raster</option>
                <option value="Line / Single B-Scan">Line / Single B-Scan</option>
                <option value="Radial / Star">Radial / Star</option>
                <option value="Circle / Ring">Circle / Ring</option>
                <option value="Mesh / Grid">Mesh / Grid</option>
              </select>
            </div>
          </div>

          {aiDisabled && (
            <div className="mt-4 rounded-2xl bg-amber-50 p-4 border border-amber-200 text-sm text-amber-900 shadow-sm">
              <div className="font-bold flex items-center gap-2 mb-1 text-amber-950">
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
              dragging ? "border-sky-400 bg-sky-50 text-sky-800" : "border-slate-300 bg-slate-50 hover:border-slate-400 text-slate-500"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".vol,.dcm,.zip,.tif,.tiff,image/png,image/jpeg,image/webp,image/tiff"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <Upload className="mb-2 h-7 w-7" />
            <p className="text-sm font-bold text-slate-800">Drag OCT/OCTA volume here</p>
            <button onClick={() => inputRef.current?.click()} className="mt-2 rounded-xl bg-slate-900 px-4 py-1.5 text-xs font-bold text-white hover:bg-slate-800">
              Browse Files
            </button>
          </div>

          <div className="mt-4 grid gap-3 text-sm">
            <div className="rounded-2xl bg-slate-50 p-4"><b>File:</b> {scan?.filename || uploadState.fileName || "No scan uploaded"}</div>
            <div className="rounded-2xl bg-slate-50 p-4"><b>Format:</b> {scan?.source_format || "Awaiting upload"}</div>
            <div className="rounded-2xl bg-slate-50 p-4"><b>Shape:</b> {scan?.volume_shape?.join(" x ") || "N/A"}</div>
          </div>
        </Card>

        <Card title="Quality Control Gate" subtitle="The model can be blocked before inference if scan quality is unsafe." icon={CheckCircle2} className="lg:col-span-4">
          <div className="space-y-3">
            <Metric label="Signal range" value={signalRange.length ? `${signalRange[0].toFixed(1)}-${signalRange[1].toFixed(1)}` : "N/A"} tone={completed ? "safe" : "neutral"} />
            <Metric label="Crop applied" value={scan?.qc?.crop_applied ? "Yes" : completed ? "No" : "N/A"} tone={scan?.qc?.crop_applied ? "info" : completed ? "safe" : "neutral"} />
            <Metric label="Warnings" value={qcWarnings.length} tone={qcWarnings.length ? "warning" : completed ? "safe" : "neutral"} />
            <Metric label="QC decision" value={uploadState.error ? "Blocked" : completed ? "Proceed" : uploadState.status} tone={uploadState.error ? "danger" : completed ? "info" : "neutral"} />
          </div>
          {uploadState.error ? <div className="mt-4 rounded-2xl bg-rose-50 p-4 text-sm text-rose-800">{uploadState.error}</div> : null}
          {qcWarnings.length ? <div className="mt-4 rounded-2xl bg-amber-50 p-4 text-sm text-amber-900">{qcWarnings.join(" ")}</div> : null}
        </Card>

        <Card title="Pipeline Status" subtitle="Clinician sees where the case is in the system." icon={Activity} className="lg:col-span-4">
          <div className="space-y-4">
            {["Upload received", "Preprocessing complete", "QC passed", "Inference complete", "Report ready"].map((step, index) => {
              const done = completed || uploadState.progress > index * 20;
              return (
                <div key={step} className={`flex items-center gap-3 rounded-2xl p-4 transition-colors ${done ? "bg-emerald-50" : "bg-slate-50"}`}>
                  <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-black ${done ? "bg-emerald-600 text-white" : "bg-white text-slate-700"}`}>
                    {index + 1}
                  </div>
                  <span className="font-semibold text-slate-700">{step}</span>
                </div>
              );
            })}
          </div>
          {!completed && uploadState.detail && (
            <div className="mt-4 rounded-2xl bg-sky-50 p-4 text-sm text-sky-800 flex items-center gap-2">
              <Spinner className="h-4 w-4 text-sky-800" />
              <b>Current step:</b> {uploadState.detail}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
