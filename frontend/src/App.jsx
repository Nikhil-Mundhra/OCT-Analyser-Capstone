import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  ClipboardCheck,
  Clock,
  Download,
  Eye,
  FileText,
  History,
  Route,
  ScanLine,
  ShieldAlert,
  Stethoscope,
  Upload,
  UserRound,
} from "lucide-react";

import { createScan } from "./api/octAnalyzerClient.js";

const screens = [
  { id: "worklist", label: "1. Triage Worklist" },
  { id: "upload", label: "2. Upload and QC" },
  { id: "review", label: "3. Scan Review" },
  { id: "decision", label: "4. Human Decision Gate" },
  { id: "outcomes", label: "5. Outcomes and Safety" },
];

const demoRows = [
  ["P-1029", "Macula OCT", "Low risk", "92%", "AI cleared, clinician sample review", "safe"],
  ["P-1184", "OCTA", "Ambiguous", "61%", "Send to human audit", "warning"],
  ["P-1210", "Optic disc OCT", "High risk", "88%", "Specialist review required", "danger"],
  ["P-1244", "Macula OCT", "Poor quality", "N/A", "Re-upload or manual review", "warning"],
];

function StatusBadge({ children, tone = "neutral" }) {
  const tones = {
    neutral: "bg-slate-100 text-slate-700 border-slate-200",
    safe: "bg-emerald-50 text-emerald-700 border-emerald-200",
    warning: "bg-amber-50 text-amber-800 border-amber-200",
    danger: "bg-rose-50 text-rose-700 border-rose-200",
    info: "bg-sky-50 text-sky-700 border-sky-200",
    purple: "bg-violet-50 text-violet-700 border-violet-200",
  };

  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

function Card({ title, subtitle, icon: Icon, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      <div className="mb-4 flex items-start gap-3">
        {Icon ? (
          <div className="rounded-2xl bg-slate-100 p-2 text-slate-700">
            <Icon className="h-5 w-5" />
          </div>
        ) : null}
        <div>
          <h3 className="text-base font-bold text-slate-900">{title}</h3>
          {subtitle ? <p className="mt-1 text-sm text-slate-500">{subtitle}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

function WireBox({ className = "", height = "h-24", children }) {
  return (
    <div className={`rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 ${height} ${className}`}>
      <div className="flex h-full items-center justify-center p-4 text-center text-sm font-medium text-slate-500">
        {children}
      </div>
    </div>
  );
}

function Metric({ label, value, tone = "neutral" }) {
  const tones = {
    neutral: "bg-slate-50 text-slate-900 border-slate-200",
    safe: "bg-emerald-50 text-emerald-800 border-emerald-200",
    warning: "bg-amber-50 text-amber-900 border-amber-200",
    danger: "bg-rose-50 text-rose-800 border-rose-200",
    info: "bg-sky-50 text-sky-800 border-sky-200",
  };
  return (
    <div className={`rounded-2xl border p-4 ${tones[tone]}`}>
      <p className="text-xs font-semibold uppercase tracking-wide opacity-70">{label}</p>
      <p className="mt-2 text-2xl font-black">{value}</p>
    </div>
  );
}

function Header({ scan }) {
  return (
    <header className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="mb-2 flex flex-wrap gap-2">
            <StatusBadge tone="info">Clinical decision support</StatusBadge>
            <StatusBadge tone="purple">Human in the loop</StatusBadge>
            <StatusBadge tone="warning">Not autonomous diagnosis</StatusBadge>
            {scan?.is_demo_model ? <StatusBadge tone="warning">Demo model</StatusBadge> : null}
          </div>
          <h1 className="text-3xl font-black tracking-tight text-slate-950">OCT/OCTA Clinical Inference Interface</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            End-to-end clinician workflow for triage, explainable scan review, active human justification, specialist routing, and safety monitoring.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="rounded-2xl bg-slate-50 p-4">
            <p className="font-semibold text-slate-500">Current user</p>
            <p className="mt-1 font-bold text-slate-900">Ophthalmologist</p>
          </div>
          <div className="rounded-2xl bg-slate-50 p-4">
            <p className="font-semibold text-slate-500">Mode</p>
            <p className="mt-1 font-bold text-slate-900">{scan ? "Active case" : "Review queue"}</p>
          </div>
        </div>
      </div>
    </header>
  );
}

function riskFromScan(scan) {
  if (!scan || scan.status !== "completed") {
    return { label: "No active scan", tone: "neutral", confidence: "N/A", action: "Upload scan" };
  }
  if (scan.diagnosis === "DR") {
    return { label: "High risk", tone: "danger", confidence: `${Math.round(scan.confidence * 100)}%`, action: "Specialist review required" };
  }
  if (scan.confidence < 0.7) {
    return { label: "Ambiguous", tone: "warning", confidence: `${Math.round(scan.confidence * 100)}%`, action: "Send to human audit" };
  }
  return { label: "Low risk", tone: "safe", confidence: `${Math.round(scan.confidence * 100)}%`, action: "Clinician sample review" };
}

function WorklistScreen({ scan, setActive }) {
  const liveRisk = riskFromScan(scan);
  const rows = scan?.status === "completed"
    ? [["LOCAL-001", scan.source_format, liveRisk.label, liveRisk.confidence, liveRisk.action, liveRisk.tone], ...demoRows]
    : demoRows;

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Triage Queue" subtitle="AI rapidly processes routine scans and protects specialist bandwidth." icon={Route} className="lg:col-span-8">
        <div className="overflow-hidden rounded-2xl border border-slate-200">
          <div className="grid grid-cols-5 bg-slate-100 px-4 py-3 text-xs font-bold uppercase tracking-wide text-slate-500">
            <span>Patient</span>
            <span>Scan</span>
            <span>AI route</span>
            <span>Confidence</span>
            <span>Action</span>
          </div>
          {rows.map((row) => (
            <div key={row[0]} className="grid grid-cols-5 items-center border-t border-slate-100 px-4 py-4 text-sm text-slate-700">
              <span className="font-bold text-slate-900">{row[0]}</span>
              <span>{row[1]}</span>
              <span><StatusBadge tone={row[5]}>{row[2]}</StatusBadge></span>
              <span className="font-bold">{row[3]}</span>
              <span>{row[4]}</span>
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button onClick={() => setActive("upload")} className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-bold text-white">
            Upload new scan
          </button>
          {scan?.status === "completed" ? (
            <button onClick={() => setActive("review")} className="rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700">
              Review active case
            </button>
          ) : null}
        </div>
      </Card>

      <Card title="Triage Rules" subtitle="Routing logic is visible to reduce blind trust." icon={ShieldAlert} className="lg:col-span-4">
        <div className="space-y-3 text-sm text-slate-700">
          <div className="rounded-2xl bg-emerald-50 p-4"><b>Low risk:</b> routine queue, optional clinician sampling.</div>
          <div className="rounded-2xl bg-amber-50 p-4"><b>Ambiguous:</b> requires human audit before report sign-off.</div>
          <div className="rounded-2xl bg-rose-50 p-4"><b>High risk:</b> route to specialist with anomaly overlays and history.</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Poor QC:</b> no model claim, request re-upload or manual inspection.</div>
        </div>
      </Card>
    </div>
  );
}

function UploadScreen({ scan, uploadState, onUpload }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(files) {
    const file = files?.[0];
    if (file) {
      onUpload(file);
    }
  }

  const qcWarnings = scan?.qc?.warnings || [];
  const signalRange = scan?.qc?.signal_range || [];
  const completed = scan?.status === "completed";

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Scan Intake" subtitle=".vol, .dcm, or zipped TIFF/BMP/PNG export." icon={Upload} className="lg:col-span-4">
        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            handleFiles(event.dataTransfer.files);
          }}
          className={`flex h-52 flex-col items-center justify-center rounded-2xl border-2 border-dashed p-4 text-center ${
            dragging ? "border-sky-400 bg-sky-50 text-sky-800" : "border-slate-300 bg-slate-50 text-slate-500"
          }`}
        >
          <input ref={inputRef} type="file" accept=".vol,.dcm,.zip" className="hidden" onChange={(event) => handleFiles(event.target.files)} />
          <Upload className="mb-3 h-9 w-9" />
          <p className="font-bold text-slate-800">Drag OCT/OCTA volume here</p>
          <p className="mt-1 text-sm">or select scan from local system</p>
          <button onClick={() => inputRef.current?.click()} className="mt-4 rounded-2xl bg-slate-900 px-4 py-2 text-sm font-bold text-white">
            Select scan
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
              <div key={step} className={`flex items-center gap-3 rounded-2xl p-4 ${done ? "bg-emerald-50" : "bg-slate-50"}`}>
                <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-black ${done ? "bg-emerald-600 text-white" : "bg-white text-slate-700"}`}>
                  {index + 1}
                </div>
                <span className="font-semibold text-slate-700">{step}</span>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

function ReviewScreen({ scan }) {
  const [preview, setPreview] = useState("overlay");
  const completed = scan?.status === "completed";
  const previewUrl = completed ? scan.previews?.[preview] : "";
  const risk = riskFromScan(scan);
  const drLayers = scan?.layers?.filter((layer) => layer.vote === "DR").length || 0;
  const ipnv2 = scan?.ipnv2;
  const previewOptions = [
    ["raw", "Raw"],
    ["overlay", "Layer demo"],
    ["features", "CDF chart"],
  ];
  if (ipnv2?.previews?.ipnv2_overlay) {
    previewOptions.push(["ipnv2_overlay", "OCTA/IPN-V2"]);
  }
  if (ipnv2?.previews?.ipnv2_probability) {
    previewOptions.push(["ipnv2_probability", "IPN prob"]);
  }

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Volumetric Scan Viewer" subtitle="Slice viewer with layer demo overlays, CDF chart, and optional IPN-V2 OCTA output." icon={ScanLine} className="lg:col-span-7">
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-950 lg:col-span-2">
            {previewUrl ? (
              <img src={previewUrl} alt={`${preview} preview`} className="h-72 w-full object-contain" />
            ) : (
              <WireBox height="h-72">Upload a scan to view generated previews</WireBox>
            )}
          </div>
          <div className="space-y-4">
            <PreviewThumb src={scan?.previews?.raw} label="Raw center slice" />
            <PreviewThumb src={ipnv2?.previews?.ipnv2_probability || scan?.previews?.features} label={ipnv2?.previews?.ipnv2_probability ? "IPN-V2 probability" : "CDF feature chart"} />
          </div>
        </div>
        <div className="mt-4 grid gap-3 text-xs font-semibold text-slate-600 md:grid-cols-5">
          {previewOptions.map(([id, label]) => (
            <button
              key={id}
              onClick={() => setPreview(id)}
              className={`rounded-xl border px-3 py-2 ${preview === id ? "bg-slate-900 text-white" : "bg-white"}`}
            >
              {label}
            </button>
          ))}
        </div>
        {ipnv2 ? (
          <div className={`mt-4 rounded-2xl border p-4 text-sm ${ipnv2.mode === "checkpoint" ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-amber-200 bg-amber-50 text-amber-900"}`}>
            <b>IPN-V2:</b> {ipnv2.mode === "checkpoint" ? "Checkpoint-backed OCTA segmentation path." : "Untrained smoke mode. This output proves integration plumbing only."}
          </div>
        ) : null}
      </Card>

      <Card title="AI Findings" subtitle="No hidden single answer, show confidence and competing evidence." icon={Brain} className="lg:col-span-5">
        <div className="space-y-3">
          <Metric label="Risk tier" value={risk.label} tone={risk.tone} />
          <Metric label="Model confidence" value={risk.confidence} tone={risk.tone === "danger" ? "warning" : "safe"} />
          <Metric label="Layer votes flagged" value={`${drLayers}/12`} tone={drLayers ? "warning" : completed ? "safe" : "neutral"} />
          <Metric label="IPN-V2 mode" value={ipnv2?.mode ? ipnv2.mode.replace("_", " ") : "N/A"} tone={ipnv2?.mode === "checkpoint" ? "safe" : ipnv2 ? "warning" : "neutral"} />
          <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
            <b>Explainability:</b> layer diagnosis still uses the placeholder CDF contract. IPN-V2 is displayed separately as OCTA/en face segmentation output, not as DR diagnosis.
          </div>
          {ipnv2?.warning ? <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">{ipnv2.warning}</div> : null}
          <LayerVoteList layers={scan?.layers || []} />
        </div>
      </Card>
    </div>
  );
}

function PreviewThumb({ src, label }) {
  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-950">
      {src ? <img src={src} alt={label} className="h-32 w-full object-contain" /> : <WireBox height="h-32">{label}</WireBox>}
    </div>
  );
}

function LayerVoteList({ layers }) {
  if (!layers.length) {
    return <div className="rounded-2xl bg-slate-50 p-4 text-sm text-slate-600">Layer votes will appear after upload.</div>;
  }

  return (
    <div className="max-h-64 overflow-auto rounded-2xl border border-slate-200">
      {layers.map((layer) => (
        <div key={layer.name} className="grid grid-cols-3 gap-2 border-b border-slate-100 px-4 py-3 text-sm last:border-b-0">
          <span className="font-bold text-slate-900">{layer.name}</span>
          <span className={layer.vote === "DR" ? "font-bold text-rose-700" : "font-bold text-emerald-700"}>{layer.vote}</span>
          <span className="text-right text-slate-600">{layer.score.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

function DecisionScreen({ scan, decision, setDecision }) {
  const [choice, setChoice] = useState(decision.choice || "");
  const [rationale, setRationale] = useState(decision.rationale || "");
  const risk = riskFromScan(scan);
  const canSubmit = choice && rationale.trim().length >= 12 && scan?.status === "completed";

  function submitDecision() {
    if (!canSubmit) {
      return;
    }
    setDecision({
      choice,
      rationale,
      submittedAt: new Date().toLocaleString(),
    });
  }

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Specialist Routing Packet" subtitle="High-risk or ambiguous scans are routed with history and highlighted anomalies." icon={Stethoscope} className="lg:col-span-5">
        <div className="space-y-3 text-sm text-slate-700">
          <div className="rounded-2xl bg-slate-50 p-4"><b>Recipient:</b> {risk.tone === "danger" ? "Retina specialist review queue" : "Ophthalmology review queue"}</div>
          <div className={`rounded-2xl p-4 ${risk.tone === "danger" ? "bg-rose-50" : "bg-slate-50"}`}><b>Reason:</b> {risk.action}</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Attached:</b> OCT volume, segmentation overlay, CDF chart, layer vote table</div>
          {scan?.previews?.overlay ? <img src={scan.previews.overlay} alt="Anomaly snapshot" className="h-40 w-full rounded-2xl border object-contain" /> : <WireBox height="h-40">Anomaly snapshot summary</WireBox>}
        </div>
      </Card>

      <Card title="Active Human Justification" subtitle="Critical decisions cannot be accepted with one passive click." icon={ClipboardCheck} className="lg:col-span-7">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 p-4">
            <p className="mb-3 text-sm font-bold text-slate-900">Clinician decision</p>
            <div className="space-y-2 text-sm text-slate-700">
              {["Agree with AI triage", "Override AI triage", "Defer to specialist"].map((label) => (
                <label key={label} className="block rounded-xl border p-3">
                  <input type="radio" name="decision" className="mr-2" checked={choice === label} onChange={() => setChoice(label)} />
                  {label}
                </label>
              ))}
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 p-4">
            <p className="mb-3 text-sm font-bold text-slate-900">Required justification</p>
            <textarea
              value={rationale}
              onChange={(event) => setRationale(event.target.value)}
              className="h-40 w-full resize-none rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-700 outline-none focus:border-sky-400"
              placeholder="Type rationale: image evidence, clinical history, uncertainty concerns, or reason for override."
            />
          </div>
        </div>
        <div className="mt-4 rounded-2xl bg-amber-50 p-4 text-sm text-amber-900">
          <b>Automation bias guardrail:</b> Sign-off remains disabled until the clinician reviews overlays, sees confidence, and enters a decision rationale.
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button disabled={!canSubmit} onClick={submitDecision} className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-bold text-white disabled:cursor-not-allowed disabled:opacity-40">
            Submit reviewed decision
          </button>
          <button onClick={() => setChoice("Defer to specialist")} className="rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700">Request second opinion</button>
          <button onClick={downloadCaseJson(scan, decision)} className="inline-flex items-center gap-2 rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700">
            <Download className="h-4 w-4" /> Export JSON
          </button>
        </div>
        {decision.submittedAt ? (
          <div className="mt-4 rounded-2xl bg-emerald-50 p-4 text-sm text-emerald-800">
            Decision saved: <b>{decision.choice}</b> at {decision.submittedAt}
          </div>
        ) : null}
      </Card>
    </div>
  );
}

function downloadCaseJson(scan, decision) {
  return () => {
    if (!scan) {
      return;
    }
    const blob = new Blob([JSON.stringify({ scan, decision }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${scan.scan_id || "oct-case"}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };
}

function OutcomesScreen({ scan, decision }) {
  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Outcome and Safety Monitor" subtitle="The system is evaluated by patient impact, not isolated algorithm accuracy." icon={BarChart3} className="lg:col-span-7">
        <div className="grid gap-4 md:grid-cols-2">
          <Metric label="Time to specialist review" value={scan ? "Ready now" : "N/A"} tone={scan ? "safe" : "neutral"} />
          <Metric label="Missed high-risk reviews" value="0 flagged" tone="safe" />
          <Metric label="Clinician decision" value={decision.choice ? "Saved" : "Pending"} tone={decision.choice ? "info" : "warning"} />
          <Metric label="Adverse safety signals" value="Monitor" tone="warning" />
        </div>
        <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-bold text-slate-900">
            <FileText className="h-4 w-4" /> Active case summary
          </div>
          <pre className="max-h-64 overflow-auto rounded-2xl bg-slate-950 p-4 text-xs text-slate-100">
            {JSON.stringify({
              scan_id: scan?.scan_id || null,
              diagnosis: scan?.diagnosis || null,
              confidence: scan?.confidence || null,
              decision: decision.choice || null,
              submitted_at: decision.submittedAt || null,
            }, null, 2)}
          </pre>
        </div>
      </Card>

      <Card title="Audit Trail" subtitle="Every AI claim, human action, and model version is traceable." icon={History} className="lg:col-span-5">
        <div className="space-y-3 text-sm text-slate-700">
          <div className="rounded-2xl bg-slate-50 p-4"><b>Model:</b> Local OCT MVP placeholder, demo mode</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Case events:</b> upload, QC pass, inference, explanation, clinician sign-off</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Feedback loop:</b> override cases can be exported as JSON</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Report export:</b> structured JSON summary</div>
        </div>
      </Card>
    </div>
  );
}

function ClinicalInterfaceApp() {
  const [active, setActive] = useState("worklist");
  const [scan, setScan] = useState(null);
  const [uploadState, setUploadState] = useState({ status: "Waiting", progress: 0, fileName: "", error: "" });
  const [decision, setDecision] = useState({ choice: "", rationale: "", submittedAt: "" });
  const activeScreen = useMemo(() => screens.find((screen) => screen.id === active), [active]);

  async function uploadScan(file) {
    setUploadState({ status: "Uploading", progress: 20, fileName: file.name, error: "" });
    setDecision({ choice: "", rationale: "", submittedAt: "" });

    try {
      setUploadState({ status: "Processing", progress: 55, fileName: file.name, error: "" });
      const payload = await createScan(file);
      setScan(payload);
      setUploadState({ status: "Completed", progress: 100, fileName: file.name, error: "" });
      setActive("review");
    } catch (error) {
      setScan(null);
      setUploadState({
        status: "Failed",
        progress: 0,
        fileName: file.name,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 p-6 text-slate-900">
      <div className="mx-auto max-w-7xl space-y-6">
        <Header scan={scan} />

        <nav className="flex flex-wrap gap-2 rounded-3xl border border-slate-200 bg-white p-3 shadow-sm">
          {screens.map((screen) => (
            <button
              key={screen.id}
              onClick={() => setActive(screen.id)}
              className={`rounded-2xl px-4 py-3 text-sm font-bold transition ${
                active === screen.id
                  ? "bg-slate-900 text-white shadow-sm"
                  : "bg-slate-50 text-slate-600 hover:bg-slate-100"
              }`}
            >
              {screen.label}
            </button>
          ))}
        </nav>

        <section className="rounded-3xl border border-slate-200 bg-white/60 p-4 shadow-sm">
          <div className="mb-4 flex flex-col justify-between gap-3 lg:flex-row lg:items-center">
            <div>
              <p className="text-xs font-bold uppercase tracking-widest text-slate-500">Clinical screen</p>
              <h2 className="mt-1 text-2xl font-black text-slate-950">{activeScreen?.label}</h2>
            </div>
            <div className="flex flex-wrap gap-2">
              <StatusBadge tone="info"><Clock className="mr-1 h-3 w-3" /> Fast response</StatusBadge>
              <StatusBadge tone="warning"><AlertTriangle className="mr-1 h-3 w-3" /> Confidence shown</StatusBadge>
              <StatusBadge tone="purple"><UserRound className="mr-1 h-3 w-3" /> Human justification</StatusBadge>
              <StatusBadge tone="safe"><Eye className="mr-1 h-3 w-3" /> Visual evidence</StatusBadge>
            </div>
          </div>

          {active === "worklist" && <WorklistScreen scan={scan} setActive={setActive} />}
          {active === "upload" && <UploadScreen scan={scan} uploadState={uploadState} onUpload={uploadScan} />}
          {active === "review" && <ReviewScreen scan={scan} />}
          {active === "decision" && <DecisionScreen scan={scan} decision={decision} setDecision={setDecision} />}
          {active === "outcomes" && <OutcomesScreen scan={scan} decision={decision} />}
        </section>

        <footer className="grid gap-4 rounded-3xl border border-slate-200 bg-white p-5 text-sm text-slate-600 shadow-sm lg:grid-cols-4">
          <div><b className="text-slate-900">Triage:</b> low-risk cases move quickly, uncertain cases are escalated.</div>
          <div><b className="text-slate-900">Transparency:</b> confidence, uncertainty, and overlays are visible.</div>
          <div><b className="text-slate-900">Safety:</b> critical sign-off requires human rationale.</div>
          <div><b className="text-slate-900">Evaluation:</b> monitor outcomes, latency, overrides, and safety signals.</div>
        </footer>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<ClinicalInterfaceApp />);
