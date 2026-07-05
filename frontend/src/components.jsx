"use client";
import React, { useEffect, useMemo, useRef, useState } from "react";

import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { AppProvider, useAppContext } from "./AppContext.jsx";
import { HomePage } from "./HomePage.jsx";
import DocsLandingPage from './docs/pages/DocsLandingPage.jsx';
import DocArticlePage from './docs/pages/DocArticlePage.jsx';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  ClipboardCheck,
  Clock,
  Download,
  ExternalLink,
  Eye,
  FileText,
  History,
  PlayCircle,
  Route as RouteIcon,
  ScanLine,
  ShieldAlert,
  Stethoscope,
  Upload,
  UserRound,
} from "lucide-react";

import { createScan } from "./api/octAnalyzerClient.js";

const screens = [
  { id: "dashboard", path: "/dashboard", label: "Dashboard" },
  { id: "worklist", path: "/worklist", label: "1. Triage Worklist" },
  { id: "upload", path: "/QC", label: "2. Upload and QC" },
  { id: "review", path: "/review", label: "3. Scan Review" },
  { id: "decision", path: "/human-check", label: "4. Human Decision Gate" },
  { id: "outcomes", path: "/outcomes", label: "5. Outcomes and Safety" },
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

export function Header({ scan }) {
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

export function StickyNav() {
  const { scan } = useAppContext();
  const pathname = usePathname();
  const activePath = pathname;

  return (
    <nav className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 shadow-sm backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-3 lg:flex-row lg:items-center lg:justify-between">
        <Link
          href="/"
          className="flex min-h-0 items-center gap-3 rounded-xl border-0 bg-transparent p-0 text-left"
        >
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-900 text-white">
            <ScanLine className="h-5 w-5" />
          </span>
          <span>
            <span className="block text-sm font-black text-slate-950">OCT Analyzer</span>
            <span className="block text-xs font-semibold text-slate-500">{scan ? "Active case loaded" : "Documentation and workflow"}</span>
          </span>
        </Link>

        <div className="flex gap-2 overflow-x-auto pb-1 lg:flex-wrap lg:justify-end lg:pb-0">
          {screens.map((screen) => (
            <Link
              key={screen.id}
              href={screen.path}
              className={`shrink-0 rounded-xl px-4 py-2 text-sm font-bold transition ${
                activePath === screen.path
                  ? "bg-slate-900 text-white shadow-sm"
                  : "bg-slate-50 text-slate-600 hover:bg-slate-100"
              }`}
            >
              {screen.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
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

export function DashboardScreen() {
  const { scan, uploadState, decision } = useAppContext();
  const router = useRouter();
  const risk = riskFromScan(scan);
  const completed = scan?.status === "completed";
  const workflow = [
    ["Intake", "Upload OCT/OCTA export", uploadState.fileName ? "Active" : "Ready", Upload],
    ["QC", "Validate signal and volume shape", completed ? "Passed" : "Pending", CheckCircle2],
    ["Review", "Inspect previews and layer evidence", completed ? "Ready" : "Waiting", ScanLine],
    ["Decision", "Capture clinician rationale", decision.choice ? "Saved" : "Open", ClipboardCheck],
  ];

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm lg:col-span-7">
        <div className="grid gap-0 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="p-6 lg:p-8">
            <div className="mb-4 flex flex-wrap gap-2">
              <StatusBadge tone="info">OCT/OCTA workflow</StatusBadge>
              <StatusBadge tone={completed ? risk.tone : "warning"}>
                {completed ? `${risk.label} case loaded` : "Awaiting scan"}
              </StatusBadge>
            </div>
            <h2 className="max-w-2xl text-3xl font-black leading-tight text-slate-950">
              Clinical OCT review workspace for triage, QC, evidence, and sign-off
            </h2>
            <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-600">
              Start a local scan intake, continue reviewing the active case, or jump into the queue while preserving human oversight at every decision point.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <button
                onClick={() => router.push("/QC")}
                className="inline-flex items-center gap-2 rounded-2xl bg-slate-900 px-5 py-3 text-sm font-bold text-white"
              >
                <Upload className="h-4 w-4" /> Upload scan
              </button>
              <button
                onClick={() => router.push(completed ? "/review" : "/worklist")}
                className="inline-flex items-center gap-2 rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700"
              >
                <PlayCircle className="h-4 w-4" /> {completed ? "Continue review" : "Open worklist"}
              </button>
            </div>
          </div>
          <div className="border-t border-slate-200 bg-slate-950 p-5 lg:border-l lg:border-t-0">
            <HomeScanPreview scan={scan} />
          </div>
        </div>
      </section>

      <Card title="Current Case" subtitle="Live summary of the active local scan." icon={Activity} className="lg:col-span-5">
        <div className="grid gap-3 sm:grid-cols-2">
          <Metric label="Case state" value={completed ? "Review ready" : uploadState.status} tone={completed ? "safe" : uploadState.error ? "danger" : "neutral"} />
          <Metric label="Risk tier" value={risk.label} tone={risk.tone} />
          <Metric label="Confidence" value={risk.confidence} tone={completed ? "info" : "neutral"} />
          <Metric label="Decision" value={decision.choice ? "Saved" : "Pending"} tone={decision.choice ? "safe" : "warning"} />
        </div>
        <div className="mt-4 rounded-2xl bg-slate-50 p-4 text-sm text-slate-700">
          <b>Active file:</b> {scan?.filename || uploadState.fileName || "No scan selected"}
        </div>
      </Card>

      <Card title="Workflow Overview" subtitle="Each step keeps evidence and clinician agency visible." icon={RouteIcon} className="lg:col-span-8">
        <div className="grid gap-3 md:grid-cols-4">
          {workflow.map(([title, detail, state, Icon]) => (
            <button
              key={title}
              onClick={() => router.push(title === "Intake" ? "/QC" : title === "Review" ? "/review" : title === "Decision" ? "/human-check" : "/QC")}
              className="flex min-h-44 flex-col items-start justify-between rounded-2xl border border-slate-200 bg-slate-50 p-4 text-left transition hover:bg-white"
            >
              <Icon className="h-5 w-5 text-slate-700" />
              <span>
                <span className="block text-base font-black text-slate-950">{title}</span>
                <span className="mt-2 block text-sm font-medium leading-5 text-slate-600">{detail}</span>
              </span>
              <StatusBadge tone={state === "Passed" || state === "Saved" || state === "Ready" ? "safe" : state === "Active" ? "info" : "neutral"}>
                {state}
              </StatusBadge>
            </button>
          ))}
        </div>
      </Card>
      <Card title="Safety Snapshot" subtitle="Deployment state for the demo workflow." icon={ShieldAlert} className="lg:col-span-4">
        <div className="space-y-3 text-sm text-slate-700">
          <div className="rounded-2xl bg-amber-50 p-4"><b>Model status:</b> Demo model, not autonomous diagnosis.</div>
          <div className="rounded-2xl bg-sky-50 p-4"><b>Backend:</b> Local FastAPI scan processor or explicit hosted API base.</div>
          <div className="rounded-2xl bg-emerald-50 p-4"><b>Guardrail:</b> Clinician rationale required before sign-off.</div>
        </div>
      </Card>
    </div>
  );
}

function HomeScanPreview({ scan }) {
  if (scan?.previews?.overlay || scan?.previews?.raw) {
    const src = scan.previews.overlay || scan.previews.raw;
    return (
      <div className="flex h-full min-h-72 items-center justify-center">
        <img src={src} alt="Active OCT scan preview" className="max-h-80 w-full rounded-2xl object-contain" />
      </div>
    );
  }

  return (
    <div className="flex min-h-72 flex-col justify-between rounded-2xl border border-slate-700 bg-slate-900 p-5">
      <div className="flex items-center justify-between text-xs font-bold uppercase text-slate-400">
        <span>Preview Bay</span>
        <span>No scan</span>
      </div>
      <div className="space-y-2">
        {Array.from({ length: 14 }).map((_, index) => (
          <div
            key={index}
            className={`h-2 rounded-full ${
              index % 5 === 0
                ? "bg-cyan-300/70"
                : index % 3 === 0
                  ? "bg-emerald-300/65"
                  : "bg-slate-500/60"
            }`}
            style={{ width: `${62 + ((index * 17) % 34)}%`, marginLeft: `${(index * 11) % 24}%` }}
          />
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 text-xs font-semibold text-slate-300">
        <span className="rounded-xl bg-slate-800 p-3">Raw</span>
        <span className="rounded-xl bg-slate-800 p-3">Overlay</span>
        <span className="rounded-xl bg-slate-800 p-3">CDF</span>
      </div>
    </div>
  );
}

export function WorklistScreen() {
  const { scan } = useAppContext();
  const router = useRouter();
  const liveRisk = riskFromScan(scan);
  const rows = scan?.status === "completed"
    ? [["LOCAL-001", scan.source_format, liveRisk.label, liveRisk.confidence, liveRisk.action, liveRisk.tone], ...demoRows]
    : demoRows;

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Triage Queue" subtitle="AI rapidly processes routine scans and protects specialist bandwidth." icon={RouteIcon} className="lg:col-span-8">
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
          <button onClick={() => router.push("/QC")} className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-bold text-white">
            Upload new scan
          </button>
          {scan?.status === "completed" ? (
            <button onClick={() => router.push("/review")} className="rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700">
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

export function UploadScreen() {
  const { scan, uploadState, setUploadState, setDecision, setScan } = useAppContext();
  const router = useRouter();
  async function onUpload(file) {
    setUploadState({ status: "Uploading", progress: 20, fileName: file.name, error: "" });
    setDecision({ choice: "", rationale: "", submittedAt: "" });

    try {
      setUploadState({ status: "Processing", progress: 55, fileName: file.name, error: "" });
      const payload = await createScan(file);
      setScan(payload);
      setUploadState({ status: "Completed", progress: 100, fileName: file.name, error: "" });
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
  }
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
      <Card title="Scan Intake" subtitle=".vol, .dcm, zipped TIFF, or 2D Image (.png, .jpg, .tif, .tiff)" icon={Upload} className="lg:col-span-4">
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
          <input ref={inputRef} type="file" accept=".vol,.dcm,.zip,.tif,.tiff,image/png,image/jpeg,image/webp,image/tiff" className="hidden" onChange={(event) => handleFiles(event.target.files)} />
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

export function ReviewScreen() {
  const { scan } = useAppContext();
  const [viewMode, setViewMode] = useState("segmented");
  
  const completed = scan?.status === "completed";
  const risk = riskFromScan(scan);
  
  const l1 = scan?.level1;
  const l2 = scan?.level2;
  const l3 = scan?.level3;
  const l1Abnormal = l1?.prediction === "ABNORMAL";
  const gradcams = scan?.gradcams;

  const getClassColor = (className) => {
    if (className === "IRF") return "rgba(255, 255, 255, 0.7)"; // White (Edema/Supranormal)
    if (className === "SRF") return "rgba(239, 68, 68, 0.7)"; // Red (Edema)
    return "rgba(34, 197, 94, 0.3)"; // Green (Normal)
  };

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="OCT Image Classification" subtitle="Hugging Face & Local Segmentation" icon={ScanLine} className="lg:col-span-7">
        <div className="grid gap-4">
          <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-slate-950 flex flex-col justify-between" style={{ minHeight: "24rem" }}>
            {completed && scan.localImageUrl ? (
              <>
                <div className="absolute inset-0 flex items-center justify-center p-4">
                  <div className="relative max-h-full max-w-full">
                    <img src={scan.localImageUrl} alt="Uploaded OCT" className="max-h-full max-w-full block rounded" />
                    {viewMode === "segmented" && scan.segmentation && (
                      <svg viewBox="0 0 512 512" className="absolute top-0 left-0 h-full w-full pointer-events-none" preserveAspectRatio="none">
                        {(scan.segmentation.layers || []).map((layer, idx) => (
                          <polyline
                            key={`layer-${idx}`}
                            points={layer.boundary_points.map(p => `${p.x},${p.y}`).join(" ")}
                            fill="none"
                            stroke={getClassColor(layer.class_name).replace("0.3", "0.8").replace("0.7", "1.0")}
                            strokeWidth="2"
                          />
                        ))}
                        {(scan.segmentation.lesions || []).map((lesion, idx) => (
                          <polygon
                            key={`lesion-${idx}`}
                            points={lesion.polygon.map(p => `${p.x},${p.y}`).join(" ")}
                            fill={getClassColor(lesion.class_name)}
                            stroke={getClassColor(lesion.class_name).replace("0.3", "0.8").replace("0.7", "1.0")}
                            strokeWidth="2"
                          />
                        ))}
                      </svg>
                    )}
                  </div>
                </div>
                <div className="absolute bottom-4 left-0 right-0 flex justify-center z-10">
                  <div className="flex bg-slate-800/80 backdrop-blur-md rounded-xl p-1 gap-1 border border-slate-700">
                    <button
                      onClick={() => setViewMode("raw")}
                      className={`px-4 py-1.5 rounded-lg text-xs font-bold transition ${viewMode === "raw" ? "bg-white text-slate-900" : "text-slate-300 hover:text-white"}`}
                    >
                      Raw
                    </button>
                    <button
                      onClick={() => setViewMode("segmented")}
                      className={`px-4 py-1.5 rounded-lg text-xs font-bold transition ${viewMode === "segmented" ? "bg-white text-slate-900" : "text-slate-300 hover:text-white"}`}
                    >
                      Segmented
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <WireBox height="h-72">Upload a scan to view predictions</WireBox>
            )}
          </div>
        </div>
      </Card>

      <div className="lg:col-span-5 space-y-5">
        <Card title="Level 1: Gatekeeper (ResNet-50)" subtitle="Binary triage screening." icon={Brain}>
          <div className="space-y-3">
            <Metric label="Gatekeeper Prediction" value={l1?.prediction || "N/A"} tone={l1Abnormal ? "danger" : "safe"} />
            {l1?.confidence && (
              <Metric label="Model Confidence" value={`${Math.round(l1.confidence * 100)}%`} tone="neutral" />
            )}
            <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
              <b>Explainability:</b> Diagnosis is based on a ResNet-50 model analyzing spatial features to triage ABNORMAL vs NORMAL scans.
            </div>
          </div>
        </Card>
        
        <Card title="Level 2: Disease Router (EfficientNet-B2)" subtitle="Specific disease classification." icon={Activity}>
           <div className="space-y-3">
             {l1Abnormal ? (
               <>
                 <Metric label="Router Prediction" value={l2?.prediction?.replace(/_/g, " ") || "N/A"} tone="warning" />
                 {l2?.confidence && (
                   <Metric label="Model Confidence" value={`${Math.round(l2.confidence * 100)}%`} tone="neutral" />
                 )}
                 <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
                   <b>Safety check:</b> Specific pathology successfully identified.
                 </div>
               </>
             ) : (
               <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900 text-center">
                  Level 2 routing bypassed. Scan is healthy.
               </div>
             )}
           </div>
        </Card>

        {l1Abnormal && l3?.prediction && (
          <Card title="Level 3: Specialist (EfficientNet-B0)" subtitle="Fine-grained subclass verification." icon={Activity}>
            <div className="space-y-3">
              <Metric label="Specialist Diagnosis" value={l3.prediction.replace(/_/g, " ")} tone="danger" />
              {l3.confidence && (
                <Metric label="Model Confidence" value={`${Math.round(l3.confidence * 100)}%`} tone="neutral" />
              )}
              <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-700">
                <b>Deep analysis:</b> Specialist model activated to verify the exact subclass of the pathology.
              </div>
            </div>
          </Card>
        )}
      </div>

      {completed && gradcams && Object.keys(gradcams).length > 0 && (
        <Card title="Explainability: Grad-CAM Heatmaps" subtitle="Visualizing network attention mapping across the pipeline." icon={Activity} className="lg:col-span-12 mt-5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {gradcams.L1 && (
              <div className="space-y-2">
                <p className="text-sm font-bold text-slate-700 text-center">Level 1 (Gatekeeper)</p>
                <img src={gradcams.L1} alt="L1 Grad-CAM" className="w-full rounded-2xl border border-slate-200 object-contain" />
              </div>
            )}
            {gradcams.L2 && (
              <div className="space-y-2">
                <p className="text-sm font-bold text-slate-700 text-center">Level 2 (Router)</p>
                <img src={gradcams.L2} alt="L2 Grad-CAM" className="w-full rounded-2xl border border-slate-200 object-contain" />
              </div>
            )}
            {gradcams.L3 && (
              <div className="space-y-2">
                <p className="text-sm font-bold text-slate-700 text-center">Level 3 (Specialist)</p>
                <img src={gradcams.L3} alt="L3 Grad-CAM" className="w-full rounded-2xl border border-slate-200 object-contain" />
              </div>
            )}
          </div>
        </Card>
      )}
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

export function DecisionScreen() {
  const { scan, decision, setDecision } = useAppContext();
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

export function OutcomesScreen() {
  const { scan, decision } = useAppContext();
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

