"use client";
import React from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  CheckCircle2,
  ClipboardCheck,
  PlayCircle,
  Route as RouteIcon,
  ScanLine,
  ShieldAlert,
  Upload,
} from "lucide-react";
import { useAppContext } from "../../AppContext";
import { Card } from "../ui/Card";
import { StatusBadge } from "../ui/StatusBadge";
import { Metric } from "../ui/Metric";
import { riskFromScan } from "../utils/riskUtils";

/** Dark-background preview panel for the dashboard hero. */
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

export function DashboardScreen() {
  const { scan, uploadState, decision } = useAppContext();
  const router = useRouter();
  const risk = riskFromScan(scan);
  const completed = scan?.status === "completed";

  const workflow = [
    ["Intake",    "Upload OCT/OCTA export",            uploadState.fileName ? "Active" : "Ready", Upload],
    ["QC",        "Validate signal and volume shape",   completed ? "Passed" : "Pending",          CheckCircle2],
    ["Review",    "Inspect previews and layer evidence",completed ? "Ready" : "Waiting",            ScanLine],
    ["Decision",  "Capture clinician rationale",        decision.choice ? "Saved" : "Open",        ClipboardCheck],
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
                onClick={() => router.push("/qc")}
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
              onClick={() => router.push(title === "Intake" ? "/qc" : title === "Review" ? "/review" : title === "Decision" ? "/human-check" : "/qc")}
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
