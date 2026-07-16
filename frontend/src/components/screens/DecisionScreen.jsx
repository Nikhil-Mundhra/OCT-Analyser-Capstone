"use client";
import React, { useCallback, useMemo, useState } from "react";
import { ClipboardCheck, Download, Stethoscope } from "lucide-react";
import { useAppContext } from "../../AppContext";
import { Card } from "../ui/Card";
import { WireBox } from "../ui/WireBox";
import { riskFromScan } from "../utils/riskUtils";
import { MIN_RATIONALE_LENGTH } from "../constants";

function downloadCaseJson(scan, decision) {
  return () => {
    if (!scan) return;
    const blob = new Blob([JSON.stringify({ scan, decision }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${scan.scan_id || "oct-case"}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };
}

export function DecisionScreen() {
  const { scan, decision, completeDecision } = useAppContext();
  const [choice, setChoice] = useState(decision.choice || "");
  const [rationale, setRationale] = useState(decision.rationale || "");
  const risk = riskFromScan(scan);

  const canSubmit = useMemo(
    () => Boolean(choice && rationale.trim().length >= MIN_RATIONALE_LENGTH && scan?.status === "completed"),
    [choice, rationale, scan?.status]
  );

  const submitDecision = useCallback(() => {
    if (!canSubmit) return;
    completeDecision(choice, rationale);
  }, [canSubmit, completeDecision, choice, rationale]);

  const handleDownload = useCallback(() => {
    downloadCaseJson(scan, decision)();
  }, [scan, decision]);

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Specialist Routing Packet" subtitle="High-risk or ambiguous scans are routed with history and highlighted anomalies." icon={Stethoscope} className="lg:col-span-5">
        <div className="space-y-3 text-sm text-slate-700">
          <div className="rounded-2xl bg-slate-50 p-4"><b>Recipient:</b> {risk.tone === "danger" ? "Retina specialist review queue" : "Ophthalmology review queue"}</div>
          <div className={`rounded-2xl p-4 ${risk.tone === "danger" ? "bg-rose-50" : "bg-slate-50"}`}><b>Reason:</b> {risk.action}</div>
          <div className="rounded-2xl bg-slate-50 p-4"><b>Attached:</b> OCT volume, segmentation overlay, CDF chart, layer vote table</div>
          {scan?.previews?.overlay
            ? <img src={scan.previews.overlay} alt="Anomaly snapshot" className="h-40 w-full rounded-2xl border object-contain" />
            : <WireBox height="h-40">Anomaly snapshot summary</WireBox>}
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
              onChange={(e) => setRationale(e.target.value)}
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
          <button onClick={() => setChoice("Defer to specialist")} className="rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700">
            Request second opinion
          </button>
          <button onClick={handleDownload} className="inline-flex items-center gap-2 rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700">
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
