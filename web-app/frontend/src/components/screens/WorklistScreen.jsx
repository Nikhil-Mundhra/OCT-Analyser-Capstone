"use client";
import React, { useCallback } from "react";
import { useRouter } from "next/navigation";
import { Route as RouteIcon, ShieldAlert, Trash2 } from "lucide-react";
import { useAppContext } from "../../AppContext";
import { Card } from "../ui/Card";
import { StatusBadge } from "../ui/StatusBadge";
import { riskFromScan } from "../utils/riskUtils";

export function WorklistScreen() {
  const { scan, scanHistory, setScan, deleteScan, setUploadState } = useAppContext();
  const router = useRouter();

  const handleView = useCallback((selectedScan) => {
    setScan(selectedScan);
    setUploadState({ status: "Completed", progress: 100, fileName: selectedScan.filename || "", error: "" });
    router.push("/review");
  }, [setScan, setUploadState, router]);

  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <Card title="Triage Queue" subtitle="AI rapidly processes routine scans and protects specialist bandwidth." icon={RouteIcon} className="lg:col-span-8">
        <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950">
          <div className="grid grid-cols-5 bg-slate-900 border-b border-slate-800 px-4 py-3 text-xs font-bold uppercase tracking-wider text-slate-400">
            <span>Patient</span>
            <span>Scan</span>
            <span>AI route</span>
            <span>Confidence</span>
            <span>Action</span>
          </div>

          {scanHistory.length === 0 ? (
            <div className="p-8 text-center text-slate-400 text-xs">No scans in triage queue. Upload a new scan to get started.</div>
          ) : (
            scanHistory.map((s) => {
              const liveRisk = riskFromScan(s);
              return (
                <div key={s.scan_id} className="grid grid-cols-5 items-center border-t border-slate-800/80 px-4 py-3.5 text-xs text-slate-300 hover:bg-slate-900/50 transition-all">
                  <span className="font-bold font-mono text-slate-100">{s.patient_id || "Unknown"}</span>
                  <span className="text-slate-300 truncate pr-2">{s.scan_type || s.source_format}</span>
                  <span><StatusBadge tone={liveRisk.tone}>{liveRisk.label}</StatusBadge></span>
                  <span className="font-bold font-mono text-sky-400">{liveRisk.confidence}</span>
                  <span className="flex items-center gap-2">
                    <button onClick={() => handleView(s)} className="rounded-lg bg-sky-500/20 text-sky-400 hover:bg-sky-500/30 border border-sky-500/30 px-3 py-1 text-xs font-bold transition-all">
                      View
                    </button>
                    <button onClick={() => deleteScan(s.scan_id || s.id)} className="rounded-lg bg-rose-500/20 text-rose-400 hover:bg-rose-500/30 border border-rose-500/30 p-1.5 transition-all" title="Delete">
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </span>
                </div>
              );
            })
          )}
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button onClick={() => router.push("/qc")} className="rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 px-5 py-2.5 text-xs font-bold transition-all shadow-md">
            Upload new scan
          </button>
          {scan?.status === "completed" ? (
            <button onClick={() => router.push("/review")} className="rounded-xl border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 px-5 py-2.5 text-xs font-bold transition-all">
              Review active case
            </button>
          ) : null}
        </div>
      </Card>

      <Card title="Triage Rules" subtitle="Routing logic is visible to reduce blind trust." icon={ShieldAlert} className="lg:col-span-4">
        <div className="space-y-3 text-xs">
          <div className="rounded-xl bg-emerald-950/40 border border-emerald-500/30 p-3.5 text-emerald-200"><b className="text-emerald-400">Low risk:</b> routine queue, optional clinician sampling.</div>
          <div className="rounded-xl bg-amber-950/40 border border-amber-500/30 p-3.5 text-amber-200"><b className="text-amber-400">Ambiguous:</b> requires human audit before report sign-off.</div>
          <div className="rounded-xl bg-rose-950/40 border border-rose-500/30 p-3.5 text-rose-200"><b className="text-rose-400">High risk:</b> route to specialist with anomaly overlays and history.</div>
          <div className="rounded-xl bg-slate-950 border border-slate-800 p-3.5 text-slate-300"><b className="text-slate-400">Poor QC:</b> no model claim, request re-upload or manual inspection.</div>
        </div>
      </Card>
    </div>
  );
}
