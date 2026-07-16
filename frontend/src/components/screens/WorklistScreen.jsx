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
        <div className="overflow-hidden rounded-2xl border border-slate-200">
          <div className="grid grid-cols-5 bg-slate-100 px-4 py-3 text-xs font-bold uppercase tracking-wide text-slate-500">
            <span>Patient</span>
            <span>Scan</span>
            <span>AI route</span>
            <span>Confidence</span>
            <span>Action</span>
          </div>

          {scanHistory.length === 0 ? (
            <div className="p-8 text-center text-slate-500">No scans in triage queue. Upload a new scan to get started.</div>
          ) : (
            scanHistory.map((s) => {
              const liveRisk = riskFromScan(s);
              return (
                <div key={s.scan_id} className="grid grid-cols-5 items-center border-t border-slate-100 px-4 py-4 text-sm text-slate-700">
                  <span className="font-bold text-slate-900">{s.patient_id || "Unknown"}</span>
                  <span>{s.scan_type || s.source_format}</span>
                  <span><StatusBadge tone={liveRisk.tone}>{liveRisk.label}</StatusBadge></span>
                  <span className="font-bold">{liveRisk.confidence}</span>
                  <span className="flex gap-2">
                    <button onClick={() => handleView(s)} className="rounded bg-sky-50 px-2 py-1 text-xs font-bold text-sky-700 hover:bg-sky-100 transition-colors">
                      View
                    </button>
                    <button onClick={() => deleteScan(s.scan_id || s.id)} className="rounded bg-rose-50 p-1 text-rose-700 hover:bg-rose-100 transition-colors" title="Delete">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </span>
                </div>
              );
            })
          )}
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button onClick={() => router.push("/QC")} className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-bold text-white shadow-sm">
            Upload new scan
          </button>
          {scan?.status === "completed" ? (
            <button onClick={() => router.push("/review")} className="rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700 shadow-sm">
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
