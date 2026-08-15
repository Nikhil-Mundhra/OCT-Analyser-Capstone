"use client";
import React from "react";
import { BarChart3, FileText, History } from "lucide-react";
import { useAppContext } from "../../AppContext";
import { Card } from "../ui/Card";
import { Metric } from "../ui/Metric";

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
