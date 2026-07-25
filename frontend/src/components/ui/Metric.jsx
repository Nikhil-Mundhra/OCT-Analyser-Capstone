"use client";
import React from "react";

const TONES = {
  neutral: "bg-slate-950/80 text-slate-100 border-slate-800",
  safe:    "bg-emerald-950/40 text-emerald-300 border-emerald-500/30",
  warning: "bg-amber-950/40 text-amber-300 border-amber-500/30",
  danger:  "bg-rose-950/40 text-rose-300 border-rose-500/30",
  info:    "bg-sky-950/40 text-sky-300 border-sky-500/30",
};

export function Metric({ label, value, tone = "neutral" }) {
  return (
    <div className={`rounded-xl border p-3.5 ${TONES[tone]}`}>
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-1.5 text-xl font-bold font-mono tracking-tight">{value}</p>
    </div>
  );
}
