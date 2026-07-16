"use client";
import React from "react";

const TONES = {
  neutral: "bg-slate-50 text-slate-900 border-slate-200",
  safe:    "bg-emerald-50 text-emerald-800 border-emerald-200",
  warning: "bg-amber-50 text-amber-900 border-amber-200",
  danger:  "bg-rose-50 text-rose-800 border-rose-200",
  info:    "bg-sky-50 text-sky-800 border-sky-200",
};

export function Metric({ label, value, tone = "neutral" }) {
  return (
    <div className={`rounded-2xl border p-4 ${TONES[tone]}`}>
      <p className="text-xs font-semibold uppercase tracking-wide opacity-70">{label}</p>
      <p className="mt-2 text-2xl font-black">{value}</p>
    </div>
  );
}
