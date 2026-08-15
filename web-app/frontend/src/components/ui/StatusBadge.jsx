"use client";
import React from "react";

const TONES = {
  neutral: "bg-slate-100 text-slate-700 border-slate-200",
  safe:    "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-800 border-amber-200",
  danger:  "bg-rose-50 text-rose-700 border-rose-200",
  info:    "bg-sky-50 text-sky-700 border-sky-200",
  purple:  "bg-violet-50 text-violet-700 border-violet-200",
};

export function StatusBadge({ children, tone = "neutral" }) {
  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${TONES[tone]}`}>
      {children}
    </span>
  );
}
