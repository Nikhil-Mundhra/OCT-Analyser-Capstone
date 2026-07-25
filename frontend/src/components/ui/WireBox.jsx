"use client";
import React from "react";

export function WireBox({ className = "", height = "h-24", children }) {
  return (
    <div className={`rounded-2xl border-2 border-dashed border-slate-800 bg-slate-950/60 ${height} ${className}`}>
      <div className="flex h-full items-center justify-center p-4 text-center text-xs font-semibold text-slate-500">
        {children}
      </div>
    </div>
  );
}
