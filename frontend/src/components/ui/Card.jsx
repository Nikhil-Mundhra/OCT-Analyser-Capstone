"use client";
import React from "react";

export function Card({ title, subtitle, icon: Icon, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-slate-800 bg-slate-900/90 backdrop-blur-md p-5 shadow-xl text-slate-100 ${className}`}>
      <div className="mb-4 flex items-start gap-3">
        {Icon ? (
          <div className="rounded-xl bg-slate-800 border border-slate-700 p-2 text-sky-400">
            <Icon className="h-5 w-5" />
          </div>
        ) : null}
        <div>
          <h3 className="text-base font-bold text-sky-400">{title}</h3>
          {subtitle ? <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}
