"use client";
import React from "react";
import { StatusBadge } from "../ui/StatusBadge";

export function Header({ scan }) {
  return (
    <header className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="mb-2 flex flex-wrap gap-2">
            <StatusBadge tone="info">Clinical decision support</StatusBadge>
            <StatusBadge tone="purple">Human in the loop</StatusBadge>
            <StatusBadge tone="warning">Not autonomous diagnosis</StatusBadge>
            {scan?.is_demo_model ? <StatusBadge tone="warning">Demo model</StatusBadge> : null}
          </div>
          <h1 className="text-3xl font-black tracking-tight text-slate-950">OCT/OCTA Clinical Inference Interface</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            End-to-end clinician workflow for triage, explainable scan review, active human justification, specialist routing, and safety monitoring.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="rounded-2xl bg-slate-50 p-4">
            <p className="font-semibold text-slate-500">Current user</p>
            <p className="mt-1 font-bold text-slate-900">Ophthalmologist</p>
          </div>
          <div className="rounded-2xl bg-slate-50 p-4">
            <p className="font-semibold text-slate-500">Mode</p>
            <p className="mt-1 font-bold text-slate-900">{scan ? "Active case" : "Review queue"}</p>
          </div>
        </div>
      </div>
    </header>
  );
}
