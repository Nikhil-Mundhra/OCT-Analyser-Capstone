"use client";
import React from 'react';
import Link from "next/link";
import { ScanLine, ArrowRight, ShieldAlert, Activity, Route } from 'lucide-react';

import { HomeNav, ROUTES } from './components';

const FEATURE_CARDS = [
  {
    icon: Route,
    color: 'text-sky-500',
    title: 'Automated Triage',
    desc: 'AI rapidly processes routine scans, routing high-risk and ambiguous cases to specialists.',
  },
  {
    icon: Activity,
    color: 'text-emerald-500',
    title: 'Explainable AI',
    desc: 'Clear confidence scores, uncertainty metrics, and overlay evidence for clinical validation.',
  },
  {
    icon: ShieldAlert,
    color: 'text-amber-500',
    title: 'Human-in-the-Loop',
    desc: 'Critical sign-offs require active human rationale to prevent automation bias.',
  },
];

function FeatureCard({ icon: Icon, color, title, desc }) {
  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 text-left">
      <Icon className={`h-8 w-8 ${color} mb-4`} />
      <h3 className="font-bold text-slate-900 text-lg mb-2">{title}</h3>
      <p className="text-slate-600 text-sm">{desc}</p>
    </div>
  );
}

export function HomePage() {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <HomeNav />

      <main className="flex-1 flex flex-col items-center justify-center p-8 text-center">
        <h1 className="text-5xl font-black text-slate-900 mb-6 tracking-tight">
          Clinical OCT Inference Engine
        </h1>
        <p className="text-lg text-slate-600 max-w-2xl mb-10 leading-relaxed">
          An end-to-end clinician workflow for triage, explainable scan review, active human justification, specialist routing, and safety monitoring.{' '}
          Powered by deep learning for 3D OCT and OCTA volumes.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl w-full mb-12">
          {FEATURE_CARDS.map((card) => (
            <FeatureCard key={card.title} {...card} />
          ))}
        </div>

        <Link
          href={ROUTES.QC}
          className="inline-flex items-center gap-2 bg-slate-900 text-white px-8 py-4 rounded-full text-lg font-bold hover:bg-slate-800 transition shadow-lg hover:shadow-xl"
        >
          Enter Clinical Workspace <ArrowRight className="h-5 w-5" />
        </Link>
      </main>

      <footer className="py-6 text-center text-slate-500 text-sm border-t border-slate-200 bg-white">
        {'\u00A9'} {new Date().getFullYear()} OCT Analyzer Capstone Project. Internal Use Only.
      </footer>
    </div>
  );
}
