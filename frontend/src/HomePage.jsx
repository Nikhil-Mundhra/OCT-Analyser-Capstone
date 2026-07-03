import React from 'react';
import { Link } from 'react-router-dom';
import { ScanLine, ArrowRight, ShieldAlert, Activity, Route } from 'lucide-react';

export function HomeNav() {
  return (
    <nav className="bg-[#232f3e] text-white py-4 px-6 flex justify-between items-center shadow-md">
      <div className="flex items-center gap-3">
        <ScanLine className="h-6 w-6 text-sky-400" />
        <span className="text-lg font-bold tracking-wide">OCT Analyzer</span>
      </div>
      <div className="flex items-center gap-4 text-sm font-medium">
        <Link to="/docs/readme" className="hover:text-sky-400 transition">Documentation</Link>
        <Link to="/docs/implementation-info" className="hover:text-sky-400 transition">Architecture</Link>
        <Link to="/QC" className="bg-sky-500 hover:bg-sky-600 text-white px-4 py-2 rounded font-bold transition">
          Launch App
        </Link>
      </div>
    </nav>
  );
}

export function HomePage() {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      {/* Navigation matching awsnav/AWS style loosely while importing the CSS */}
      <HomeNav />
      
      <main className="flex-1 flex flex-col items-center justify-center p-8 text-center">
        <h1 className="text-5xl font-black text-slate-900 mb-6 tracking-tight">
          Clinical OCT Inference Engine
        </h1>
        <p className="text-lg text-slate-600 max-w-2xl mb-10 leading-relaxed">
          An end-to-end clinician workflow for triage, explainable scan review, active human justification, specialist routing, and safety monitoring. 
          Powered by deep learning for 3D OCT and OCTA volumes.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl w-full mb-12">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 text-left">
            <Route className="h-8 w-8 text-sky-500 mb-4" />
            <h3 className="font-bold text-slate-900 text-lg mb-2">Automated Triage</h3>
            <p className="text-slate-600 text-sm">AI rapidly processes routine scans, routing high-risk and ambiguous cases to specialists.</p>
          </div>
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 text-left">
            <Activity className="h-8 w-8 text-emerald-500 mb-4" />
            <h3 className="font-bold text-slate-900 text-lg mb-2">Explainable AI</h3>
            <p className="text-slate-600 text-sm">Clear confidence scores, uncertainty metrics, and overlay evidence for clinical validation.</p>
          </div>
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 text-left">
            <ShieldAlert className="h-8 w-8 text-amber-500 mb-4" />
            <h3 className="font-bold text-slate-900 text-lg mb-2">Human-in-the-Loop</h3>
            <p className="text-slate-600 text-sm">Critical sign-offs require active human rationale to prevent automation bias.</p>
          </div>
        </div>

        <Link 
          to="/QC" 
          className="inline-flex items-center gap-2 bg-slate-900 text-white px-8 py-4 rounded-full text-lg font-bold hover:bg-slate-800 transition shadow-lg hover:shadow-xl"
        >
          Enter Clinical Workspace <ArrowRight className="h-5 w-5" />
        </Link>
      </main>
      
      <footer className="py-6 text-center text-slate-500 text-sm border-t border-slate-200 bg-white">
        &copy; {new Date().getFullYear()} OCT Analyzer Capstone Project. Internal Use Only.
      </footer>
    </div>
  );
}
