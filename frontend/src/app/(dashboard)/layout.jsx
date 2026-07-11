"use client";
import React, { useMemo } from 'react';
import { HomeNav, Sidebar, Header, screens, CaseSwitcher } from '../../components';
import { useAppContext } from '../../AppContext';
import { usePathname } from 'next/navigation';
import { Clock, AlertTriangle, UserRound, Eye } from 'lucide-react';

function StatusBadge({ children, tone = "neutral" }) {
  const tones = {
    neutral: "bg-slate-100 text-slate-700 border-slate-200",
    safe: "bg-emerald-50 text-emerald-700 border-emerald-200",
    warning: "bg-amber-50 text-amber-800 border-amber-200",
    danger: "bg-rose-50 text-rose-700 border-rose-200",
    info: "bg-sky-50 text-sky-700 border-sky-200",
    purple: "bg-violet-50 text-violet-700 border-violet-200",
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

export default function DashboardLayout({ children }) {
  const { scan } = useAppContext();
  const pathname = usePathname();
  const activeScreen = useMemo(() => screens.find((screen) => screen.path === pathname) || screens[0], [pathname]);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-50 text-slate-900 font-sans">
      <HomeNav />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar />
        <main className="flex-1 overflow-y-auto pl-[72px]">
          <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
            {pathname === '/dashboard' && <Header scan={scan} />}
            <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-6 flex flex-col justify-between gap-3 lg:flex-row lg:items-center pb-4 border-b border-slate-100">
                <div>
                  <p className="text-xs font-bold uppercase tracking-widest text-slate-500">Clinical Workspace</p>
                  <div className="flex items-center gap-4 mt-1">
                    <h2 className="text-2xl font-black text-slate-950">{activeScreen?.label}</h2>
                    <CaseSwitcher />
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <StatusBadge tone="info"><Clock className="mr-1 h-3 w-3" /> Fast response</StatusBadge>
                  <StatusBadge tone="warning"><AlertTriangle className="mr-1 h-3 w-3" /> Confidence shown</StatusBadge>
                  <StatusBadge tone="purple"><UserRound className="mr-1 h-3 w-3" /> Human justification</StatusBadge>
                  <StatusBadge tone="safe"><Eye className="mr-1 h-3 w-3" /> Visual evidence</StatusBadge>
                </div>
              </div>
              {children}
            </section>
            <footer className="grid gap-4 rounded-3xl border border-slate-200 bg-white p-5 text-sm text-slate-600 shadow-sm lg:grid-cols-4">
              <div><b className="text-slate-900">Triage:</b> low-risk cases move quickly, uncertain cases are escalated.</div>
              <div><b className="text-slate-900">Transparency:</b> confidence, uncertainty, and overlays are visible.</div>
              <div><b className="text-slate-900">Safety:</b> critical sign-off requires human rationale.</div>
              <div><b className="text-slate-900">Evaluation:</b> monitor outcomes, latency, overrides, and safety signals.</div>
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}
