"use client";
import React, { useState } from "react";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import {
  Activity,
  ChevronDown,
  ChevronUp,
  LayoutDashboard,
  Plus,
  Search,
  UploadCloud,
} from "lucide-react";
import { screens } from "../constants";

/** Width of the collapsed sidebar — keep in sync with the Tailwind class w-[72px] on the <aside>. */
const SIDEBAR_WIDTH_COLLAPSED_PX = 72; // eslint-disable-line no-unused-vars

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [expanded, setExpanded] = useState({ clinical: true, system: false });

  const toggleSection = (section) => {
    setExpanded((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const clinicalScreens = screens.filter((s) => s.group === "clinical");

  return (
    <aside className="group w-[72px] hover:w-[260px] transition-all duration-300 ease-in-out border-r border-slate-200 bg-white flex flex-col h-full shrink-0 shadow-sm z-40 overflow-hidden absolute md:relative">
      <div className="p-4">
        <button
          onClick={() => router.push("/QC")}
          className="flex items-center justify-center gap-0 group-hover:gap-3 mx-auto h-11 w-11 group-hover:w-full group-hover:h-auto bg-sky-500 hover:bg-sky-600 text-white rounded-full group-hover:py-3 group-hover:px-4 font-bold transition-all shadow-sm overflow-hidden"
        >
          <Plus className="h-5 w-5 shrink-0" />
          <span className="whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-300 w-0 group-hover:w-auto">New Upload</span>
        </button>
      </div>

      <div className="px-4 pb-4">
        <div className="relative flex items-center justify-center group-hover:justify-start">
          <Search className="absolute left-3 h-5 w-5 text-slate-400 shrink-0" />
          <input
            type="text"
            placeholder="Search"
            className="w-full rounded-full border border-slate-300 py-2 pl-10 pr-4 text-sm outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 transition opacity-0 group-hover:opacity-100 cursor-default group-hover:cursor-text"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-2 space-y-1">
        <Link
          href="/dashboard"
          className={`flex items-center gap-3 rounded-lg px-4 py-3 text-sm font-medium transition-colors overflow-hidden whitespace-nowrap ${
            pathname === "/dashboard" ? "bg-slate-100 text-slate-900 font-bold" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          <LayoutDashboard className={`h-5 w-5 shrink-0 ${pathname === "/dashboard" ? "text-sky-500" : "text-slate-400"}`} />
          <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">Home</span>
        </Link>

        {/* Clinical Flow section */}
        <div>
          <button
            onClick={() => toggleSection("clinical")}
            className="flex items-center justify-between w-full rounded-lg px-4 py-3 text-sm font-medium text-slate-600 hover:bg-slate-50 transition-colors overflow-hidden whitespace-nowrap"
          >
            <div className="flex items-center gap-3">
              <UploadCloud className="h-5 w-5 text-slate-400 shrink-0" />
              <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">Clinical Flow</span>
            </div>
            <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              {expanded.clinical ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </div>
          </button>

          <div className={`overflow-hidden transition-all duration-300 ease-in-out ${expanded.clinical ? "max-h-0 opacity-0 group-hover:max-h-96 group-hover:opacity-100" : "max-h-0 opacity-0"}`}>
            <ul className="mt-1 space-y-1 pl-[3.25rem]">
              {clinicalScreens.map((screen) => (
                <li key={screen.id}>
                  <Link
                    href={screen.path}
                    className={`block rounded-lg px-3 py-2 text-sm transition-colors whitespace-nowrap opacity-0 group-hover:opacity-100 duration-300 ${
                      pathname === screen.path ? "bg-slate-100 text-slate-900 font-bold" : "text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    {screen.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* System & Admin section */}
        <div>
          <button
            onClick={() => toggleSection("system")}
            className="flex items-center justify-between w-full rounded-lg px-4 py-3 text-sm font-medium text-slate-600 hover:bg-slate-50 transition-colors overflow-hidden whitespace-nowrap"
          >
            <div className="flex items-center gap-3">
              <Activity className="h-5 w-5 text-slate-400 shrink-0" />
              <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">System &amp; Admin</span>
            </div>
            <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              {expanded.system ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </div>
          </button>

          <div className={`overflow-hidden transition-all duration-300 ease-in-out ${expanded.system ? "max-h-0 opacity-0 group-hover:max-h-96 group-hover:opacity-100" : "max-h-0 opacity-0"}`}>
            <ul className="mt-1 space-y-1 pl-[3.25rem]">
              <li>
                <Link
                  href="/outcomes"
                  className={`block rounded-lg px-3 py-2 text-sm transition-colors whitespace-nowrap opacity-0 group-hover:opacity-100 duration-300 ${
                    pathname === "/outcomes" ? "bg-slate-100 text-slate-900 font-bold" : "text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  Clinical Outcomes
                </Link>
              </li>
              <li>
                <Link
                  href="/docs"
                  className={`block rounded-lg px-3 py-2 text-sm transition-colors whitespace-nowrap opacity-0 group-hover:opacity-100 duration-300 ${
                    pathname === "/docs" ? "bg-slate-100 text-slate-900 font-bold" : "text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  System Documentation
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </aside>
  );
}
