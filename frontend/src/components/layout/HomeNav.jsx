"use client";
import React from "react";
import Link from "next/link";
import { ScanLine } from "lucide-react";

export function HomeNav() {
  return (
    <nav className="bg-[#232f3e] text-white py-4 px-6 flex justify-between items-center shadow-md">
      <Link href="/" className="flex items-center gap-3">
        <ScanLine className="h-6 w-6 text-sky-400" />
        <span className="text-lg font-bold tracking-wide">OCT Analyzer</span>
      </Link>
      <div className="flex items-center gap-4 text-sm font-medium">
        <Link href="/docs/readme" className="hover:text-sky-400 transition">Documentation</Link>
        <Link href="/QC" className="bg-sky-500 hover:bg-sky-600 text-white px-4 py-2 rounded font-bold transition">
          Launch App
        </Link>
      </div>
    </nav>
  );
}
