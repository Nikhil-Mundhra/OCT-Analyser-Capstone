import { Link } from "react-router-dom";
import React from "react";
import { HomeNav } from "../../HomePage";
import Footer from "./Footer";
import DocsSidebar from "./DocsSidebar";

interface DocLayoutProps {
  children: React.ReactNode;
  toc?: React.ReactNode;
}

export default function DocLayout({ children, toc }: DocLayoutProps) {
  return (
    <div className="docs-theme flex flex-col min-h-screen bg-docs-bg-page text-docs-text-primary selection:bg-blue-500/30 font-sans transition-colors duration-200">
      <HomeNav />

      <div className="flex flex-1 w-full relative">
        <DocsSidebar />

        <main className="flex-1 min-w-0 px-6 py-12 md:px-12 lg:px-16">
          {/* Back to index link */}
          <Link to="/docs"
            className="inline-flex items-center gap-2 text-sm text-slate-500 dark:text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 transition-colors mb-10 group"
          >
            <svg
              className="w-4 h-4 transition-transform group-hover:-translate-x-0.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
            Back to Docs
          </Link>

          <div className="max-w-3xl mx-auto">
            {children}
          </div>
        </main>

        {toc && toc}
      </div>

      <Footer />
    </div>
  );
}
