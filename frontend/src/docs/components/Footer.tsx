"use client";

import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="bg-[#232f3e] text-white py-8 px-6 border-t border-docs-header-border mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold tracking-wide">OCT Analyzer</span>
          <span className="text-slate-400 text-sm ml-2">© {new Date().getFullYear()} Capstone Project</span>
        </div>
        
        <div className="flex items-center gap-6 text-sm font-medium">
          <Link to="/docs/whitepaper" className="hover:text-sky-400 transition-colors">
            About Us
          </Link>
          <a href="https://kanerika.com/contact-us/" target="_blank" rel="noopener noreferrer" className="hover:text-sky-400 transition-colors">
            Contact Us
          </a>
        </div>
      </div>
    </footer>
  );
}
