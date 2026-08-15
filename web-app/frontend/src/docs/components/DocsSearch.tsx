"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { docCategories, DocCategoryColor } from "../manifest";

interface SearchItem {
  slug: string;
  title: string;
  description: string;
  categoryTitle: string;
  categoryColor: DocCategoryColor;
}

export default function DocsSearch() {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  // Flatten and filter doc categories based on query
  const searchResults = useMemo(() => {
    if (!query.trim()) return [];

    const searchLower = query.toLowerCase();
    const results: SearchItem[] = [];

    docCategories.forEach((category) => {
      category.articles.forEach((article) => {
        if (
          article.title.toLowerCase().includes(searchLower) ||
          article.description.toLowerCase().includes(searchLower)
        ) {
          results.push({
            slug: article.slug,
            title: article.title,
            description: article.description,
            categoryTitle: category.title,
            categoryColor: category.color,
          });
        }
      });
    });

    return results.slice(0, 8); // Limit to top 8 results
  }, [query]);

  // Handle Cmd+K / Ctrl+K
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsOpen(true);
        setActiveIndex(0);
      }
    };

    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, []);

  // Handle clicking outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        isOpen &&
        modalRef.current &&
        !modalRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
        setTimeout(() => setQuery(""), 200);
      }
    };

    if (isOpen) {
      window.addEventListener("mousedown", handleClickOutside);
    }
    return () => window.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen) {
      // Small timeout to allow render before focusing
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  const quickLinks = [
    { slug: "enablement-guide", title: "Enablement Guide", description: "Learn how to setup workload environments." },
    { slug: "user-guide", title: "User Guide", description: "Complete lifecycle from trial to adoption." },
    { slug: "security-document", title: "Security Overview", description: "Compliance and infrastructure policies." },
    { slug: "pricing", title: "Pricing Details", description: "Information on credits and tiers." }
  ];

  // Handle keyboard navigation within the dropdown
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setIsOpen(false);
      setTimeout(() => setQuery(""), 200);
      return;
    }

    const isSearching = !!query.trim();
    const currentListLength = isSearching ? searchResults.length : quickLinks.length;
    if (currentListLength === 0) return;

    if (isSearching) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((prev) => (prev + 1) % currentListLength);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((prev) => (prev - 1 + currentListLength) % currentListLength);
      }
    } else {
      // 2D grid navigation for Quick Links (2x2 layout)
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((prev) => (prev + 2) % 4);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((prev) => (prev - 2 + 4) % 4);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setActiveIndex((prev) => {
          const row = Math.floor(prev / 2);
          const col = prev % 2;
          const nextCol = (col + 1) % 2;
          return row * 2 + nextCol;
        });
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setActiveIndex((prev) => {
          const row = Math.floor(prev / 2);
          const col = prev % 2;
          const nextCol = (col - 1 + 2) % 2;
          return row * 2 + nextCol;
        });
      }
    }

    if (e.key === "Enter") {
      e.preventDefault();
      const selectedItem = isSearching ? searchResults[activeIndex] : quickLinks[activeIndex];
      if (selectedItem) {
        router.push(`/docs/${selectedItem.slug}`);
        setIsOpen(false);
        setTimeout(() => setQuery(""), 200);
      }
    }
  };

  const getCategoryColorClass = (color: DocCategoryColor) => {
    switch (color) {
      case "blue": return "text-blue-400 bg-blue-500/10";
      case "purple": return "text-purple-400 bg-purple-500/10";
      case "emerald": return "text-emerald-400 bg-emerald-500/10";
      default: return "text-slate-400 bg-slate-500/10";
    }
  };

  return (
    <>
      {/* Search Trigger Button */}
      <button 
        type="button"
        className="flex items-center gap-2 px-[18px] py-[8px] bg-docs-active-link-bg/60 hover:bg-docs-active-link-bg border border-docs-border-main rounded-md transition-colors duration-200 group cursor-pointer"
        onClick={() => {
          setIsOpen(true);
          setActiveIndex(0);
        }}
      >
        <svg 
          className="w-4 h-4 text-docs-text-muted group-hover:text-docs-text-primary transition-colors duration-200" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <span className="text-[13.5px] font-semibold text-docs-text-secondary group-hover:text-docs-text-primary transition-colors duration-200 hidden sm:block">Search</span>
        <kbd className="hidden sm:inline-block font-sans text-[10px] font-medium text-docs-text-muted bg-docs-bg-page border border-docs-border-main rounded px-1.5 py-0.5 ml-2 transition-colors duration-200">
          ⌘K
        </kbd>
      </button>

      {/* Modal Overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[10vh] px-4 sm:px-6">
          {/* Backdrop */}
          <div 
            className="fixed inset-0 bg-docs-text-primary/10 transition-opacity duration-200" 
            aria-hidden="true" 
          />
          
          {/* Modal Content */}
          <div 
            ref={modalRef}
            className="relative w-full max-w-2xl bg-docs-bg-card border border-docs-border-main rounded-2xl shadow-2xl overflow-hidden transition-colors duration-200"
          >
            {/* Input Header */}
            <div className="flex items-center px-4 border-b border-docs-border-main transition-colors duration-200">
              <svg className="w-5 h-5 text-docs-text-muted mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setActiveIndex(0);
                }}
                onKeyDown={handleKeyDown}
                placeholder="Search documentation..."
                className="w-full h-14 bg-transparent border-none outline-none text-lg text-docs-text-primary placeholder:text-docs-text-muted transition-colors duration-200"
              />
              <button 
                className="p-1 text-docs-text-muted hover:text-docs-text-primary transition-colors duration-200 rounded-md hover:bg-docs-active-link-bg"
                onClick={() => setIsOpen(false)}
              >
                <kbd className="font-sans text-[10px] font-medium px-1.5 py-0.5">ESC</kbd>
              </button>
            </div>

            {/* Results Body */}
            {query.trim() !== "" ? (
              <div className="max-h-[60vh] overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-docs-border-main scrollbar-track-transparent">
                {searchResults.length > 0 ? (
                  <ul>
                    {searchResults.map((item, index) => (
                      <li key={item.slug}>
                        <Link href={`/docs/${item.slug}`}
                          onClick={() => {
                            setIsOpen(false);
                            setTimeout(() => setQuery(""), 200);
                          }}
                          className={`block px-4 py-3 rounded-xl transition-colors duration-200 ${
                            index === activeIndex ? "bg-docs-active-link-bg" : "hover:bg-docs-active-link-bg/50"
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-base font-medium text-docs-text-primary truncate pr-4 transition-colors duration-200">
                              {item.title}
                            </span>
                            <span className={`text-[11px] px-2.5 py-0.5 rounded-full font-medium whitespace-nowrap ${getCategoryColorClass(item.categoryColor)}`}>
                              {item.categoryTitle}
                            </span>
                          </div>
                          <p className="text-sm text-docs-text-secondary line-clamp-1 transition-colors duration-200">
                            {item.description}
                          </p>
                        </Link>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="py-14 text-center text-docs-text-secondary transition-colors duration-200">
                    <p className="mb-2 text-lg">No results found</p>
                    <p className="text-sm">We couldn&apos;t find anything matching &quot;{query}&quot;</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4 sm:p-6 text-sm text-docs-text-secondary transition-colors duration-200">
                <p className="mb-4 text-xs font-semibold tracking-wider uppercase text-docs-text-muted transition-colors duration-200">Quick Links</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {quickLinks.map((link, index) => (
                    <Link
                      key={link.slug}
                      href={`/docs/${link.slug}`}
                      onClick={() => { setIsOpen(false); setTimeout(() => setQuery(""), 200); }}
                      className={`text-left px-3 py-2 rounded-lg transition-colors duration-200 block group ${
                        index === activeIndex ? "bg-docs-active-link-bg" : "hover:bg-docs-active-link-bg/50"
                      }`}
                    >
                      <span className="block text-docs-text-primary font-medium mb-0.5 transition-colors duration-200">{link.title}</span>
                      <span className="block text-xs text-docs-text-secondary transition-colors duration-200">{link.description}</span>
                    </Link>
                  ))}
                </div>
              </div>
            )}
            
            {/* Footer */}
            <div className="px-4 py-3 bg-docs-bg-sidebar border-t border-docs-border-main flex items-center justify-between text-xs text-docs-text-muted transition-colors duration-200">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1"><kbd className="bg-docs-active-link-bg px-1.5 rounded transition-colors duration-200">↑</kbd><kbd className="bg-docs-active-link-bg px-1.5 rounded transition-colors duration-200">↓</kbd> to navigate</span>
                <span className="flex items-center gap-1"><kbd className="bg-docs-active-link-bg px-1.5 rounded transition-colors duration-200">↵</kbd> to select</span>
              </div>
              <span className="flex items-center gap-1"><kbd className="bg-docs-active-link-bg px-1.5 rounded transition-colors duration-200">esc</kbd> to close</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
