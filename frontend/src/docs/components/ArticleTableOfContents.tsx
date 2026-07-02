"use client";

import { useEffect, useState } from "react";

export interface Heading {
  id: string;
  text: string;
  level: number;
}

interface ArticleTableOfContentsProps {
  headings: Heading[];
}

export default function ArticleTableOfContents({
  headings,
}: ArticleTableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>("");
  const [indicatorStyle, setIndicatorStyle] = useState<React.CSSProperties>({
    opacity: 0,
    top: 0,
    height: 0,
  });

  useEffect(() => {
    if (!headings || headings.length === 0) return;

    const handleScroll = () => {
      let active = "";

      // Bottom detection
      const isAtBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 10;
      if (isAtBottom) {
        setActiveId(headings[headings.length - 1].id);
        return;
      }

      // Check headings scroll positions relative to viewport
      // We trigger active when heading is near 130px from top (below the 120px scroll offset)
      for (let i = 0; i < headings.length; i++) {
        const el = document.getElementById(headings[i].id);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= 130) {
            active = headings[i].id;
          }
        }
      }

      if (!active && headings.length > 0) {
        active = headings[0].id;
      }
      setActiveId(active);
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll);
    
    const interval = setInterval(handleScroll, 1000);

    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
      clearInterval(interval);
    };
  }, [headings]);

  useEffect(() => {
    if (!activeId) {
      setIndicatorStyle({ opacity: 0 });
      return;
    }

    const activeEl = document.getElementById(`toc-link-${activeId}`);
    if (activeEl) {
      setIndicatorStyle({
        opacity: 1,
        top: `${activeEl.offsetTop}px`,
        height: `${activeEl.offsetHeight}px`,
      });
      activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeId]);

  if (!headings || headings.length === 0) {
    return null;
  }

  return (
    <aside className="hidden xl:flex flex-col shrink-0 sticky top-[68px] h-[calc(100vh-120px)] w-[280px] bg-docs-bg-page overflow-y-auto scrollbar-hide transition-colors duration-200">
      <div className="sticky top-0 z-50 bg-docs-bg-page px-6 pt-10 pb-4 transition-colors duration-200">
        <h4 className="text-xs font-semibold text-docs-text-primary uppercase tracking-widest transition-colors duration-200">
          On this page
        </h4>
      </div>

      <div className="relative px-6 pt-4 pb-8 flex flex-col gap-1 text-[13px] leading-snug text-docs-text-muted mt-4 transition-colors duration-200">
        {/* Left rule line — spans the link area */}
        <div className="absolute left-6 top-4 bottom-8 w-[1px] bg-docs-border-main transition-colors duration-200"></div>

        {/* Sliding active indicator */}
        <div
          className="absolute left-6 w-[2px] bg-blue-600 dark:bg-blue-500 shadow-[0_0_8px_rgba(37,99,235,0.5)] dark:shadow-[0_0_8px_rgba(59,130,246,0.5)] rounded-full transition-all duration-300 ease-out -translate-x-[0.5px]"
          style={indicatorStyle}
        />

        {headings.map((heading) => {
          const isActive = activeId === heading.id;

          return (
            <a
              key={heading.id}
              id={`toc-link-${heading.id}`}
              href={`#${heading.id}`}
              className={`relative flex items-center py-1.5 transition-colors hover:text-docs-text-primary ${
                heading.level === 3 ? "pl-7 text-[12px]" : "pl-4"
              } ${
                isActive ? "text-blue-600 dark:text-blue-400 font-medium" : ""
              }`}
            >
              <span className="line-clamp-2">{heading.text}</span>
            </a>
          );
        })}
      </div>
    </aside>
  );
}
