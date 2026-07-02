"use client";

import { useEffect, useState } from "react";
import {
  categoryColorStyles,
  docCategories,
} from "../manifest";

interface DocsTableOfContentsProps {
  activeSection?: string;
}

export default function DocsTableOfContents({
  activeSection: propActiveSection,
}: DocsTableOfContentsProps) {
  const [activeSection, setActiveSection] = useState("getting-started");
  const [indicatorStyle, setIndicatorStyle] = useState<React.CSSProperties>({
    opacity: 0,
    top: 0,
    height: 0,
  });

  useEffect(() => {
    const handleScroll = () => {
      let active = "";

      // Bottom detection
      const isAtBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 10;
      if (isAtBottom && docCategories.length > 0) {
        setActiveSection(docCategories[docCategories.length - 1].id);
        return;
      }

      // Check category scroll positions relative to viewport
      // Trigger active when category top is near 130px from top (below the 120px scroll offset)
      for (let i = 0; i < docCategories.length; i++) {
        const el = document.getElementById(docCategories[i].id);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= 130) {
            active = docCategories[i].id;
          }
        }
      }

      if (!active && docCategories.length > 0) {
        active = docCategories[0].id;
      }
      setActiveSection(active);
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
  }, []);

  useEffect(() => {
    const currentActive = propActiveSection || activeSection;
    if (!currentActive) {
      setIndicatorStyle({ opacity: 0 });
      return;
    }

    const activeEl = document.getElementById(`toc-link-${currentActive}`);
    if (activeEl) {
      setIndicatorStyle({
        opacity: 1,
        top: `${activeEl.offsetTop}px`,
        height: `${activeEl.offsetHeight}px`,
      });
      activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [propActiveSection, activeSection]);

  return (
    <aside className="hidden xl:flex flex-col shrink-0 sticky top-[68px] h-[calc(100vh-120px)] w-[240px] bg-docs-bg-page overflow-y-auto scrollbar-hide transition-colors duration-200">
      <div className="sticky top-0 z-50 bg-docs-bg-page px-6 pt-10 pb-4 transition-colors duration-200">
        <h4 className="text-xs font-semibold text-docs-text-primary uppercase tracking-widest transition-colors duration-200">
          On this page
        </h4>
      </div>

      <div className="relative px-6 pt-4 pb-8 flex flex-col gap-1 text-sm text-docs-text-muted mt-4 transition-colors duration-200">
        {/* Left rule line — spans the link area */}
        <div className="absolute left-6 top-4 bottom-8 w-[1px] bg-docs-border-main transition-colors duration-200"></div>

        {/* Sliding active indicator */}
        {docCategories.map((category) => {
          const isActive = (propActiveSection || activeSection) === category.id;
          if (!isActive) return null;
          const styles = categoryColorStyles[category.color];
          
          return (
            <div
              key={category.id}
              className={`absolute left-6 w-[2px] rounded-full transition-all duration-300 ease-out -translate-x-[0.5px] ${styles.tocShadow}`}
              style={indicatorStyle}
            />
          );
        })}

        {docCategories.map((category) => {
          const styles = categoryColorStyles[category.color];
          const isActive = (propActiveSection || activeSection) === category.id;

          return (
            <a
              key={category.id}
              id={`toc-link-${category.id}`}
              href={`#${category.id}`}
              className={`relative flex items-center pl-4 py-1.5 transition-colors hover:text-docs-text-primary ${
                isActive ? `${styles.tocActive} font-medium` : ""
              }`}
            >
              {category.sidebarTitle}
            </a>
          );
        })}
      </div>
    </aside>
  );
}
