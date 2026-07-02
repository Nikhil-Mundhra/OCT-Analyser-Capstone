"use client";

import { Link } from "react-router-dom";
import { useLocation, useNavigate } from "react-router-dom";
import {
  categoryColorStyles,
  docCategories,
} from "../manifest";

export default function DocsSidebar() {
  const pathname = useLocation().pathname;

  return (
    <aside className="hidden md:flex flex-col sticky top-[68px] h-[calc(100vh-68px)] w-[280px] shrink-0 border-r border-docs-border-main bg-docs-bg-sidebar overflow-y-auto scrollbar-hide transition-colors duration-200">
      {/* Scrollable nav content */}
      <div className="relative px-6 pt-8 pb-6 flex flex-col gap-8">
        {docCategories.map((category) => {
          const styles = categoryColorStyles[category.color];

          return (
            <div key={category.id}>
              <h4 className="text-xs font-semibold text-docs-text-primary uppercase tracking-widest mb-3 flex items-center gap-2 transition-colors duration-200">
                <div className={`w-2 h-2 rounded-full shrink-0 ${styles.dot}`}></div>
                {category.sidebarTitle}
              </h4>
              <ul className="flex flex-col gap-1.5 border-l border-docs-border-main ml-1 pl-4 text-sm transition-colors duration-200">
                {category.articles.map((article) => {
                  const href = `/docs/${article.slug}`;
                  const isActive = pathname === href;

                  return (
                    <li key={article.slug}>
                      <Link to={href}
                        className={`block py-1 transition-colors duration-200 ${
                          isActive
                            ? `${styles.tocActive} font-medium`
                            : `text-docs-text-muted hover:text-docs-text-primary`
                        }`}
                      >
                        {article.sidebarTitle}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
