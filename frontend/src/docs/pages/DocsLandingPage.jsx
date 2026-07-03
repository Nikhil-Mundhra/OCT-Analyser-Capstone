import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { HomeNav } from '../../HomePage.jsx';
import Footer from '../components/Footer';
import DocsSidebar from '../components/DocsSidebar';
import DocsCardGrid from '../components/DocsCardGrid';
import DocsTableOfContents from '../components/DocsTableOfContents';
import { docCategories } from '../manifest';

const queryDocToSlugMap = {
  readme: 'readme',
  implementation: 'implementation-info',
};

export default function DocsLandingPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState("getting-started");

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const queryDoc = params.get('doc')?.trim().toLowerCase();

    if (queryDoc) {
      const targetSlug = queryDocToSlugMap[queryDoc] ?? queryDoc;
      const exists = docCategories.some((category) =>
        category.articles.some((article) => article.slug === targetSlug)
      );

      if (exists) {
        navigate(`/docs/${targetSlug}`, { replace: true });
        return;
      }
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: "-90px 0px -60% 0px" }
    );

    docCategories.forEach((category) => {
      const el = document.getElementById(category.id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [location.search, navigate]);

  return (
    <div className="docs-theme flex flex-col min-h-screen bg-docs-bg-page text-docs-text-primary selection:bg-blue-500/30 font-sans transition-colors duration-200">
      <HomeNav />

      <div className="flex flex-1 w-full relative">
        <DocsSidebar />

        <main className="flex-1 min-w-0 px-6 py-12 md:px-12 lg:px-16 xl:px-24">
          <div className="max-w-4xl mx-auto">
            <div className="relative mb-20 mt-8 overflow-hidden">
              <div className="absolute -top-10 -left-10 w-40 h-40 bg-docs-glow-bg blur-[100px] rounded-full pointer-events-none"></div>
              <h1 className="text-5xl md:text-6xl font-light text-slate-900 dark:text-white mb-6 tracking-tight transition-colors duration-200">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-500 via-purple-500 to-blue-500 dark:from-rose-400 dark:via-purple-400 dark:to-blue-400">
                  OCT Analyser
                </span>{" "}
                Documentation
              </h1>
              <p className="text-lg md:text-xl text-slate-600 dark:text-slate-400 leading-relaxed max-w-2xl transition-colors duration-200">
                Everything you need to set up, operate, and extend the OCT/OCTA Clinical Inference Interface.
              </p>
            </div>

            {docCategories.map((category) => (
              <DocsCardGrid key={category.id} category={category} />
            ))}
          </div>
        </main>

        <DocsTableOfContents activeSection={activeSection} />
      </div>

      <Footer />
    </div>
  );
}
