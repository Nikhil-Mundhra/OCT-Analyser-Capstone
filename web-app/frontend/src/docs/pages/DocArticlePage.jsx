"use client";
import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';

import { marked } from 'marked';
import DOMPurify from 'dompurify';
import DocLayout from '../components/DocLayout';
import DocsTableOfContents from '../components/DocsTableOfContents';
import DocArticle from '../components/DocArticle';
import { getDocArticle } from '../manifest';
import { fetchDocContent, extractHeadingsFromMdx, getHeadingIdGenerator } from '../utils/content';

export default function DocArticlePage() {
  const params = useParams();
  const slug = params?.doc;
  const [content, setContent] = useState('');
  const [headings, setHeadings] = useState([]);
  const [loading, setLoading] = useState(true);

  const articleInfo = getDocArticle(slug);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const mdx = await fetchDocContent(slug);
      if (mdx) {
        setHeadings(extractHeadingsFromMdx(mdx));
        
        // Configure marked to add IDs to headings using the same logic
        const generateId = getHeadingIdGenerator();
        const renderer = new marked.Renderer();
        renderer.heading = function({text, depth, tokens}) {
          const content = this.parser.parseInline(tokens);
          if (depth === 2) {
            const id = generateId(text);
            return `<h2 id="${id}">${content}</h2>\n`;
          }
          return `<h${depth}>${content}</h${depth}>\n`;
        };
        marked.setOptions({ renderer, gfm: true });
        
        const html = DOMPurify.sanitize(marked.parse(mdx), {
          ADD_ATTR: ['target', 'class', 'style', 'loading', 'alt', 'width', 'height'],
          ADD_TAGS: ['iframe', 'span', 'div', 'img']
        });
        setContent(html);
      } else {
        setContent('<h1>Article not found</h1><p>The requested document could not be found.</p>');
        setHeadings([]);
      }
      setLoading(false);
    }
    if (slug) {
      load();
    }
  }, [slug]);

  if (!articleInfo && !loading) {
    return (
      <DocLayout>
        <div className="py-12">
          <h1 className="text-3xl font-light">Document Not Found</h1>
          <p className="mt-4 text-docs-text-secondary">We couldn't find the requested documentation page.</p>
        </div>
      </DocLayout>
    );
  }

  const toc = headings.length > 0 ? <DocsTableOfContents headings={headings} /> : null;

  return (
    <DocLayout toc={toc}>
      <DocArticle
        title={articleInfo?.title || 'Documentation'}
        description={articleInfo?.description || ''}
      >
        {articleInfo?.pdfHref && (
          <div className="mb-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-cat-blue/5 border border-cat-blue/20">
            <div className="flex items-start sm:items-center gap-3">
              <div className="p-2 rounded-lg bg-cat-blue/10 text-cat-blue shrink-0">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <h4 className="font-medium text-docs-text-primary text-sm">Executive Clinical PDF Report Available</h4>
                <p className="text-xs text-docs-text-secondary">Official publication-grade evaluation report compiled with full figures and tabular evidence.</p>
              </div>
            </div>
            <a
              href={articleInfo.pdfHref}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-cat-blue text-white text-xs font-semibold hover:opacity-90 transition-opacity shadow-sm whitespace-nowrap self-start sm:self-center"
            >
              Download Executive PDF
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        )}

        {articleInfo?.externalHref ? (
          <iframe 
            src={articleInfo.externalHref} 
            className="w-full h-[800px] border-0 rounded-lg shadow-sm bg-white mt-8"
            title={articleInfo.title}
          />
        ) : loading ? (
          <div className="py-8 animate-pulse">
            <div className="h-8 bg-docs-border-main rounded w-1/3 mb-6"></div>
            <div className="h-4 bg-docs-border-main rounded w-full mb-4"></div>
            <div className="h-4 bg-docs-border-main rounded w-5/6 mb-4"></div>
            <div className="h-4 bg-docs-border-main rounded w-4/6"></div>
          </div>
        ) : (
          <div dangerouslySetInnerHTML={{ __html: content }} />
        )}
      </DocArticle>
    </DocLayout>
  );
}
