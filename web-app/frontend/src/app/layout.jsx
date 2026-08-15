import React from 'react';
import { AppProvider } from '../AppContext';
import '../../docs-input.css';

export const metadata = {
  title: 'OCT Analyser Capstone',
  description: 'End-to-end clinician workflow for OCT/OCTA triage, explainable scan review, and human-in-the-loop decision support.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>
        <AppProvider>
          {children}
        </AppProvider>
      </body>
    </html>
  );
}
