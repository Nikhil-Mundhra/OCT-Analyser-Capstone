import React from 'react';
import { AppProvider } from '../AppContext';
import '../../docs-input.css';

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <title>OCT Analyser Capstone</title>
      </head>
      <body>
        <AppProvider>
          {children}
        </AppProvider>
      </body>
    </html>
  );
}
