/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  corePlugins: {
    preflight: false,
  },
  content: [
    "./src/docs/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "cat-blue": "#3b82f6",
        "cat-purple": "#a855f7",
        "cat-emerald": "#10b981",
        "docs-bg": {
          light: "#ffffff",
          dark: "#050505",
        },
        "docs-sidebar": {
          light: "#fafaf9",
          dark: "#050505",
        },
        "docs-card": {
          light: "#ffffff",
          dark: "#0a0a0a",
        },
      },
    },
  },
  plugins: [],
};
