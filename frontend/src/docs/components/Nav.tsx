"use client";

import DocsSearch from "./DocsSearch";
import { Link } from "react-router-dom";
import { useLocation, useNavigate } from "react-router-dom";
import { useState, useEffect, useRef } from "react";

/**
 * Nav — top navigation bar.
 */
interface NavProps {
  showDocsSearch?: boolean;
}

export default function Nav({ showDocsSearch = false }: NavProps) {
  const pathname = useLocation().pathname;
  const [isOpen, setIsOpen] = useState(false);
  const navRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside of the navbar
  useEffect(() => {
    if (!isOpen) return;

    const handleOutsideClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target && navRef.current && !navRef.current.contains(target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("click", handleOutsideClick);
    return () => document.removeEventListener("click", handleOutsideClick);
  }, [isOpen]);

  // Lock scroll when mobile menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  // Close mobile menu on screen resize to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setIsOpen(false);
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <nav ref={navRef}>
      <Link to="/" className="nav-logo" onClick={() => setIsOpen(false)}>
        <span className="font-extrabold text-2xl tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-rose-500 via-purple-500 to-blue-500 dark:from-rose-400 dark:via-purple-400 dark:to-blue-400">
          Karl
        </span>
        {showDocsSearch && (
          <span className="text-black dark:text-white font-bold text-2xl hidden sm:inline-block">Docs</span>
        )}
      </Link>

      <ul className="nav-links">
        <li>
          <Link to="/" className={pathname === "/" ? "active" : ""}>
            Home
          </Link>
        </li>
        <li className="nav-dropdown-wrapper">
          <Link to="/docs" className={`nav-dropdown-trigger ${pathname?.startsWith("/docs") ? "active" : ""}`}>
            Resources
          </Link>
          <div className="nav-dropdown-menu">
            <Link to="/docs/whitepaper" className={pathname === "/docs/whitepaper" ? "active" : ""}>
              About
            </Link>
            <Link to="/docs/security-document" className={pathname === "/docs/security-document" ? "active" : ""}>
              Security
            </Link>
            <Link to="/docs" className={pathname === "/docs" ? "active" : ""}>
              Documentation
            </Link>
          </div>
        </li>
      </ul>

      <div className="nav-right">
        {showDocsSearch && <DocsSearch />}

        <Link to="https://kanerika.com/contact-us/" 
          target="_blank" 
          rel="noopener noreferrer" 
          className="nav-cta-desktop flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-[#8C48FF] to-[#FF5F3D] rounded-md hover:opacity-90 transition-opacity"
        >
          <span>Contact Us</span>
          <svg aria-hidden="true" className="w-4 h-4" fill="currentColor" viewBox="0 0 448 512" xmlns="http://www.w3.org/2000/svg">
            <path d="M190.5 66.9l22.2-22.2c9.4-9.4 24.6-9.4 33.9 0L441 239c9.4 9.4 9.4 24.6 0 33.9L246.6 467.3c-9.4 9.4-24.6 9.4-33.9 0l-22.2-22.2c-9.5-9.5-9.3-25 .4-34.3L311.4 296H24c-13.3 0-24-10.7-24-24v-32c0-13.3 10.7-24 24-24h287.4L190.9 101.2c-9.8-9.3-10-24.8-.4-34.3z" />
          </svg>
        </Link>
        <button
          className="hamburger-button"
          onClick={() => setIsOpen(!isOpen)}
          aria-label="Toggle menu"
          aria-expanded={isOpen}
        >
          <span className={`hamburger-bar ${isOpen ? "open" : ""}`} />
          <span className={`hamburger-bar ${isOpen ? "open" : ""}`} />
          <span className={`hamburger-bar ${isOpen ? "open" : ""}`} />
        </button>
      </div>

      <div className={`mobile-menu ${isOpen ? "open" : ""}`}>
        <ul className="mobile-nav-links">
          <li>
            <Link to="/" className={pathname === "/" ? "active" : ""} onClick={() => setIsOpen(false)}>
              Home
            </Link>
          </li>
          <li className="mobile-dropdown-wrapper">
            <span className="mobile-dropdown-title">Resources</span>
            <ul className="mobile-submenu">
              <li>
                <Link to="/docs/whitepaper" className={pathname === "/docs/whitepaper" ? "active" : ""} onClick={() => setIsOpen(false)}>
                  About
                </Link>
              </li>
              <li>
                <Link to="/docs/security-document" className={pathname === "/docs/security-document" ? "active" : ""} onClick={() => setIsOpen(false)}>
                  Security
                </Link>
              </li>
              <li>
                <Link to="/docs" className={pathname === "/docs" ? "active" : ""} onClick={() => setIsOpen(false)}>
                  Documentation
                </Link>
              </li>
            </ul>
          </li>
        </ul>

        <div className="mobile-nav-cta">
          <Link to="https://kanerika.com/contact-us/" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="flex items-center justify-center gap-2 px-4 py-3 text-sm font-semibold text-white bg-gradient-to-r from-[#8C48FF] to-[#FF5F3D] rounded-md hover:opacity-90 transition-opacity w-full text-center"
            onClick={() => setIsOpen(false)}
          >
            <span>Contact Us</span>
            <svg aria-hidden="true" className="w-4 h-4" fill="currentColor" viewBox="0 0 448 512" xmlns="http://www.w3.org/2000/svg">
              <path d="M190.5 66.9l22.2-22.2c9.4-9.4 24.6-9.4 33.9 0L441 239c9.4 9.4 9.4 24.6 0 33.9L246.6 467.3c-9.4 9.4-24.6 9.4-33.9 0l-22.2-22.2c-9.5-9.5-9.3-25 .4-34.3L311.4 296H24c-13.3 0-24-10.7-24-24v-32c0-13.3 10.7-24 24-24h287.4L190.9 101.2c-9.8-9.3-10-24.8-.4-34.3z" />
            </svg>
          </Link>
        </div>
      </div>
    </nav>
  );
}

