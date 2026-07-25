import {
  Activity,
  Eye,
  LayoutDashboard,
  ListTodo,
  UploadCloud,
  UserCheck,
} from "lucide-react";

/** Minimum character count for a clinician rationale to be considered valid. */
export const MIN_RATIONALE_LENGTH = 12;

/** Standardized Application Route Constants */
export const ROUTES = {
  HOME: "/",
  DASHBOARD: "/dashboard",
  WORKLIST: "/worklist",
  QC: "/qc",
  REVIEW: "/review",
  DECISION: "/human-check",
  OUTCOMES: "/outcomes",
};

/**
 * Master navigation / screen registry.
 * Each entry has:
 *   id     — stable string key
 *   group  — "overview" | "clinical" | "system"
 *   path   — Next.js App Router route
 *   label  — human-readable label shown in Sidebar
 *   icon   — Lucide React icon component
 */
export const screens = [
  { id: "dashboard", group: "overview",  path: ROUTES.DASHBOARD,   label: "Dashboard",           icon: LayoutDashboard },
  { id: "worklist",  group: "clinical",  path: ROUTES.WORKLIST,    label: "Triage Worklist",      icon: ListTodo },
  { id: "upload",    group: "clinical",  path: ROUTES.QC,          label: "Upload and QC",        icon: UploadCloud },
  { id: "review",    group: "clinical",  path: ROUTES.REVIEW,      label: "Scan Review",          icon: Eye },
  { id: "decision",  group: "clinical",  path: ROUTES.DECISION,    label: "Human Decision Gate",  icon: UserCheck },
  { id: "outcomes",  group: "system",    path: ROUTES.OUTCOMES,    label: "Outcomes and Safety",  icon: Activity },
];
