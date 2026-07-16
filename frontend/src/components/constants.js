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
  { id: "dashboard", group: "overview",  path: "/dashboard",   label: "Dashboard",           icon: LayoutDashboard },
  { id: "worklist",  group: "clinical",  path: "/worklist",    label: "Triage Worklist",      icon: ListTodo },
  { id: "upload",    group: "clinical",  path: "/QC",          label: "Upload and QC",        icon: UploadCloud },
  { id: "review",    group: "clinical",  path: "/review",      label: "Scan Review",          icon: Eye },
  { id: "decision",  group: "clinical",  path: "/human-check", label: "Human Decision Gate",  icon: UserCheck },
  { id: "outcomes",  group: "system",    path: "/outcomes",    label: "Outcomes and Safety",  icon: Activity },
];
