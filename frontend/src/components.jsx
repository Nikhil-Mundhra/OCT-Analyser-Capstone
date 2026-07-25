/**
 * Backward-compatibility shim.
 *
 * The original monolithic components.jsx has been split into components/
 * (ui/, layout/, screens/, utils/, constants.js).
 *
 * This file re-exports everything so that any existing import path
 * `from "./components"` or `from "../../components"` continues to work
 * without change during the migration period.
 *
 * New code should import directly from the sub-module, e.g.:
 *   import { StatusBadge } from "./components/ui/StatusBadge";
 */
export {
  // Constants
  screens,
  MIN_RATIONALE_LENGTH,
  ROUTES,

  // UI
  StatusBadge,
  Spinner,
  Card,
  WireBox,
  Metric,

  // Layout
  HomeNav,
  Header,
  Sidebar,
  CaseSwitcher,

  // Screens
  DashboardScreen,
  WorklistScreen,
  UploadScreen,
  ReviewScreen,
  DecisionScreen,
  OutcomesScreen,

  // Utils
  riskFromScan,
  getClassColor,
} from "./components/index.js";
