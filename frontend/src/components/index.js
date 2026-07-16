/**
 * Barrel export — re-exports every public symbol from the components/ tree.
 * External consumers (route page.jsx files, layouts) import from here.
 *
 * Internal imports should use the specific sub-module path for clarity.
 */

// Constants
export { screens, MIN_RATIONALE_LENGTH } from "./constants";

// UI primitives
export { StatusBadge } from "./ui/StatusBadge";
export { Spinner }     from "./ui/Spinner";
export { Card }        from "./ui/Card";
export { WireBox }     from "./ui/WireBox";
export { Metric }      from "./ui/Metric";

// Layout components
export { HomeNav }      from "./layout/HomeNav";
export { Header }       from "./layout/Header";
export { Sidebar }      from "./layout/Sidebar";
export { CaseSwitcher } from "./layout/CaseSwitcher";

// Screen components
export { DashboardScreen } from "./screens/DashboardScreen";
export { WorklistScreen }  from "./screens/WorklistScreen";
export { UploadScreen }    from "./screens/UploadScreen";
export { ReviewScreen }    from "./screens/ReviewScreen";
export { DecisionScreen }  from "./screens/DecisionScreen";
export { OutcomesScreen }  from "./screens/OutcomesScreen";

// Utils (exported for consumers that need them directly)
export { riskFromScan }  from "./utils/riskUtils";
export { getClassColor } from "./utils/colorUtils";
