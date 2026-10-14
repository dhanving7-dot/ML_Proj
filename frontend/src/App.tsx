import { lazy } from "react";
import { createBrowserRouter, Navigate, RouterProvider, type RouteObject } from "react-router";
import { Layout } from "./components/layout/Layout";
import { ToastProvider } from "./components/ui/Toast";
import { HealthProvider } from "./hooks/HealthContext";
import { RouteError } from "./pages/RouteError";

const Home = lazy(() => import("./pages/Home"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Analyze = lazy(() => import("./pages/Analyze"));
const Transactions = lazy(() => import("./pages/Transactions"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Lab = lazy(() => import("./pages/Lab"));
const Lifecycle = lazy(() => import("./pages/Lifecycle"));
const Monitoring = lazy(() => import("./pages/Monitoring"));
const Models = lazy(() => import("./pages/Models"));
const Data = lazy(() => import("./pages/Data"));
const NotFound = lazy(() => import("./pages/NotFound"));

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Home /> },
      { path: "overview", element: <Dashboard /> },
      { path: "analyze", element: <Analyze /> },
      { path: "transactions", element: <Transactions /> },
      { path: "evaluation", element: <Evaluation /> },
      // Paths of the previous PlacementPredict app keep working.
      { path: "dashboard", element: <Navigate to="/overview" replace /> },
      { path: "predict", element: <Navigate to="/analyze" replace /> },
      { path: "lab", element: <Lab /> },
      { path: "lifecycle", element: <Lifecycle /> },
      { path: "monitoring", element: <Monitoring /> },
      { path: "models", element: <Models /> },
      { path: "data", element: <Data /> },
      { path: "*", element: <NotFound /> },
    ],
  },
];

const router = createBrowserRouter(routes);

export default function App() {
  return (
    <HealthProvider>
      <ToastProvider>
        <RouterProvider router={router} />
      </ToastProvider>
    </HealthProvider>
  );
}
