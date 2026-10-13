import { lazy } from "react";
import { createBrowserRouter, RouterProvider, type RouteObject } from "react-router";
import { Layout } from "./components/layout/Layout";
import { ToastProvider } from "./components/ui/Toast";
import { HealthProvider } from "./hooks/HealthContext";
import { RouteError } from "./pages/RouteError";

const Home = lazy(() => import("./pages/Home"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Predict = lazy(() => import("./pages/Predict"));
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
      { path: "dashboard", element: <Dashboard /> },
      { path: "predict", element: <Predict /> },
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
