import { useSearchParams } from "react-router";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { TabPanel, Tabs } from "../components/ui/Tabs";
import { M2 } from "./lab/M2";
import { M3 } from "./lab/M3";
import { M4 } from "./lab/M4";
import { M5 } from "./lab/M5";
import { M6 } from "./lab/M6";

const MODULES = [
  { key: "m2", label: "M2 · Linear", hint: "Linear & logistic regression, preprocessing" },
  { key: "m3", label: "M3 · Trees", hint: "Decision trees, pruning, ensembles, importance" },
  { key: "m4", label: "M4 · Unsupervised", hint: "Clustering, PCA, t-SNE/UMAP, anomaly detection" },
  { key: "m5", label: "M5 · Evaluation", hint: "Validation, metrics, calibration, search, tests" },
  { key: "m6", label: "M6 · Production", hint: "Skew, point-in-time, packaging, drift" },
] as const;
type ModuleKey = (typeof MODULES)[number]["key"];

export default function Lab() {
  const [params, setParams] = useSearchParams();
  const raw = params.get("module");
  const module: ModuleKey = MODULES.some((m) => m.key === raw) ? (raw as ModuleKey) : "m2";
  const exp = params.get("exp") ?? undefined;
  const setModule = (m: ModuleKey) => setParams({ module: m });
  const setExp = (e: string) => setParams({ module, exp: e });
  const View = { m2: M2, m3: M3, m4: M4, m5: M5, m6: M6 }[module];

  return (
    <>
      <PageHeader
        crumb="ML Lab"
        title="The machine-learning laboratory"
        description="Run the M2–M6 techniques on the credit-card transactions. Experiments fit on the training split and score on the validation split with PR-AUC; the test split stays reserved for the production model."
      />
      <Section>
        <Tabs idPrefix="lab-module" ariaLabel="Course module" items={MODULES.map((m) => ({ key: m.key, label: m.label, hint: m.hint }))} value={module} onChange={setModule} className="mb-8 w-full sm:w-fit" />
        <TabPanel idPrefix="lab-module" value={module}>
          <View exp={exp} setExp={setExp} />
        </TabPanel>
      </Section>
    </>
  );
}

export interface ModuleProps {
  exp?: string;
  setExp: (e: string) => void;
}
