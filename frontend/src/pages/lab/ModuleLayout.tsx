import type { ComponentType, ReactNode } from "react";
import { TabPanel, Tabs } from "../../components/ui/Tabs";
import type { ModuleProps } from "../Lab";

export interface ExperimentDef {
  key: string;
  label: string;
  Component: ComponentType;
}

/** Second-level navigation inside a module; the selected experiment lives in the URL (?exp=). */
export function ModuleLayout({ id, intro, experiments, exp, setExp }: ModuleProps & { id: string; intro: ReactNode; experiments: ExperimentDef[] }) {
  const current = experiments.find((e) => e.key === exp) ?? experiments[0];
  const C = current.Component;
  return (
    <div>
      <div className="mb-6 max-w-3xl text-sm leading-relaxed text-ink-2">{intro}</div>
      <Tabs idPrefix={`${id}-exp`} ariaLabel="Experiment" size="sm" items={experiments.map((e) => ({ key: e.key, label: e.label }))} value={current.key} onChange={setExp} className="mb-6 w-full sm:w-fit" />
      <TabPanel idPrefix={`${id}-exp`} value={current.key}>
        <C key={current.key} />
      </TabPanel>
    </div>
  );
}
