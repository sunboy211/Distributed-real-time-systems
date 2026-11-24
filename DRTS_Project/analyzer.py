# analyzer.py
import os
import math
import pandas as pd
import argparse
import logging
from dataclasses import dataclass, field
from scheduling_formulas import (
    compute_busy_window,
    demand_bound_edf,
    supply_bound_bdr,
    rta_rm_bdr,
    half_half_interface
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

@dataclass
class AnalysisConfig:
    input_dir: str = 'data/3'
    output_dir: str = 'results/3'
    epsilon: float = 1e-9
    def setup(self):
        os.makedirs(self.output_dir, exist_ok=True)


def load_data(directory):
    tasks = pd.read_csv(os.path.join(directory, 'tasks.csv'))
    budgets = pd.read_csv(os.path.join(directory, 'budgets.csv'))
    architecture = pd.read_csv(os.path.join(directory, 'architecture.csv'))
    return tasks, budgets, architecture


def build_components_and_tasks(tasks_df, budgets_df, arch_df):
    # map budgets and architecture to dicts
    budget_map = budgets_df.set_index('component_id').to_dict(orient='index')
    arch_map = arch_df.set_index('core_id').to_dict(orient='index')

    components = {}
    task_list = []

    for comp_id, info in budget_map.items():
        components[comp_id] = {
            'algorithm': info['scheduler'].upper(),
            'core': info['core_id'],
            'Q': float(info['budget']),
            'P': float(info['period']),
            'priority': float(info['priority']) if pd.notna(info['priority']) else None
        }

    for _, row in tasks_df.iterrows():
        comp_info = budget_map[row['component_id']]
        arch_info = arch_map[comp_info['core_id']]
        wcet = float('inf') if arch_info['speed_factor'] <= 0 else float(row['wcet']) / arch_info['speed_factor']
        period = float(row['period'])
        prio_val = (
            float(row['priority']) if pd.notna(row['priority'])
            else 1.0 / period if period > 0 else 0.0
        )
        task_list.append({
            'name': row['task_name'],
            'component': row['component_id'],
            'C': wcet,
            'T': period,
            'prio': prio_val
        })

    return components, task_list


def analyze_component(component_id, comp_cfg, tasks, epsilon):
    logging.info(f"Analyzing component {component_id} on core {comp_cfg['core']} using {comp_cfg['algorithm']}")
    # extract tasks for this component
    params = [(t['C'], t['T']) for t in tasks if t['component'] == component_id]
    full_tasks = [t for t in tasks if t['component'] == component_id]

    if not params:
        return True, (0, 1)  # no tasks -> schedulable, dummy supply

    alpha = comp_cfg['Q'] / comp_cfg['P'] if comp_cfg['P'] > 0 else 0.0
    delta = 2 * (comp_cfg['P'] - comp_cfg['Q']) if comp_cfg['P'] >= comp_cfg['Q'] else 0.0
    logging.info(f"  Interface: bandwidth={alpha:.4f}, latency={delta:.4f}")

    sched_ok = True
    if comp_cfg['algorithm'] == 'EDF':
        upper = compute_busy_window(params)
        points = set(k * T for C, T in params if T > 0 for k in range(1, int(upper / T) + 2)) | {upper}
        for t in sorted(p for p in points if 0 <= p <= upper):
            if demand_bound_edf(params, t) > supply_bound_bdr(alpha, delta, t) + epsilon:
                logging.warning(f"    Unscheduled at t={t}")
                sched_ok = False
                break
    else:
        results = rta_rm_bdr(full_tasks, alpha, delta, epsilon)
        for name, (ok, R) in results.items():
            Ti = next(t['T'] for t in full_tasks if t['name'] == name)
            if not ok:
                logging.warning(f"    {name} misses: R={R:.3f} > T={Ti:.3f}")
                sched_ok = False
            else:
                logging.info(f"    {name}: R={R:.3f} ≤ T={Ti:.3f}")

    # half-half supply task
    try:
        Cs, Ts = half_half_interface(alpha, delta)
    except ValueError:
        Cs, Ts = 0, 1
    return sched_ok, (Cs, Ts)


def analyze_core(core_id, algorithm, supply_tasks, comp_status, epsilon):
    # collect supply tasks for this core
    sup_list = [supply_tasks[c] for c, ok in comp_status.items() if ok and components[c]['core'] == core_id]
    U = sum(C / T for C, T in sup_list if T > 0)
    if algorithm == 'EDF':
        bound = 1.0
    else:
        n = len(sup_list)
        bound = n * (2 ** (1 / n) - 1) if n else 1.0
    ok = U <= bound + epsilon

    return ok


def run_analysis(config: AnalysisConfig):
    config.setup()
    logging.info('=== Hierarchical Scheduling Analysis ===')
    # Load inputs
    try:
        tasks_df, budgets_df, arch_df = load_data(config.input_dir)
    except FileNotFoundError as e:
        logging.error(f'Input error: {e}')
        return

    global components
    components, task_list = build_components_and_tasks(tasks_df, budgets_df, arch_df)

    comp_status = {}
    comp_supply = {}
    # Component-level analysis
    for comp_id, cfg in components.items():
        sched_ok, supply = analyze_component(comp_id, cfg, task_list, config.epsilon)
        comp_status[comp_id] = sched_ok
        comp_supply[comp_id] = supply
        # Show per-component summary
        logging.info('-' * 40)
        logging.info(f'Component {comp_id}: sched={cfg["algorithm"]} on core {cfg["core"]}')
        status_str = 'PASS' if sched_ok else 'FAIL'
        logging.info(f'  Overall: {status_str}')

    # Core-level analysis
    core_results = {}
    for core_id, row in arch_df.set_index('core_id').iterrows():
        core_res = analyze_core(core_id, row['scheduler'], comp_supply, comp_status, config.epsilon)
        core_results[core_id] = core_res
        # Show per-core summary
        result_str = 'OK' if core_res else 'NOT OK'
        logging.info(f'Core {core_id} result: {result_str}')

    # Generate solution.csv with detailed per-task entries
    sol = []
    for t in task_list:
        comp_id = t['component']
        core_ok = core_results.get(components[comp_id]['core'], False)
        sol.append({
            'task': t['name'],
            'component': comp_id,
            'component_schedulable': comp_status[comp_id],
            'core_schedulable': core_ok,
            'task_schedulable': comp_status[comp_id] and core_ok
        })
    output_path = os.path.join(config.output_dir, 'solution_analyzer.csv')
    pd.DataFrame(sol).to_csv(output_path, index=False)
    logging.info(f'Solution written to {output_path}')

    # Summary
    overall = all(core_results.values()) and all(comp_status.values())
    logging.info('=' * 40)
    logging.info(f'Overall system schedulable: {overall}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/3')
    parser.add_argument('--output_dir', default='results/3')
    parser.add_argument('--epsilon', type=float, default=1e-9)
    args = parser.parse_args()
    cfg = AnalysisConfig(args.input_dir, args.output_dir, args.epsilon)
    run_analysis(cfg)