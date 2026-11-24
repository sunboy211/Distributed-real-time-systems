# simulator.py
import os
import csv
import argparse
import heapq
from typing import List, Dict, Tuple, Optional
from scheduling_formulas import lcm

# Precision constant for float comparisons
EPSILON = 1e-9


class SchedulingComponent:
    """Represents a component in the hierarchical scheduling system"""

    def __init__(self, identifier: str, scheduling_algorithm: str, allocation_amount: float,
                 allocation_period: int, host_processor: str):
        self.identifier = identifier
        self.scheduling_algorithm = scheduling_algorithm.upper()
        self.allocation_amount = allocation_amount
        self.allocation_period = allocation_period
        self.host_processor = host_processor

        self.available_budget = 0.0
        self.importance = float('inf')  # For RM scheduling (lower is more important)

        self.contained_tasks = []


class TaskInstance:
    """Represents a task in the hierarchical scheduling system"""

    def __init__(self, identifier: str, execution_time: float, time_period: int, parent_component: str):
        self.identifier = identifier
        self.base_execution = execution_time * 1.01
        self.time_period = time_period
        self.parent_component = parent_component

        self.activation_time = 0.0
        self.due_time = float('inf')
        self.execution_left = 0.0
        self.importance = float('inf')
        self.completion_times = []


class Processor:
    """Represents a processing core in the multicore platform"""

    def __init__(self, identifier: str, scheduling_algorithm: str, performance_factor: float):
        self.identifier = identifier
        self.scheduling_algorithm = scheduling_algorithm.upper()
        self.performance_factor = performance_factor if performance_factor > EPSILON else EPSILON

        self.assigned_components = []

        self.active_duration = 0.0

        self.virtual_deadlines = {}


class HierarchicalSchedulingSimulator:
    """Event-driven simulator for hierarchical scheduling systems"""

    def __init__(self):
        self.processors = {}
        self.components = {}
        self.tasks = []

        self._event_queue = []

    def reset(self):
        """Clear all system state data"""
        self.processors.clear()
        self.components.clear()
        self.tasks.clear()

    def import_processor_configuration(self, config_file: str):
        """Load processor definitions from file"""
        with open(config_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                proc_id = row['core_id']
                self.processors[proc_id] = Processor(
                    identifier=proc_id,
                    scheduling_algorithm=row['scheduler'],
                    performance_factor=float(row['speed_factor'])
                )

    def import_component_configuration(self, config_file: str):
        """Load component definitions and assignments from file"""
        with open(config_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                comp_id = row['component_id']
                proc_id = row['core_id']

                if proc_id not in self.processors:
                    continue

                component = SchedulingComponent(
                    identifier=comp_id,
                    scheduling_algorithm=row['scheduler'],
                    allocation_amount=float(row['budget']),
                    allocation_period=int(row['period']),
                    host_processor=proc_id
                )

                self.components[comp_id] = component
                self.processors[proc_id].assigned_components.append(component)

        for proc in self.processors.values():
            if proc.scheduling_algorithm == 'RM':
                proc.assigned_components.sort(
                    key=lambda c: c.allocation_period if c.allocation_period > 0 else float('inf')
                )

                for comp in proc.assigned_components:
                    comp.importance = (comp.allocation_period if comp.allocation_period > 0
                                       else float('inf'))

    def import_task_configuration(self, config_file: str):
        """Load task definitions from file"""
        with open(config_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                comp_id = row['component_id']

                # Skip if component doesn't exist
                if comp_id not in self.components:
                    continue

                task = TaskInstance(
                    identifier=row['task_name'],
                    execution_time=float(row['wcet']),
                    time_period=int(row['period']),
                    parent_component=comp_id
                )

                self.tasks.append(task)
                self.components[comp_id].contained_tasks.append(task)

        # Set priorities based on rate monotonic scheduling
        for comp in self.components.values():
            if comp.scheduling_algorithm == 'RM':
                # Sort tasks by period (ascending)
                comp.contained_tasks.sort(
                    key=lambda t: t.time_period if t.time_period > 0 else float('inf')
                )

                # Assign priorities
                for task in comp.contained_tasks:
                    task.importance = (task.time_period if task.time_period > 0
                                       else float('inf'))

    def _calculate_hyperperiod(self, processor: Processor) -> int:
        """Calculate the hyperperiod for all tasks on a processor"""
        periods = set()

        # Collect all relevant periods
        for comp in processor.assigned_components:
            if comp.allocation_period > 0:
                periods.add(comp.allocation_period)

            for task in comp.contained_tasks:
                if task.time_period > 0:
                    periods.add(task.time_period)

        # Calculate LCM of all periods
        if not periods:
            return 1

        result = 1
        for period in periods:
            result = lcm(result, period)

        return result if result > 0 else 1

    def _initialize_events(self, processor: Processor, end_time: int):
        """Initialize event queue with all scheduled events"""
        events = []

        # Budget replenishment events
        for comp in processor.assigned_components:
            if comp.allocation_period > 0:
                time = 0
                while time < end_time:
                    heapq.heappush(events, (time, 'budget', comp.identifier))
                    time += comp.allocation_period

        # Task release events
        for comp in processor.assigned_components:
            for task in comp.contained_tasks:
                if task.time_period > 0:
                    time = 0
                    while time < end_time:
                        heapq.heappush(events, (time, 'release', task.identifier))
                        time += task.time_period
                else:
                    heapq.heappush(events, (0, 'release', task.identifier))

        return events

    def simulate_processor(self, processor: Processor) -> Tuple[bool, Dict]:
        """Simulate scheduling on a single processor"""
        hyperperiod = self._calculate_hyperperiod(processor)
        processor.active_duration = 0.0

        # Track if simulation is successful
        simulation_success = True

        for comp in processor.assigned_components:
            comp.available_budget = 0.0
            for task in comp.contained_tasks:
                task.execution_left = 0.0
                task.completion_times.clear()
                task.activation_time = 0.0
                task.due_time = float('inf')

        # Create lookup dictionaries
        component_lookup = {c.identifier: c for c in processor.assigned_components}
        task_lookup = {}
        for comp in processor.assigned_components:
            for task in comp.contained_tasks:
                task_lookup[task.identifier] = task

        event_queue = self._initialize_events(processor, hyperperiod)

        # Current simulation time
        current_time = 0.0

        statistics = {}

        # Main simulation loop
        while current_time < hyperperiod:
            # Process all events at current time
            while event_queue and abs(event_queue[0][0] - current_time) < EPSILON:
                time, event_type, entity_id = heapq.heappop(event_queue)

                if event_type == 'budget':
                    # Replenish component budget
                    component = component_lookup[entity_id]
                    component.available_budget += component.allocation_amount
                elif event_type == 'release':
                    # Release a new task instance
                    task = task_lookup[entity_id]
                    task.activation_time = current_time
                    task.due_time = current_time + task.time_period if task.time_period > 0 else float('inf')

                    # Calculate adjusted execution time based on processor speed
                    task_component = component_lookup[task.parent_component]
                    proc_speed = self.processors[task_component.host_processor].performance_factor
                    task.execution_left = task.base_execution / proc_speed

            ready_components = []
            for comp in processor.assigned_components:
                if comp.available_budget > EPSILON:
                    has_ready_tasks = any(
                        task.execution_left > EPSILON and task.activation_time <= current_time
                        for task in comp.contained_tasks
                    )
                    if has_ready_tasks:
                        ready_components.append(comp)

            if not ready_components:
                # No ready components, jump to next event
                if not event_queue:
                    break
                current_time = event_queue[0][0]
                continue

            # Select component based on processor scheduling policy
            selected_component = None
            if processor.scheduling_algorithm == 'EDF':
                earliest_deadlines = {}
                for comp in ready_components:
                    earliest = float('inf')
                    for task in comp.contained_tasks:
                        if task.execution_left > EPSILON and task.activation_time <= current_time:
                            earliest = min(earliest, task.due_time)
                    earliest_deadlines[comp.identifier] = earliest

                # Select component with earliest deadline task
                selected_component = min(ready_components, key=lambda c: earliest_deadlines[c.identifier])
            else:
                # Rate Monotonic - select highest priority component
                selected_component = min(ready_components, key=lambda c: c.importance)

            # Find ready tasks in the selected component
            ready_tasks = [
                task for task in selected_component.contained_tasks
                if task.execution_left > EPSILON and task.activation_time <= current_time
            ]

            if not ready_tasks:
                # No ready tasks (shouldn't happen due to our filtering)
                if not event_queue:
                    break
                current_time = event_queue[0][0]
                continue

            selected_task = None
            if selected_component.scheduling_algorithm == 'EDF':
                selected_task = min(ready_tasks, key=lambda t: t.due_time)
            else:
                selected_task = min(ready_tasks, key=lambda t: t.importance)

            # Calculate when next event occurs
            next_event_time = event_queue[0][0] if event_queue else hyperperiod

            # Calculate execution duration
            execution_duration = min(
                selected_task.execution_left,  # Task completion
                selected_component.available_budget,  # Budget exhaustion
                next_event_time - current_time  # Next event
            )

            if execution_duration <= EPSILON:
                # No meaningful execution possible
                current_time = next_event_time
                continue

            # Perform execution
            selected_task.execution_left -= execution_duration
            selected_component.available_budget -= execution_duration
            processor.active_duration += execution_duration
            current_time += execution_duration

            # Check if task completed
            if selected_task.execution_left <= EPSILON:
                response_time = current_time - selected_task.activation_time
                selected_task.completion_times.append(response_time)

                # Check deadline
                if current_time > selected_task.due_time + EPSILON:
                    print(
                        f"!!! DEADLINE MISS: Task {selected_task.identifier} in component {selected_component.identifier} on core {processor.identifier}")
                    simulation_success = False  # Mark as failed but continue

        # Verify all periodic tasks completed
        for comp in processor.assigned_components:
            for task in comp.contained_tasks:
                if task.time_period > 0 and task.execution_left > EPSILON:
                    print(
                        f"!!! INCOMPLETE: Task {task.identifier} in {comp.identifier} on core {processor.identifier} rem={task.execution_left}")
                    simulation_success = False  # Mark as failed but continue

        # Gather performance statistics - regardless of success/failure
        for comp in processor.assigned_components:
            for task in comp.contained_tasks:
                avg_response = max_response = 0.0
                if task.completion_times:
                    avg_response = sum(task.completion_times) / len(task.completion_times)
                    max_response = max(task.completion_times)
                statistics[(comp.identifier, task.identifier)] = (avg_response, max_response)

        print(f"[Core {processor.identifier}] util≈{processor.active_duration / hyperperiod:.3f}")
        for comp in processor.assigned_components:
            print(f"  [Component {comp.identifier}] Q/P={comp.allocation_amount}/{comp.allocation_period}")
            for task in comp.contained_tasks:
                if task.completion_times:
                    print(
                        f"    Task {task.identifier}: Avg={sum(task.completion_times) / len(task.completion_times):.3f}, Max={max(task.completion_times):.3f}")
                else:
                    print(f"    Task {task.identifier}: No completions recorded.")

        return simulation_success, statistics  # Return actual statistics even if failed

    def simulate_case(self, arch_file: str, budget_file: str, task_file: str, output_dir: str) -> bool:
        """Run simulation for a complete test case"""
        self.reset()
        self.import_processor_configuration(arch_file)
        self.import_component_configuration(budget_file)
        self.import_task_configuration(task_file)

        # Create output directory
        case_name = os.path.basename(os.path.dirname(arch_file))
        results_dir = os.path.join(output_dir, case_name)
        os.makedirs(results_dir, exist_ok=True)

        # Output file
        solution_file = os.path.join(results_dir, "solution.csv")

        overall_success = True

        # Collect all results first
        all_results = {}
        processor_success = {}
        component_results = {}

        # Run simulation for each processor
        for processor in self.processors.values():
            success, results = self.simulate_processor(processor)
            processor_success[processor.identifier] = success
            all_results[processor.identifier] = results

            if not success:
                overall_success = False

            # Track component schedulability
            for comp in processor.assigned_components:
                component_results[comp.identifier] = success

        # Write all results
        with open(solution_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            # Use the exact column names requested
            writer.writerow(['task_name', 'component_id', 'task_schedulable', 'avg_response_time', 'max_response_time',
                             'component_schedulable'])

            for processor_id, results in all_results.items():
                success = processor_success[processor_id]

                # Write results for all tasks regardless of success/failure
                for (comp_id, task_id), (avg_rt, max_rt) in results.items():
                    comp_schedulable = component_results[comp_id]
                    writer.writerow([
                        task_id,  # task_name
                        comp_id,  # component_id
                        "True" if success else "False",  # task_schedulable
                        f"{avg_rt:.6f}",  # avg_response_time
                        f"{max_rt:.6f}",  # max_response_time
                        "True" if comp_schedulable else "False"  # component_schedulable
                    ])

        return overall_success


# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    # Find test cases
    cases = sorted([
        d for d in os.listdir(args.input_dir)
        if d[0].isdigit() and os.path.isdir(os.path.join(args.input_dir, d))
    ], key=lambda x: int(x.split('-', 1)[0]))

    # Run simulation
    simulator = HierarchicalSchedulingSimulator()
    total_cases = 0
    passed_cases = 0

    for case in cases:
        print(f"\n=== Running Case: {case} ===")

        # Input files
        architecture_file = os.path.join(args.input_dir, case, 'architecture.csv')
        budget_file = os.path.join(args.input_dir, case, 'budgets.csv')
        task_file = os.path.join(args.input_dir, case, 'tasks.csv')

        # Run simulation
        success = simulator.simulate_case(architecture_file, budget_file, task_file, args.output_dir)

        print(f"=== Case {case} Result: {'PASS' if success else 'FAIL'} ===")

        total_cases += 1
        if success:
            passed_cases += 1

    print(
        f"\n=== SIMULATION SUMMARY: Total={total_cases}, Passed={passed_cases}, Failed={total_cases - passed_cases} ===")