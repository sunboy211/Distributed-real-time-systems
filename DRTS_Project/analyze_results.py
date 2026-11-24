# analyze_results.py
import os
import pandas as pd
import numpy as np
from collections import defaultdict


def collect_response_time_data(results_dir='results'):

    component_data = defaultdict(list)


    for i in range(1, 11):
        solution_file = os.path.join(results_dir, str(i), 'solution.csv')

        if os.path.exists(solution_file):
            # read solution.csv
            df = pd.read_csv(solution_file)

            # Collect data grouped by component type
            for _, row in df.iterrows():
                component_type = row['component_id']

                # Handling different formats for schedulable fields (True/False, 1/0, “True”/“False”)
                schedulable = str(row['task_schedulable']).strip().lower() in ['true', '1']

                component_data[component_type].append({
                    'test_case': i,
                    'task_name': row['task_name'],
                    'avg_response_time': float(row['avg_response_time']),
                    'max_response_time': float(row['max_response_time']),
                    'schedulable': schedulable
                })
        else:
            print(f"Warning: {solution_file} not found")

    return component_data


def calculate_statistics(component_data):

    statistics = {}

    for component_type, tasks in component_data.items():
        if not tasks:
            continue

        # Extract all response time data
        avg_times = [task['avg_response_time'] for task in tasks]
        max_times = [task['max_response_time'] for task in tasks]

        # Calculation of the dispatchability ratio
        schedulable_count = sum(1 for task in tasks if task['schedulable'])
        schedulable_ratio = schedulable_count / len(tasks) if tasks else 0

        # Calculate jitter for each task
        task_jitters = [task['max_response_time'] - task['avg_response_time']
                        for task in tasks]

        # Calculation of statistical indicators
        statistics[component_type] = {
            'count': len(tasks),
            'avg_response_time': np.mean(avg_times),
            'max_response_time': np.max(max_times),
            'min_avg_time': np.min(avg_times),
            'max_avg_time': np.max(avg_times),
            'percentile_95': np.percentile(avg_times, 95) if avg_times else 0,
            'jitter': np.mean(task_jitters),  # 平均jitter
            'max_jitter': np.max(task_jitters),  # 最大jitter
            'std_dev': np.std(avg_times),
            'schedulable_ratio': schedulable_ratio,
            'schedulable_count': schedulable_count
        }

    return statistics


def generate_report(statistics, output_file='component_analysis.csv'):

    # create DataFrame
    df = pd.DataFrame.from_dict(statistics, orient='index')
    df.index.name = 'Component_Type'

    # Sort by component type
    df = df.sort_index()

    # save to CSV
    df.to_csv(output_file, float_format='%.6f')


    print("\n=== Component Response Time Analysis ===\n")
    print(
        f"{'Component Type':<25} {'Count':<6} {'Avg RT':<12} {'Max RT':<12} {'95%ile':<12} {'Jitter':<12} {'Sched%':<8}")
    print("-" * 95)

    for component_type, stats in statistics.items():
        print(f"{component_type:<25} {stats['count']:<6} "
              f"{stats['avg_response_time']:<12.3f} {stats['max_response_time']:<12.3f} "
              f"{stats['percentile_95']:<12.3f} {stats['jitter']:<12.3f} "
              f"{stats['schedulable_ratio'] * 100:<8.1f}")


    print("\n=== Detailed Statistics ===\n")
    for component_type, stats in statistics.items():
        print(f"\n{component_type}:")
        print(f"  Tasks Count: {stats['count']}")
        print(f"  Average Response Time: {stats['avg_response_time']:.3f} ms")
        print(f"  Maximum Response Time: {stats['max_response_time']:.3f} ms")
        print(f"  95th Percentile: {stats['percentile_95']:.3f} ms")
        print(f"  Average Jitter: {stats['jitter']:.3f} ms")
        print(f"  Maximum Jitter: {stats['max_jitter']:.3f} ms")
        print(f"  Standard Deviation: {stats['std_dev']:.3f} ms")
        print(
            f"  Schedulability: {stats['schedulable_ratio'] * 100:.1f}% ({stats['schedulable_count']}/{stats['count']})")
        print(f"  Min Average Time: {stats['min_avg_time']:.3f} ms")
        print(f"  Max Average Time: {stats['max_avg_time']:.3f} ms")


def visualize_data(component_data, statistics):

    try:
        import matplotlib.pyplot as plt

        # Creating a Maximum Response Time Boxplot for a Component Type
        plt.figure(figsize=(12, 8))

        component_types = []
        max_times_data = []

        for component_type, tasks in component_data.items():
            component_types.append(component_type)
            max_times = [task['max_response_time'] for task in tasks]
            max_times_data.append(max_times)

        # Using tick_labels instead of labels to avoid warnings
        bp = plt.boxplot(max_times_data, tick_labels=component_types)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Maximum Response Time (ms)')
        plt.title('Maximum Response Time Distribution by Component Type')
        plt.tight_layout()
        plt.savefig('max_response_time_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create average response time boxplots
        plt.figure(figsize=(12, 8))

        avg_times_data = []
        for component_type, tasks in component_data.items():
            avg_times = [task['avg_response_time'] for task in tasks]
            avg_times_data.append(avg_times)

        bp = plt.boxplot(avg_times_data, tick_labels=component_types)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Response Time (ms)')
        plt.title('Average Response Time Distribution by Component Type')
        plt.tight_layout()
        plt.savefig('avg_response_time_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create jitter comparison chart (using corrected jitter)
        plt.figure(figsize=(12, 6))
        components = list(statistics.keys())
        jitters = [stats['jitter'] for stats in statistics.values()]

        bars = plt.bar(components, jitters)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Jitter (ms)')
        plt.title('Average Response Time Jitter by Component Type')


        for bar, jitter in zip(bars, jitters):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{jitter:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('jitter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Creating a Maximum Jitter Comparison Chart
        plt.figure(figsize=(12, 6))
        max_jitters = [stats['max_jitter'] for stats in statistics.values()]

        bars = plt.bar(components, max_jitters)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Maximum Jitter (ms)')
        plt.title('Maximum Response Time Jitter by Component Type')


        for bar, jitter in zip(bars, max_jitters):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{jitter:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('max_jitter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("\nVisualization plots saved:")
        print("  - max_response_time_boxplot.png")
        print("  - avg_response_time_boxplot.png")
        print("  - jitter_comparison.png")
        print("  - max_jitter_comparison.png")

    except ImportError:
        print("\nMatplotlib not available for visualization. Install with: pip install matplotlib")


def main():
    # Collection of data
    component_data = collect_response_time_data('results')

    # Calculation of statistical information
    statistics = calculate_statistics(component_data)

    # generate report
    generate_report(statistics)


    visualize_data(component_data, statistics)

    # Generating Component Performance Categories
    print("\n=== Component Performance Classification ===\n")

    # Categorized by average response time
    low_latency = []  # < 10ms
    medium_latency = []  # 10-50ms
    high_latency = []  # > 50ms

    for comp_type, stats in statistics.items():
        avg_rt = stats['avg_response_time']
        if avg_rt < 10:
            low_latency.append((comp_type, avg_rt))
        elif avg_rt < 50:
            medium_latency.append((comp_type, avg_rt))
        else:
            high_latency.append((comp_type, avg_rt))

    print(f"Low Latency Components (<10ms): {len(low_latency)}")
    for comp, rt in sorted(low_latency, key=lambda x: x[1]):
        print(f"  {comp}: {rt:.3f} ms")

    print(f"\nMedium Latency Components (10-50ms): {len(medium_latency)}")
    for comp, rt in sorted(medium_latency, key=lambda x: x[1]):
        print(f"  {comp}: {rt:.3f} ms")

    print(f"\nHigh Latency Components (>50ms): {len(high_latency)}")
    for comp, rt in sorted(high_latency, key=lambda x: x[1]):
        print(f"  {comp}: {rt:.3f} ms")

    # Identifying Performance Problem Components
    print("\n=== Components with Performance Issues ===\n")

    problem_components = []

    for comp_type, stats in statistics.items():
        issues = []

        # High jitter problems (using average jitter)
        if stats['jitter'] > 50:  # 降低阈值，因为现在是平均jitter
            issues.append(f"High average jitter: {stats['jitter']:.1f}ms")

        # Extremely high maximum jitter problem
        if stats['max_jitter'] > 100:
            issues.append(f"Very high max jitter: {stats['max_jitter']:.1f}ms")

        # Schedulability problem - only a problem if the ratio is less than 1.0
        if stats['schedulable_ratio'] < 1.0:
            percentage = stats['schedulable_ratio'] * 100
            failed_count = stats['count'] - stats['schedulable_count']
            issues.append(f"Schedulability issues: {percentage:.1f}% ({failed_count} tasks failed)")

        # High response time problem
        if stats['max_response_time'] > 200:
            issues.append(f"Very high max RT: {stats['max_response_time']:.1f}ms")

        # High variance problem
        if stats['std_dev'] > 50:
            issues.append(f"High variance: std={stats['std_dev']:.1f}ms")

        if issues:
            problem_components.append(comp_type)
            print(f"{comp_type}:")
            for issue in issues:
                print(f"  - {issue}")
            print()

    if not problem_components:
        print("No components with significant performance issues detected.")

    # final  report
    print("\n=== System Summary ===\n")
    total_tasks = sum(stats['count'] for stats in statistics.values())
    total_schedulable = sum(stats['schedulable_count'] for stats in statistics.values())
    overall_schedulability = (total_schedulable / total_tasks * 100) if total_tasks > 0 else 0

    print(f"Total Tasks: {total_tasks}")
    print(f"Total Schedulable: {total_schedulable}")
    print(f"Overall Schedulability: {overall_schedulability:.1f}%")
    print(f"Number of Component Types: {len(statistics)}")
    print(f"Components with Issues: {len(problem_components)}")

    if problem_components:
        print(f"Problematic Components: {', '.join(problem_components)}")

    # Additional jitter analysis
    print("\n=== Jitter Analysis ===\n")


    sorted_by_jitter = sorted(statistics.items(), key=lambda x: x[1]['jitter'], reverse=True)

    print("Top 5 Components with Highest Average Jitter:")
    for i, (comp_type, stats) in enumerate(sorted_by_jitter[:5]):
        print(f"  {i + 1}. {comp_type}: {stats['jitter']:.1f}ms (max: {stats['max_jitter']:.1f}ms)")

    print("\nTop 5 Components with Lowest Average Jitter:")
    for i, (comp_type, stats) in enumerate(sorted_by_jitter[-5:]):
        print(f"  {i + 1}. {comp_type}: {stats['jitter']:.1f}ms (max: {stats['max_jitter']:.1f}ms)")


if __name__ == "__main__":
    main()