#!/usr/bin/env python3
"""
Performance Analysis Script for ByBit Trading Bot
Run this script to analyze system performance and get optimization recommendations.
"""

import asyncio
import argparse
import json
from datetime import datetime
from performance_analyzer import PerformanceAnalyzer


def save_report_to_file(report, filename=None):
    """Save performance report to a file"""
    if filename is None:
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {filename}")
    return filename


def create_html_report(report, filename=None):
    """Create an HTML version of the performance report"""
    if filename is None:
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Report - {report['timestamp']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #0A0A0F;
            color: #ffffff;
            margin: 20px;
        }}
        h1, h2, h3 {{
            color: #00D9FF;
        }}
        .status-good {{
            color: #008844;
        }}
        .status-warning {{
            color: #FFD700;
        }}
        .status-critical {{
            color: #CC0000;
        }}
        .metric {{
            background-color: #0F0F1A;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border: 1px solid #1A1A2E;
        }}
        .recommendation {{
            background-color: #16213E;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #FFD700;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #1A1A2E;
        }}
        th {{
            background-color: #0F0F1A;
            color: #00D9FF;
        }}
    </style>
</head>
<body>
    <h1>Performance Analysis Report</h1>
    <p>Generated: {report['timestamp']}</p>
    
    <h2>Overall Status: <span class="status-{report['overall_status']}">{report['overall_status'].upper()}</span></h2>
    <p>Critical Issues: {report['critical_issues_count']} | Warnings: {report['warning_issues_count']}</p>
    
    <h2>System Resources</h2>
    <div class="metric">
        <p>CPU Usage: {report['system_resources']['cpu_percent']}%</p>
        <p>Memory Usage: {report['system_resources']['memory_percent']}%</p>
        <p>Available Memory: {report['system_resources']['memory_available_gb']:.2f} GB</p>
    </div>
    
    <h2>Database Performance</h2>
    <div class="metric">
        <p>Database Size: {report['database_performance'].get('database_size_mb', 0):.2f} MB</p>
    </div>
    
    <h2>Performance Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Status</th>
            <th>Recommendation</th>
        </tr>
"""
    
    for metric in report.get('metrics', []):
        status_class = f"status-{metric['status']}"
        recommendation = metric.get('recommendation', '-')
        html_content += f"""
        <tr>
            <td>{metric['name']}</td>
            <td>{metric['value']:.2f} {metric['unit']}</td>
            <td class="{status_class}">{metric['status'].upper()}</td>
            <td>{recommendation}</td>
        </tr>
"""
    
    html_content += """
    </table>
    
    <h2>Optimization Plan</h2>
"""
    
    plan = report.get('optimization_plan', {})
    
    if plan.get('immediate_actions'):
        html_content += "<h3>Immediate Actions</h3>"
        for action in plan['immediate_actions']:
            html_content += f"""
            <div class="recommendation">
                <strong>{action['action']}</strong><br>
                {action['details']}<br>
                <em>Expected: {action['expected_improvement']}</em>
            </div>
"""
    
    if plan.get('short_term_improvements'):
        html_content += "<h3>Short-term Improvements</h3>"
        for action in plan['short_term_improvements']:
            html_content += f"""
            <div class="recommendation">
                <strong>{action['action']}</strong><br>
                {action['details']}<br>
                <em>Expected: {action['expected_improvement']}</em>
            </div>
"""
    
    if plan.get('long_term_optimizations'):
        html_content += "<h3>Long-term Optimizations</h3>"
        for action in plan['long_term_optimizations']:
            html_content += f"""
            <div class="recommendation">
                <strong>{action['action']}</strong><br>
                {action['details']}<br>
                <em>Expected: {action['expected_improvement']}</em>
            </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(filename, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {filename}")
    return filename


async def main():
    parser = argparse.ArgumentParser(description='Analyze ByBit Trading Bot Performance')
    parser.add_argument('--save-json', action='store_true', 
                        help='Save report as JSON file')
    parser.add_argument('--save-html', action='store_true', 
                        help='Save report as HTML file')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick analysis only')
    parser.add_argument('--api-url', default='http://localhost:5000',
                        help='API URL for testing (default: http://localhost:5000)')
    
    args = parser.parse_args()
    
    print("Starting Performance Analysis...")
    print("=" * 80)
    
    analyzer = PerformanceAnalyzer()
    
    if args.quick:
        print("\nRunning quick performance check...\n")
        # Quick check - just system resources and basic metrics
        system_resources = await analyzer.analyze_system_resources()
        
        print(f"CPU Usage: {system_resources['cpu_percent']}%")
        print(f"Memory Usage: {system_resources['memory_percent']}%")
        print(f"Available Memory: {system_resources['memory_available_gb']:.2f} GB")
        
        # Check database size
        import os
        if os.path.exists('trading_bot.db'):
            db_size = os.path.getsize('trading_bot.db') / (1024**2)
            print(f"\nDatabase Size: {db_size:.2f} MB")
            if db_size > 100:
                print("⚠️  Warning: Database is large. Consider archiving old data.")
        
        print("\nQuick Recommendations:")
        print("1. If dashboard loads slowly, enable caching in the web interface")
        print("2. Consider bundling JavaScript files for faster loading")
        print("3. Use WebSocket instead of polling for real-time updates")
        
    else:
        # Full analysis
        report = await analyzer.run_full_analysis()
        
        # Print the report
        analyzer.print_report(report)
        
        # Save reports if requested
        if args.save_json:
            save_report_to_file(report)
        
        if args.save_html:
            create_html_report(report)
        
        # Print specific dashboard optimization advice
        print("\n" + "="*80)
        print(" DASHBOARD OPTIMIZATION ADVICE ".center(80, "="))
        print("="*80 + "\n")
        
        print("Based on the analysis, here are specific steps to improve dashboard performance:\n")
        
        print("1. IMMEDIATE FIXES (Can do right now):")
        print("   - Enable caching by clicking 'Enable Cache' in the Performance page")
        print("   - This will cache API responses and reduce server load")
        print("   - Expected improvement: 50-70% faster page loads\n")
        
        print("2. JAVASCRIPT OPTIMIZATION (Developer task):")
        print("   - Bundle Chart.js, Plotly, and other libraries")
        print("   - Use a CDN with preload/prefetch hints")
        print("   - Lazy load charts only when visible")
        print("   - Expected improvement: 2-3 seconds faster initial load\n")
        
        print("3. REAL-TIME DATA (Developer task):")
        print("   - Replace polling with WebSocket push updates")
        print("   - Implement smart polling with exponential backoff")
        print("   - Expected improvement: 90% reduction in unnecessary API calls\n")
        
        print("4. DATABASE OPTIMIZATION:")
        if 'database_performance' in report:
            db_size = report['database_performance'].get('database_size_mb', 0)
            if db_size > 50:
                print(f"   - Database is {db_size:.1f} MB - consider archiving old trades")
                print("   - Add indexes on frequently queried columns")
                print("   - Expected improvement: 60-80% faster queries")
            else:
                print("   - Database size is reasonable")
                print("   - Still consider adding indexes for better query performance")
        
        print("\n5. QUICK WIN - Disable animations on slow devices:")
        print("   - Add a 'Reduce Motion' toggle in settings")
        print("   - This will disable CPU-intensive animations")
        print("   - Expected improvement: Smoother experience on older devices")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())