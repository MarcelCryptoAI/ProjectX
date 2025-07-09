#!/usr/bin/env python3
"""
Simple Performance Check
Basic performance analysis without external dependencies.
"""

import os
import time
import sqlite3
import json
from datetime import datetime
import argparse


def check_database_performance():
    """Check database size and table statistics"""
    results = {}
    
    # Check if database exists
    db_path = 'trading_bot.db'
    if os.path.exists(db_path):
        # Get database size
        db_size_bytes = os.path.getsize(db_path)
        db_size_mb = db_size_bytes / (1024 * 1024)
        
        results['database_size_mb'] = db_size_mb
        results['database_status'] = 'large' if db_size_mb > 100 else 'ok'
        
        # Check table statistics
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            table_stats = {}
            for table in tables:
                table_name = table[0]
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Check for indexes
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = cursor.fetchall()
                
                table_stats[table_name] = {
                    'row_count': row_count,
                    'index_count': len(indexes),
                    'needs_index': row_count > 10000 and len(indexes) == 0
                }
            
            results['table_stats'] = table_stats
            conn.close()
            
        except Exception as e:
            results['database_error'] = str(e)
    else:
        results['database_exists'] = False
    
    return results


def check_file_sizes():
    """Check sizes of important files"""
    files_to_check = [
        'web_app.py',
        'templates/dashboard.html',
        'static/js/dashboard-optimized.js'
    ]
    
    file_stats = {}
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            file_stats[file_path] = {
                'size_kb': size_kb,
                'status': 'large' if size_kb > 100 else 'ok'
            }
        else:
            file_stats[file_path] = {'exists': False}
    
    return file_stats


def analyze_dashboard_performance():
    """Analyze dashboard HTML for performance issues"""
    dashboard_path = 'templates/dashboard.html'
    issues = []
    
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for performance issues
        if 'cdn.jsdelivr.net' in content:
            external_libs = content.count('cdn.jsdelivr.net') + content.count('cdn.plot.ly') + content.count('cdnjs.cloudflare.com')
            issues.append({
                'issue': f'Multiple external CDN requests ({external_libs} detected)',
                'severity': 'medium',
                'recommendation': 'Bundle and minify JavaScript files locally'
            })
        
        if 'setInterval' in content:
            intervals = content.count('setInterval')
            issues.append({
                'issue': f'Multiple polling intervals detected ({intervals})',
                'severity': 'medium',
                'recommendation': 'Use WebSocket for real-time updates'
            })
        
        if '@keyframes' in content:
            animations = content.count('@keyframes')
            issues.append({
                'issue': f'Heavy CSS animations ({animations} keyframes)',
                'severity': 'low',
                'recommendation': 'Use will-change property and GPU acceleration'
            })
        
        if 'fetch(' in content:
            api_calls = content.count('fetch(')
            if api_calls > 5:
                issues.append({
                    'issue': f'Many API calls on page load ({api_calls})',
                    'severity': 'high',
                    'recommendation': 'Implement caching and batch requests'
                })
    
    return issues


def generate_optimization_recommendations():
    """Generate specific optimization recommendations"""
    return [
        {
            'category': 'Immediate Fixes',
            'items': [
                'Enable API response caching (60s TTL)',
                'Reduce polling frequency (30s instead of 10s)',
                'Add preload hints for critical resources'
            ]
        },
        {
            'category': 'Short-term Improvements',
            'items': [
                'Bundle JavaScript libraries locally',
                'Implement lazy loading for charts',
                'Add database indexes for large tables',
                'Use WebSocket for real-time updates'
            ]
        },
        {
            'category': 'Long-term Optimizations',
            'items': [
                'Implement service worker for offline caching',
                'Use CDN for static assets',
                'Add Redis caching layer',
                'Optimize images and use WebP format'
            ]
        }
    ]


def print_performance_report(data):
    """Print a formatted performance report"""
    print("\n" + "="*80)
    print(" SIMPLE PERFORMANCE ANALYSIS REPORT ".center(80, "="))
    print("="*80)
    
    print(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Database Performance
    print("\n" + "-"*60)
    print("DATABASE PERFORMANCE")
    print("-"*60)
    
    db_data = data['database']
    if db_data.get('database_exists', True):
        print(f"Database size: {db_data['database_size_mb']:.2f} MB")
        print(f"Status: {db_data['database_status']}")
        
        if 'table_stats' in db_data:
            print("\nTable Statistics:")
            for table, stats in db_data['table_stats'].items():
                status = "⚠️ " if stats['needs_index'] else "✅ "
                print(f"  {status}{table}: {stats['row_count']} rows, {stats['index_count']} indexes")
                if stats['needs_index']:
                    print(f"    → Recommendation: Add indexes for better performance")
    else:
        print("Database file not found")
    
    # File Sizes
    print("\n" + "-"*60)
    print("FILE SIZES")
    print("-"*60)
    
    for file_path, stats in data['file_sizes'].items():
        if stats.get('exists', True):
            status = "⚠️ " if stats['status'] == 'large' else "✅ "
            print(f"  {status}{file_path}: {stats['size_kb']:.1f} KB")
        else:
            print(f"  ❌ {file_path}: Not found")
    
    # Dashboard Issues
    print("\n" + "-"*60)
    print("DASHBOARD PERFORMANCE ISSUES")
    print("-"*60)
    
    if data['dashboard_issues']:
        for issue in data['dashboard_issues']:
            severity_icon = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢"
            print(f"  {severity_icon} {issue['issue']}")
            print(f"     → {issue['recommendation']}")
    else:
        print("  ✅ No major issues detected")
    
    # Optimization Recommendations
    print("\n" + "-"*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("-"*60)
    
    for category_data in data['recommendations']:
        print(f"\n{category_data['category']}:")
        for item in category_data['items']:
            print(f"  • {item}")
    
    # Quick Win Summary
    print("\n" + "="*80)
    print(" QUICK WINS FOR DASHBOARD PERFORMANCE ".center(80, "="))
    print("="*80)
    
    print("\n1. Enable caching (can do immediately):")
    print("   - Visit /performance page and click 'Enable Cache'")
    print("   - This will cache API responses for 60 seconds")
    print("   - Expected improvement: 50-70% faster page loads")
    
    print("\n2. Reduce update frequency:")
    print("   - Change dashboard update from 10s to 30s")
    print("   - Change status update from 15s to 60s")
    print("   - Expected improvement: 60% less server load")
    
    print("\n3. Optimize JavaScript loading:")
    print("   - Run './optimize_dashboard.py' to apply optimizations")
    print("   - This adds preload hints and lazy loading")
    print("   - Expected improvement: 30% faster initial load")
    
    if db_data.get('database_size_mb', 0) > 100:
        print("\n4. Database optimization needed:")
        print(f"   - Database is {db_data['database_size_mb']:.1f} MB")
        print("   - Consider archiving old trade data")
        print("   - Add indexes to frequently queried columns")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Simple Performance Check')
    parser.add_argument('--save', action='store_true', help='Save report to JSON file')
    args = parser.parse_args()
    
    print("🔍 Running simple performance check...")
    
    # Collect performance data
    data = {
        'timestamp': datetime.now().isoformat(),
        'database': check_database_performance(),
        'file_sizes': check_file_sizes(),
        'dashboard_issues': analyze_dashboard_performance(),
        'recommendations': generate_optimization_recommendations()
    }
    
    # Print report
    print_performance_report(data)
    
    # Save to file if requested
    if args.save:
        filename = f"performance_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"📄 Report saved to: {filename}")


if __name__ == "__main__":
    main()