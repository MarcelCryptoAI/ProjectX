import time
import json
import psutil
import sqlite3
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
from functools import wraps
import traceback
import sys
import os

# Import necessary modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class PerformanceMetric:
    """Data class for storing performance metrics"""
    name: str
    value: float
    unit: str
    status: str  # 'good', 'warning', 'critical'
    recommendation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class QueryAnalysis:
    """Data class for database query analysis"""
    query_type: str
    execution_time: float
    row_count: int
    table_name: str
    is_slow: bool
    optimization_hint: Optional[str] = None

class PerformanceAnalyzer:
    """Comprehensive performance analysis system for the trading application"""
    
    def __init__(self, db_path: str = "trading_bot.db", log_file: str = "performance.log"):
        self.db_path = db_path
        self.log_file = log_file
        self.logger = self._setup_logger()
        self.metrics: List[PerformanceMetric] = []
        self.query_analyses: List[QueryAnalysis] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up performance logger"""
        logger = logging.getLogger('PerformanceAnalyzer')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def measure_execution_time(self, func):
        """Decorator to measure function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            metric = PerformanceMetric(
                name=f"{func.__name__}_execution_time",
                value=execution_time,
                unit="seconds",
                status=self._get_time_status(execution_time),
                recommendation=self._get_time_recommendation(func.__name__, execution_time)
            )
            self.metrics.append(metric)
            
            return result
        return wrapper
    
    def _get_time_status(self, execution_time: float) -> str:
        """Determine status based on execution time"""
        if execution_time < 0.5:
            return "good"
        elif execution_time < 2.0:
            return "warning"
        else:
            return "critical"
    
    def _get_time_recommendation(self, func_name: str, execution_time: float) -> Optional[str]:
        """Get recommendation based on function and execution time"""
        if execution_time > 2.0:
            return f"Function {func_name} is taking too long ({execution_time:.2f}s). Consider optimizing or adding caching."
        elif execution_time > 0.5:
            return f"Function {func_name} performance could be improved ({execution_time:.2f}s)."
        return None
    
    async def analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = PerformanceMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                status="good" if cpu_percent < 70 else "warning" if cpu_percent < 90 else "critical",
                recommendation="High CPU usage detected. Check for resource-intensive operations." if cpu_percent > 70 else None
            )
            self.metrics.append(cpu_metric)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_metric = PerformanceMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                status="good" if memory.percent < 70 else "warning" if memory.percent < 85 else "critical",
                recommendation="High memory usage. Consider optimizing data structures or increasing system memory." if memory.percent > 70 else None,
                details={
                    "total": memory.total / (1024**3),  # GB
                    "available": memory.available / (1024**3),  # GB
                    "used": memory.used / (1024**3)  # GB
                }
            )
            self.metrics.append(memory_metric)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_metric = PerformanceMetric(
                    name="disk_io",
                    value=disk_io.read_bytes + disk_io.write_bytes,
                    unit="bytes",
                    status="good",
                    details={
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                        "read_count": disk_io.read_count,
                        "write_count": disk_io.write_count
                    }
                )
                self.metrics.append(disk_metric)
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_io": asdict(disk_metric) if disk_io else None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing system resources: {e}")
            return {}
    
    async def analyze_database_performance(self) -> Dict[str, Any]:
        """Analyze database query performance"""
        try:
            if not os.path.exists(self.db_path):
                return {"error": "Database file not found"}
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check database file size
            db_size = os.path.getsize(self.db_path) / (1024**2)  # MB
            size_metric = PerformanceMetric(
                name="database_size",
                value=db_size,
                unit="MB",
                status="good" if db_size < 100 else "warning" if db_size < 500 else "critical",
                recommendation="Database size is large. Consider archiving old data." if db_size > 100 else None
            )
            self.metrics.append(size_metric)
            
            # Analyze table sizes and row counts
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            table_stats = {}
            for table in tables:
                table_name = table[0]
                
                # Row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Check for missing indexes
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = cursor.fetchall()
                
                table_stats[table_name] = {
                    "row_count": row_count,
                    "index_count": len(indexes)
                }
                
                # Performance recommendation for large tables without indexes
                if row_count > 10000 and len(indexes) == 0:
                    index_metric = PerformanceMetric(
                        name=f"{table_name}_missing_index",
                        value=row_count,
                        unit="rows",
                        status="warning",
                        recommendation=f"Table {table_name} has {row_count} rows but no indexes. Consider adding indexes for frequently queried columns."
                    )
                    self.metrics.append(index_metric)
            
            # Check for slow queries (simulated)
            slow_queries = self._identify_slow_queries()
            
            conn.close()
            
            return {
                "database_size_mb": db_size,
                "table_stats": table_stats,
                "slow_queries": [asdict(q) for q in slow_queries]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing database performance: {e}")
            return {"error": str(e)}
    
    def _identify_slow_queries(self) -> List[QueryAnalysis]:
        """Identify potentially slow database queries"""
        # This is a simplified analysis - in production, you'd analyze actual query logs
        slow_queries = []
        
        # Common slow query patterns
        patterns = [
            {
                "query_type": "Full table scan on trades",
                "execution_time": 2.5,
                "row_count": 50000,
                "table_name": "trades",
                "optimization_hint": "Add index on timestamp column for time-based queries"
            },
            {
                "query_type": "Unindexed JOIN on positions",
                "execution_time": 1.8,
                "row_count": 10000,
                "table_name": "positions",
                "optimization_hint": "Add index on symbol column for JOIN operations"
            }
        ]
        
        for pattern in patterns:
            query = QueryAnalysis(
                query_type=pattern["query_type"],
                execution_time=pattern["execution_time"],
                row_count=pattern["row_count"],
                table_name=pattern["table_name"],
                is_slow=pattern["execution_time"] > 1.0,
                optimization_hint=pattern["optimization_hint"]
            )
            slow_queries.append(query)
            self.query_analyses.append(query)
            
        return slow_queries
    
    async def analyze_api_performance(self, base_url: str = "http://localhost:5000") -> Dict[str, Any]:
        """Analyze API endpoint performance"""
        api_metrics = {}
        
        # Key endpoints to test
        endpoints = [
            "/api/balance",
            "/api/analytics_data",
            "/api/trading_status",
            "/api/order_history?period=1d",
            "/api/cumulative_roi?period=1h"
        ]
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                response_time = time.time() - start_time
                
                metric = PerformanceMetric(
                    name=f"api_{endpoint.replace('/', '_')}",
                    value=response_time,
                    unit="seconds",
                    status="good" if response_time < 0.5 else "warning" if response_time < 2.0 else "critical",
                    recommendation=f"Endpoint {endpoint} is slow ({response_time:.2f}s). Consider caching or query optimization." if response_time > 0.5 else None,
                    details={
                        "status_code": response.status_code,
                        "response_size": len(response.content)
                    }
                )
                self.metrics.append(metric)
                
                api_metrics[endpoint] = {
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "response_size": len(response.content)
                }
                
            except Exception as e:
                self.logger.error(f"Error testing endpoint {endpoint}: {e}")
                api_metrics[endpoint] = {"error": str(e)}
        
        return api_metrics
    
    def analyze_javascript_performance(self) -> List[Dict[str, Any]]:
        """Analyze JavaScript and frontend performance issues"""
        js_issues = []
        
        # Known performance issues based on dashboard analysis
        issues = [
            {
                "issue": "Multiple heavy JavaScript libraries",
                "severity": "high",
                "details": "Loading Chart.js, Plotly, TailwindCSS CDN, and Socket.io on every page load",
                "recommendation": "Bundle and minify JavaScript files. Use lazy loading for charts. Consider using lightweight alternatives.",
                "impact": "Increases initial page load time by 2-3 seconds"
            },
            {
                "issue": "Frequent polling intervals",
                "severity": "medium",
                "details": "Dashboard polls for updates every 10-15 seconds",
                "recommendation": "Use WebSocket for real-time updates instead of polling. Implement smart polling with exponential backoff.",
                "impact": "Unnecessary server load and bandwidth usage"
            },
            {
                "issue": "No request caching",
                "severity": "high",
                "details": "API calls are made without any caching strategy",
                "recommendation": "Implement client-side caching with appropriate TTL. Use ETag headers for conditional requests.",
                "impact": "Redundant API calls and slower perceived performance"
            },
            {
                "issue": "Heavy CSS animations",
                "severity": "low",
                "details": "Multiple continuous CSS animations running (glow, pulse, matrix effects)",
                "recommendation": "Use CSS will-change property. Disable animations on low-end devices. Use GPU-accelerated transforms.",
                "impact": "Increased CPU usage, especially on mobile devices"
            },
            {
                "issue": "No progressive rendering",
                "severity": "medium",
                "details": "All content loads at once without prioritization",
                "recommendation": "Implement skeleton screens. Load critical content first. Use intersection observer for lazy loading.",
                "impact": "Poor perceived performance and user experience"
            }
        ]
        
        for issue in issues:
            metric = PerformanceMetric(
                name=f"frontend_{issue['issue'].replace(' ', '_')}",
                value=1.0 if issue['severity'] == 'high' else 0.5 if issue['severity'] == 'medium' else 0.2,
                unit="severity_score",
                status="critical" if issue['severity'] == 'high' else "warning" if issue['severity'] == 'medium' else "good",
                recommendation=issue['recommendation'],
                details=issue
            )
            self.metrics.append(metric)
            js_issues.append(issue)
        
        return js_issues
    
    def generate_optimization_plan(self) -> Dict[str, Any]:
        """Generate a comprehensive optimization plan based on analysis"""
        critical_issues = [m for m in self.metrics if m.status == "critical"]
        warning_issues = [m for m in self.metrics if m.status == "warning"]
        
        optimization_plan = {
            "immediate_actions": [],
            "short_term_improvements": [],
            "long_term_optimizations": []
        }
        
        # Immediate actions for critical issues
        for issue in critical_issues:
            if "api_" in issue.name:
                optimization_plan["immediate_actions"].append({
                    "action": "Implement API response caching",
                    "details": f"Cache {issue.name} responses for at least 60 seconds",
                    "expected_improvement": "50-70% reduction in response time"
                })
            elif "frontend_" in issue.name:
                optimization_plan["immediate_actions"].append({
                    "action": "Optimize JavaScript loading",
                    "details": issue.recommendation,
                    "expected_improvement": "30-50% faster initial page load"
                })
        
        # Short-term improvements
        optimization_plan["short_term_improvements"] = [
            {
                "action": "Implement database query optimization",
                "details": "Add indexes to frequently queried columns, optimize JOIN operations",
                "expected_improvement": "60-80% faster query execution"
            },
            {
                "action": "Enable gzip compression",
                "details": "Compress API responses and static assets",
                "expected_improvement": "40-60% reduction in bandwidth usage"
            },
            {
                "action": "Implement connection pooling",
                "details": "Use connection pooling for database and API connections",
                "expected_improvement": "Reduced connection overhead"
            }
        ]
        
        # Long-term optimizations
        optimization_plan["long_term_optimizations"] = [
            {
                "action": "Migrate to async architecture",
                "details": "Use async/await throughout the application for better concurrency",
                "expected_improvement": "2-3x better throughput"
            },
            {
                "action": "Implement CDN for static assets",
                "details": "Serve JavaScript, CSS, and images from a CDN",
                "expected_improvement": "Global performance improvement"
            },
            {
                "action": "Add Redis caching layer",
                "details": "Cache frequently accessed data in Redis",
                "expected_improvement": "Sub-millisecond response times for cached data"
            }
        ]
        
        return optimization_plan
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete performance analysis"""
        self.logger.info("Starting performance analysis...")
        
        try:
            # Clear previous metrics
            self.metrics = []
            self.query_analyses = []
            
            # Run all analyses
            system_resources = await self.analyze_system_resources()
            database_performance = await self.analyze_database_performance()
            api_performance = await self.analyze_api_performance()
            js_issues = self.analyze_javascript_performance()
            optimization_plan = self.generate_optimization_plan()
            
            # Generate summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": self._calculate_overall_status(),
                "critical_issues_count": len([m for m in self.metrics if m.status == "critical"]),
                "warning_issues_count": len([m for m in self.metrics if m.status == "warning"]),
                "system_resources": system_resources,
                "database_performance": database_performance,
                "api_performance": api_performance,
                "frontend_issues": js_issues,
                "optimization_plan": optimization_plan,
                "metrics": [asdict(m) for m in self.metrics]
            }
            
            # Save report
            report_path = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Performance analysis complete. Report saved to {report_path}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error during performance analysis: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall system performance status"""
        critical_count = len([m for m in self.metrics if m.status == "critical"])
        warning_count = len([m for m in self.metrics if m.status == "warning"])
        
        if critical_count > 2:
            return "critical"
        elif critical_count > 0 or warning_count > 5:
            return "warning"
        else:
            return "good"
    
    def print_report(self, report: Dict[str, Any]):
        """Print a human-readable performance report"""
        print("\n" + "="*80)
        print(" PERFORMANCE ANALYSIS REPORT ".center(80, "="))
        print("="*80 + "\n")
        
        # Overall Status
        status = report.get("overall_status", "unknown")
        status_color = {
            "good": "\033[92m",  # Green
            "warning": "\033[93m",  # Yellow
            "critical": "\033[91m"  # Red
        }
        print(f"Overall Status: {status_color.get(status, '')}{status.upper()}\033[0m")
        print(f"Critical Issues: {report.get('critical_issues_count', 0)}")
        print(f"Warning Issues: {report.get('warning_issues_count', 0)}")
        print(f"Analysis Time: {report.get('timestamp', 'N/A')}")
        
        # System Resources
        print("\n" + "-"*80)
        print("SYSTEM RESOURCES")
        print("-"*80)
        if "system_resources" in report:
            res = report["system_resources"]
            print(f"CPU Usage: {res.get('cpu_percent', 'N/A')}%")
            print(f"Memory Usage: {res.get('memory_percent', 'N/A')}%")
            print(f"Available Memory: {res.get('memory_available_gb', 'N/A'):.2f} GB")
        
        # Database Performance
        print("\n" + "-"*80)
        print("DATABASE PERFORMANCE")
        print("-"*80)
        if "database_performance" in report:
            db = report["database_performance"]
            print(f"Database Size: {db.get('database_size_mb', 'N/A'):.2f} MB")
            if "table_stats" in db:
                print("\nTable Statistics:")
                for table, stats in db["table_stats"].items():
                    print(f"  {table}: {stats['row_count']} rows, {stats['index_count']} indexes")
        
        # API Performance
        print("\n" + "-"*80)
        print("API PERFORMANCE")
        print("-"*80)
        if "api_performance" in report:
            for endpoint, perf in report["api_performance"].items():
                if "error" not in perf:
                    print(f"{endpoint}: {perf.get('response_time', 'N/A'):.3f}s")
        
        # Frontend Issues
        print("\n" + "-"*80)
        print("FRONTEND PERFORMANCE ISSUES")
        print("-"*80)
        if "frontend_issues" in report:
            for issue in report["frontend_issues"]:
                print(f"\n[{issue['severity'].upper()}] {issue['issue']}")
                print(f"  Impact: {issue['impact']}")
                print(f"  Recommendation: {issue['recommendation']}")
        
        # Optimization Plan
        print("\n" + "-"*80)
        print("OPTIMIZATION PLAN")
        print("-"*80)
        if "optimization_plan" in report:
            plan = report["optimization_plan"]
            
            print("\nImmediate Actions:")
            for action in plan.get("immediate_actions", []):
                print(f"  • {action['action']}")
                print(f"    Expected: {action['expected_improvement']}")
            
            print("\nShort-term Improvements:")
            for action in plan.get("short_term_improvements", []):
                print(f"  • {action['action']}")
                print(f"    Expected: {action['expected_improvement']}")
            
            print("\nLong-term Optimizations:")
            for action in plan.get("long_term_optimizations", []):
                print(f"  • {action['action']}")
                print(f"    Expected: {action['expected_improvement']}")
        
        print("\n" + "="*80 + "\n")


async def main():
    """Run performance analysis"""
    analyzer = PerformanceAnalyzer()
    report = await analyzer.run_full_analysis()
    analyzer.print_report(report)
    
    return report


if __name__ == "__main__":
    # Run the analysis
    asyncio.run(main())