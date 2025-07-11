"""
Production Dashboard for NanoBrain Viral Protein Analysis

This module provides a comprehensive production monitoring dashboard
with real-time metrics, health checks, and system status.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from nanobrain.core.logging_system import get_logger
from .performance_monitor import performance_monitor, PerformanceMetrics


@dataclass
class SystemHealth:
    """System health status"""
    status: str  # "healthy", "warning", "critical"
    uptime_seconds: float
    last_check: datetime
    components: Dict[str, str]  # component -> status
    alerts: List[Dict[str, Any]]


class ProductionDashboard:
    """
    Production monitoring dashboard for viral protein analysis workflow.
    
    Features:
    - Real-time performance metrics
    - System health monitoring
    - Workflow execution tracking
    - Alert management
    - API status monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("production_dashboard")
        
        # Dashboard state
        self.start_time = datetime.now()
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.system_alerts: List[Dict[str, Any]] = []
        self.connected_clients: List[WebSocket] = []
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.last_health_check = None
        self.system_health = SystemHealth(
            status="unknown",
            uptime_seconds=0,
            last_check=datetime.now(),
            components={},
            alerts=[]
        )
        
        # Create FastAPI app
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with dashboard endpoints"""
        app = FastAPI(
            title="NanoBrain Viral Protein Analysis - Production Dashboard",
            description="Real-time monitoring and management dashboard",
            version="4.0.0"
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
        
    def _add_routes(self, app: FastAPI) -> None:
        """Add dashboard routes"""
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Main dashboard page"""
            return self._get_dashboard_html()
            
        @app.get("/health")
        async def health_check():
            """System health check endpoint"""
            health_status = await self._perform_health_check()
            return {
                "status": health_status.status,
                "uptime_seconds": health_status.uptime_seconds,
                "last_check": health_status.last_check.isoformat(),
                "components": health_status.components,
                "alerts": health_status.alerts
            }
            
        @app.get("/metrics")
        async def get_metrics():
            """Get current performance metrics"""
            return performance_monitor.get_performance_summary()
            
        @app.get("/prometheus")
        async def prometheus_metrics():
            """Prometheus metrics endpoint"""
            metrics = await performance_monitor.export_metrics_to_prometheus()
            return HTMLResponse(content=metrics, media_type="text/plain")
            
        @app.get("/workflows")
        async def get_active_workflows():
            """Get currently active workflows"""
            return {
                "active_workflows": self.active_workflows,
                "total_active": len(self.active_workflows)
            }
            
        @app.get("/workflows/{workflow_id}")
        async def get_workflow_status(workflow_id: str):
            """Get status of specific workflow"""
            if workflow_id not in self.active_workflows:
                raise HTTPException(status_code=404, detail="Workflow not found")
            return self.active_workflows[workflow_id]
            
        @app.post("/workflows/{workflow_id}/stop")
        async def stop_workflow(workflow_id: str):
            """Stop a running workflow"""
            if workflow_id not in self.active_workflows:
                raise HTTPException(status_code=404, detail="Workflow not found")
                
            # Mark workflow as stopped
            self.active_workflows[workflow_id]["status"] = "stopped"
            self.active_workflows[workflow_id]["end_time"] = datetime.now().isoformat()
            
            return {"message": f"Workflow {workflow_id} stopped"}
            
        @app.get("/alerts")
        async def get_alerts():
            """Get current system alerts"""
            return {
                "alerts": self.system_alerts,
                "total_alerts": len(self.system_alerts)
            }
            
        @app.delete("/alerts/{alert_id}")
        async def dismiss_alert(alert_id: str):
            """Dismiss a system alert"""
            self.system_alerts = [a for a in self.system_alerts if a.get("id") != alert_id]
            return {"message": f"Alert {alert_id} dismissed"}
            
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket_connection(websocket)
            
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>NanoBrain Viral Protein Analysis - Production Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5; 
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                }
                .header { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin-bottom: 20px; 
                }
                .metrics-grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin-bottom: 20px; 
                }
                .metric-card { 
                    background: white; 
                    padding: 20px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                }
                .metric-title { 
                    font-size: 18px; 
                    font-weight: bold; 
                    margin-bottom: 10px; 
                    color: #333; 
                }
                .metric-value { 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #667eea; 
                }
                .status-indicator { 
                    display: inline-block; 
                    width: 12px; 
                    height: 12px; 
                    border-radius: 50%; 
                    margin-right: 8px; 
                }
                .status-healthy { background-color: #4CAF50; }
                .status-warning { background-color: #FF9800; }
                .status-critical { background-color: #F44336; }
                .workflow-table { 
                    width: 100%; 
                    border-collapse: collapse; 
                    background: white; 
                    border-radius: 10px; 
                    overflow: hidden; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                }
                .workflow-table th, .workflow-table td { 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd; 
                }
                .workflow-table th { 
                    background-color: #f8f9fa; 
                    font-weight: bold; 
                }
                .alert-item { 
                    background: #fff3cd; 
                    border: 1px solid #ffeaa7; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin-bottom: 10px; 
                }
                .alert-critical { 
                    background: #f8d7da; 
                    border-color: #f5c6cb; 
                }
                .real-time-indicator {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border-radius: 20px;
                    font-size: 14px;
                }
                .refresh-button {
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <div class="real-time-indicator" id="connectionStatus">
                <span class="status-indicator status-healthy"></span>
                Connected
            </div>
            
            <div class="container">
                <div class="header">
                    <h1>🧬 NanoBrain Viral Protein Analysis</h1>
                    <h2>Production Dashboard</h2>
                    <p>Real-time monitoring and system status</p>
                </div>
                
                <button class="refresh-button" onclick="refreshData()">🔄 Refresh Data</button>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">System Health</div>
                        <div class="metric-value" id="systemHealth">
                            <span class="status-indicator status-healthy"></span>
                            Checking...
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">CPU Usage</div>
                        <div class="metric-value" id="cpuUsage">--%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Memory Usage</div>
                        <div class="metric-value" id="memoryUsage">--%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Cache Hit Rate</div>
                        <div class="metric-value" id="cacheHitRate">--%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Active Workflows</div>
                        <div class="metric-value" id="activeWorkflows">0</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Proteins Analyzed</div>
                        <div class="metric-value" id="proteinsAnalyzed">0</div>
                    </div>
                </div>
                
                <div class="metric-card" style="margin-bottom: 20px;">
                    <div class="metric-title">🔍 Current EEEV Analysis Status</div>
                    <div id="eeevStatus">
                        <p><strong>Boundaries Detected:</strong> <span id="boundariesDetected">0</span></p>
                        <p><strong>Literature References:</strong> <span id="literatureRefs">0</span></p>
                        <p><strong>Clustering Efficiency:</strong> <span id="clusteringEfficiency">0%</span></p>
                    </div>
                </div>
                
                <div class="metric-card" style="margin-bottom: 20px;">
                    <div class="metric-title">⚠️ System Alerts</div>
                    <div id="systemAlerts">
                        <p>No active alerts</p>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">🔄 Active Workflows</div>
                    <table class="workflow-table">
                        <thead>
                            <tr>
                                <th>Workflow ID</th>
                                <th>Status</th>
                                <th>Started</th>
                                <th>Duration</th>
                                <th>Progress</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="workflowTable">
                            <tr>
                                <td colspan="6" style="text-align: center; color: #666;">
                                    No active workflows
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <script>
                let ws = null;
                let connectionRetries = 0;
                const maxRetries = 5;
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {
                        console.log('WebSocket connected');
                        connectionRetries = 0;
                        updateConnectionStatus(true);
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    
                    ws.onclose = function() {
                        console.log('WebSocket disconnected');
                        updateConnectionStatus(false);
                        
                        if (connectionRetries < maxRetries) {
                            connectionRetries++;
                            setTimeout(connectWebSocket, 2000 * connectionRetries);
                        }
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                        updateConnectionStatus(false);
                    };
                }
                
                function updateConnectionStatus(connected) {
                    const statusElement = document.getElementById('connectionStatus');
                    if (connected) {
                        statusElement.innerHTML = '<span class="status-indicator status-healthy"></span>Connected';
                        statusElement.style.backgroundColor = '#4CAF50';
                    } else {
                        statusElement.innerHTML = '<span class="status-indicator status-critical"></span>Disconnected';
                        statusElement.style.backgroundColor = '#F44336';
                    }
                }
                
                function updateDashboard(data) {
                    if (data.type === 'metrics') {
                        updateMetrics(data.data);
                    } else if (data.type === 'workflows') {
                        updateWorkflows(data.data);
                    } else if (data.type === 'alerts') {
                        updateAlerts(data.data);
                    }
                }
                
                function updateMetrics(metrics) {
                    if (metrics.current_status) {
                        document.getElementById('cpuUsage').textContent = 
                            `${metrics.current_status.cpu_usage.toFixed(1)}%`;
                        document.getElementById('memoryUsage').textContent = 
                            `${metrics.current_status.memory_usage.toFixed(1)}%`;
                        document.getElementById('cacheHitRate').textContent = 
                            `${(metrics.current_status.cache_hit_rate * 100).toFixed(1)}%`;
                    }
                    
                    if (metrics.eeev_analysis) {
                        document.getElementById('proteinsAnalyzed').textContent = 
                            metrics.eeev_analysis.proteins_analyzed;
                        document.getElementById('boundariesDetected').textContent = 
                            metrics.eeev_analysis.boundaries_detected;
                        document.getElementById('literatureRefs').textContent = 
                            metrics.eeev_analysis.literature_references;
                        document.getElementById('clusteringEfficiency').textContent = 
                            `${(metrics.eeev_analysis.clustering_efficiency * 100).toFixed(1)}%`;
                    }
                }
                
                function updateWorkflows(workflows) {
                    const tbody = document.getElementById('workflowTable');
                    tbody.innerHTML = '';
                    
                    if (Object.keys(workflows).length === 0) {
                        tbody.innerHTML = `
                            <tr>
                                <td colspan="6" style="text-align: center; color: #666;">
                                    No active workflows
                                </td>
                            </tr>
                        `;
                    } else {
                        Object.entries(workflows).forEach(([id, workflow]) => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${id}</td>
                                <td>
                                    <span class="status-indicator status-${workflow.status === 'running' ? 'healthy' : 'warning'}"></span>
                                    ${workflow.status}
                                </td>
                                <td>${new Date(workflow.start_time).toLocaleString()}</td>
                                <td>${workflow.duration || 'Running...'}</td>
                                <td>${workflow.progress || 0}%</td>
                                <td>
                                    ${workflow.status === 'running' ? 
                                        `<button onclick="stopWorkflow('${id}')">Stop</button>` : 
                                        'Completed'}
                                </td>
                            `;
                            tbody.appendChild(row);
                        });
                    }
                    
                    document.getElementById('activeWorkflows').textContent = Object.keys(workflows).length;
                }
                
                function updateAlerts(alerts) {
                    const alertsContainer = document.getElementById('systemAlerts');
                    
                    if (alerts.length === 0) {
                        alertsContainer.innerHTML = '<p>No active alerts</p>';
                    } else {
                        alertsContainer.innerHTML = alerts.map(alert => `
                            <div class="alert-item ${alert.severity === 'critical' ? 'alert-critical' : ''}">
                                <strong>${alert.type}:</strong> ${alert.message}
                                <br>
                                <small>Time: ${new Date(alert.timestamp).toLocaleString()}</small>
                            </div>
                        `).join('');
                    }
                }
                
                async function refreshData() {
                    try {
                        const [metrics, workflows, alerts] = await Promise.all([
                            fetch('/metrics').then(r => r.json()),
                            fetch('/workflows').then(r => r.json()),
                            fetch('/alerts').then(r => r.json())
                        ]);
                        
                        updateMetrics(metrics);
                        updateWorkflows(workflows.active_workflows);
                        updateAlerts(alerts.alerts);
                        
                    } catch (error) {
                        console.error('Error refreshing data:', error);
                    }
                }
                
                async function stopWorkflow(workflowId) {
                    try {
                        await fetch(`/workflows/${workflowId}/stop`, { method: 'POST' });
                        refreshData();
                    } catch (error) {
                        console.error('Error stopping workflow:', error);
                    }
                }
                
                // Initialize
                connectWebSocket();
                refreshData();
                
                // Auto-refresh every 30 seconds
                setInterval(refreshData, 30000);
            </script>
        </body>
        </html>
        """
        
    async def _handle_websocket_connection(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection for real-time updates"""
        await websocket.accept()
        self.connected_clients.append(websocket)
        
        try:
            while True:
                # Send periodic updates
                await self._send_real_time_updates()
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
                
    async def _send_real_time_updates(self) -> None:
        """Send real-time updates to connected clients"""
        if not self.connected_clients:
            return
            
        try:
            # Get current metrics
            metrics = performance_monitor.get_performance_summary()
            
            # Prepare update data
            update_data = {
                "type": "metrics",
                "data": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all connected clients
            disconnected_clients = []
            for client in self.connected_clients:
                try:
                    await client.send_text(json.dumps(update_data))
                except Exception:
                    disconnected_clients.append(client)
                    
            # Remove disconnected clients
            for client in disconnected_clients:
                self.connected_clients.remove(client)
                
        except Exception as e:
            self.logger.error(f"Error sending real-time updates: {e}")
            
    async def _perform_health_check(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            components = {}
            alerts = []
            
            # Check performance monitor
            try:
                metrics = performance_monitor.get_performance_summary()
                components["performance_monitor"] = "healthy"
                
                # Check for violations
                violations = metrics.get("threshold_violations", [])
                if violations:
                    alerts.extend([{
                        "id": f"violation_{i}",
                        "type": "threshold_violation",
                        "message": violation,
                        "severity": "warning",
                        "timestamp": datetime.now().isoformat()
                    } for i, violation in enumerate(violations)])
                    
            except Exception as e:
                components["performance_monitor"] = "critical"
                alerts.append({
                    "id": "perf_monitor_error",
                    "type": "component_failure",
                    "message": f"Performance monitor error: {e}",
                    "severity": "critical",
                    "timestamp": datetime.now().isoformat()
                })
                
            # Determine overall status
            if any(status == "critical" for status in components.values()):
                overall_status = "critical"
            elif any(status == "warning" for status in components.values()) or alerts:
                overall_status = "warning"
            else:
                overall_status = "healthy"
                
            self.system_health = SystemHealth(
                status=overall_status,
                uptime_seconds=uptime,
                last_check=datetime.now(),
                components=components,
                alerts=alerts
            )
            
            # Store alerts
            self.system_alerts = alerts
            
            return self.system_health
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return SystemHealth(
                status="critical",
                uptime_seconds=0,
                last_check=datetime.now(),
                components={"health_check": "critical"},
                alerts=[{
                    "id": "health_check_error",
                    "type": "system_error",
                    "message": f"Health check failed: {e}",
                    "severity": "critical",
                    "timestamp": datetime.now().isoformat()
                }]
            )
            
    def register_workflow(self, workflow_id: str, workflow_info: Dict[str, Any]) -> None:
        """Register a new active workflow"""
        self.active_workflows[workflow_id] = {
            **workflow_info,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "progress": 0
        }
        self.logger.info(f"Registered workflow: {workflow_id}")
        
    def update_workflow_progress(self, workflow_id: str, progress: int, status: str = None) -> None:
        """Update workflow progress"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["progress"] = progress
            if status:
                self.active_workflows[workflow_id]["status"] = status
                
    def complete_workflow(self, workflow_id: str, success: bool = True) -> None:
        """Mark workflow as completed"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = "completed" if success else "failed"
            self.active_workflows[workflow_id]["end_time"] = datetime.now().isoformat()
            self.active_workflows[workflow_id]["progress"] = 100
            
            # Remove from active workflows after delay
            asyncio.create_task(self._cleanup_completed_workflow(workflow_id))
            
    async def _cleanup_completed_workflow(self, workflow_id: str, delay: int = 300) -> None:
        """Clean up completed workflow after delay"""
        await asyncio.sleep(delay)  # Keep for 5 minutes
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
            self.logger.info(f"Cleaned up completed workflow: {workflow_id}")
            
    async def start_dashboard(self, host: str = "0.0.0.0", port: int = 8002) -> None:
        """Start the production dashboard server"""
        self.logger.info(f"Starting production dashboard on {host}:{port}")
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        # Start the server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop"""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)


# Global dashboard instance
production_dashboard = ProductionDashboard() 