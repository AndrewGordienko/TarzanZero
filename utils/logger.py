import sys
import time
from colorama import Fore, Style, init
import threading
import webbrowser
import os
from collections import deque
import json

# For web logger
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS  # Add this import

# Initialize colorama for Windows support
init(autoreset=True)

# Define log levels
def log(level, msg):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{current_time} {level} {msg}")

def info(msg):
    log(f"{Fore.CYAN}[INFO]{Style.RESET_ALL}", msg)

def success(msg):
    log(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL}", msg)

def warning(msg):
    log(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}", msg)

def error(msg):
    log(f"{Fore.RED}[ERROR]{Style.RESET_ALL}", msg)

def debug(msg):
    log(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL}", msg)

# Optional for progress updates or steps
def step(step_msg):
    print(f"{Fore.BLUE}==> {Style.RESET_ALL}{step_msg}")

# Web logger implementation
class WebLogger:
    def __init__(self, port=5000):
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        
        # Store the templates directory path
        self.templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        
        self.data = {
            'episodes': [],
            'total_rewards': [],
            'best_rewards': [],
            'average_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'hyperparameters': {},
            'status': 'Initializing',
            'recent_episodes': deque(maxlen=5),
            'paused': False,
            'restart_requested': False,  # Initialize to False explicitly
            
            # Add trial-specific data arrays
            'trial_numbers': [],
            'trial_best_rewards': [],
            'trial_avg_rewards': [],
            'trial_best_reward_episodes': [],  # Episodes to reach best reward
            'trial_scores': [],  # Overall normalized scores
            'current_trial': 0,
            'trials_data': {},  # Dictionary to store data for each trial
            
            # Tracking for current trial's best reward episode
            'current_trial_best_reward': -float('inf'),
            'current_trial_best_reward_episode': 0
        }
        
        # Set up routes
        self.app.route('/')(self.dashboard)
        self.app.route('/data')(self.get_data)
        self.app.route('/toggle_pause', methods=['POST'])(self.toggle_pause)
        self.app.route('/restart_training', methods=['POST'])(self.restart_training)
        
        # Create templates directory
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Create the HTML template
        self._create_dashboard_template()
    
    def _create_dashboard_template(self):
        """Create the dashboard HTML template file"""
        template_path = os.path.join(self.templates_dir, 'dashboard.html')
        with open(template_path, 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-gap: 20px;
            margin-bottom: 20px;
        }
        .full-width {
            grid-column: 1 / span 2;
        }
        .box {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            display: flex;
            flex-direction: column;
            height: auto;
        }
        .box-small {
            max-height: 200px;
            overflow-y: auto;
        }
        h1, h2, h3 {
            color: #333;
            margin-top: 0;
            padding-top: 0;
        }
        h3 {
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .status {
            padding: 10px;
            background-color: #e0f7fa;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            display: block;
            overflow-y: auto;
            max-height: 300px;
        }
        th, td {
            padding: 6px;
            border: 1px solid #ddd;
            text-align: center;
        }
        th {
            background-color: #f5f5f5;
            position: sticky;
            top: 0;
        }
        .chart-container {
            position: relative;
            height: 300px;
        }
        .chart-controls {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
            gap: 10px;
        }
        .chart-controls label {
            display: flex;
            align-items: center;
            margin-right: 15px;
            cursor: pointer;
        }
        .chart-controls input {
            margin-right: 5px;
        }
        .buttons-container {
            display: flex;
            gap: 10px;
        }
        .button {
            padding: 10px 20px;
            font-size: 16px;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .button:hover {
            opacity: 0.9;
        }
        .pause-button {
            background-color: #4CAF50;
        }
        .pause-button.paused {
            background-color: #f44336;
        }
        .restart-button {
            background-color: #ff9800;
        }
        .restart-confirm {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .restart-dialog {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            max-width: 400px;
            width: 100%;
        }
        .restart-dialog h3 {
            margin-top: 0;
        }
        .restart-dialog-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 20px;
        }
        .hidden {
            display: none;
        }
        .tab-container {
            margin-bottom: 20px;
        }
        .tab-buttons {
            display: flex;
            border-bottom: 1px solid #ddd;
        }
        .tab-button {
            padding: 10px 20px;
            background-color: #f0f0f0;
            border: 1px solid #ddd;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
            cursor: pointer;
        }
        .tab-button.active {
            background-color: white;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
        }
        .tab-content {
            display: none;
            padding: 20px;
            border: 1px solid #ddd;
            border-top: none;
            background-color: white;
        }
        .tab-content.active {
            display: block;
        }
        .highlight-row {
            background-color: #f8f9d7;
            font-weight: bold;
        }
        .score-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            background-color: #3498db;
            color: white;
            font-size: 12px;
            margin-left: 10px;
        }
        .best-trial-label {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            background-color: #f39c12;
            color: white;
            font-size: 12px;
            margin-left: 10px;
        }
        .comparison-container {
            display: flex;
            gap: 20px;
        }
        .table-wrapper {
            flex: 1;
            min-width: 0;
        }
        #best-trial-hyperparameters, #trials-table {
            max-height: 300px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>RL Training Dashboard</h1>
        <div class="buttons-container">
            <button id="pauseButton" class="button pause-button">Pause Training</button>
            <button id="restartButton" class="button restart-button">Restart Training</button>
        </div>
    </div>
    
    <div class="status" id="status">Status: Initializing...</div>
    
    <div class="tab-container">
        <div class="tab-buttons">
            <div class="tab-button active" data-tab="current-trial">Current Trial</div>
            <div class="tab-button" data-tab="trials-comparison">Trials Comparison</div>
        </div>
        
        <div class="tab-content active" id="current-trial">
            <div class="container">
                <div class="box box-small">
                    <h2>Hyperparameters</h2>
                    <table id="hyperparameters">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    </table>
                </div>
                
                <div class="box box-small">
                    <h2>Recent Episodes</h2>
                    <table id="episodes">
                        <tr>
                            <th>Episode</th>
                            <th>Total Reward</th>
                            <th>Best Reward</th>
                            <th>Policy Loss</th>
                            <th>Value Loss</th>
                        </tr>
                    </table>
                </div>
            </div>
            
            <div class="box full-width">
                <h2>Rewards</h2>
                <div class="chart-controls">
                    <label><input type="checkbox" id="total-reward-toggle" checked> Total Reward</label>
                    <label><input type="checkbox" id="best-reward-toggle" checked> Best Reward</label>
                    <label><input type="checkbox" id="avg-reward-toggle" checked> Average Reward</label>
                </div>
                <div class="chart-container">
                    <canvas id="rewardsChart"></canvas>
                </div>
            </div>
            
            <div class="container">
                <div class="box">
                    <h2>Policy Loss</h2>
                    <div class="chart-container">
                        <canvas id="policyLossChart"></canvas>
                    </div>
                </div>
                
                <div class="box">
                    <h2>Value Loss</h2>
                    <div class="chart-container">
                        <canvas id="valueLossChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="tab-content" id="trials-comparison">
            <div class="box full-width">
                <h2>Trials Comparison <span id="best-trial-badge" class="score-badge">Best: Trial #0</span></h2>
                <div class="chart-controls">
                    <label><input type="checkbox" id="avg-reward-trial-toggle" checked> Average Reward</label>
                    <label><input type="checkbox" id="best-reward-trial-toggle" checked> Best Reward</label>
                    <label><input type="checkbox" id="episode-count-toggle" checked> Episodes to Best</label>
                    <label><input type="checkbox" id="score-toggle" checked> Overall Score</label>
                </div>
                <div class="chart-container">
                    <canvas id="trialsComparisonChart"></canvas>
                </div>
            </div>
            
            <div class="box full-width">
                <h2>Trials Summary & Best Configuration <span class="best-trial-label">Trial #<span id="best-trial-number">0</span></span></h2>
                <div class="comparison-container">
                    <div class="table-wrapper">
                        <h3>Trials Performance</h3>
                        <table id="trials-table">
                            <tr>
                                <th>Trial #</th>
                                <th>Avg Reward</th>
                                <th>Best Reward</th>
                                <th>Episodes to Best</th>
                                <th>Overall Score</th>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="table-wrapper">
                        <h3>Best Hyperparameters</h3>
                        <table id="best-trial-hyperparameters">
                            <tr>
                                <th>Parameter</th>
                                <th>Value</th>
                            </tr>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Restart Confirmation Dialog -->
    <div id="restartConfirm" class="restart-confirm hidden">
        <div class="restart-dialog">
            <h3>Restart Training?</h3>
            <p>This will reset all training progress. Are you sure you want to restart?</p>
            <div class="restart-dialog-buttons">
                <button id="cancelRestart" class="button">Cancel</button>
                <button id="confirmRestart" class="button restart-button">Restart</button>
            </div>
        </div>
    </div>

    <script>
        // Initialize tabs
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons and content
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                this.classList.add('active');
                document.getElementById(this.dataset.tab).classList.add('active');
            });
        });
        
        // Initialize charts
        const rewardsCtx = document.getElementById('rewardsChart').getContext('2d');
        const rewardsChart = new Chart(rewardsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Reward',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: 'Best Reward',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }, {
                    label: 'Average Reward',
                    data: [],
                    borderColor: 'rgb(255, 159, 64)',
                    borderDash: [5, 5],
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
        
        const policyLossCtx = document.getElementById('policyLossChart').getContext('2d');
        const policyLossChart = new Chart(policyLossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Policy Loss',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
        
        const valueLossCtx = document.getElementById('valueLossChart').getContext('2d');
        const valueLossChart = new Chart(valueLossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Value Loss',
                    data: [],
                    borderColor: 'rgb(153, 102, 255)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
        
        // Initialize trials comparison chart
        const trialsComparisonCtx = document.getElementById('trialsComparisonChart').getContext('2d');
        const trialsComparisonChart = new Chart(trialsComparisonCtx, {
            type: 'bar',
            data: {
                labels: [], // Trial numbers
                datasets: [{
                    label: 'Average Reward',
                    data: [], // Average rewards
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1,
                    yAxisID: 'y'
                }, {
                    label: 'Best Reward',
                    data: [], // Best rewards
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgb(255, 99, 132)',
                    borderWidth: 1,
                    yAxisID: 'y'
                }, {
                    label: 'Episodes to Best Reward',
                    data: [], // Episodes to best reward
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgb(54, 162, 235)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }, {
                    label: 'Overall Score',
                    data: [], // Overall trial scores
                    backgroundColor: 'rgba(153, 102, 255, 0.6)',
                    borderColor: 'rgb(153, 102, 255)',
                    borderWidth: 1,
                    yAxisID: 'y2'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        position: 'left',
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Reward'
                        }
                    },
                    y1: {
                        position: 'right',
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Episodes'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    },
                    y2: {
                        position: 'right',
                        beginAtZero: true,
                        min: -1,
                        max: 2,
                        title: {
                            display: true,
                            text: 'Score'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
        
        // Pause Button functionality
        const pauseButton = document.getElementById('pauseButton');
        let isPaused = false;
        
        pauseButton.addEventListener('click', function() {
            // Send request to server without changing state yet (wait for response)
            $.ajax({
                url: '/toggle_pause',
                type: 'POST',
                success: function(response) {
                    // Update button text and color based on response
                    isPaused = response.paused;
                    if (isPaused) {
                        pauseButton.textContent = 'Resume Training';
                        pauseButton.classList.add('paused');
                    } else {
                        pauseButton.textContent = 'Pause Training';
                        pauseButton.classList.remove('paused');
                    }
                },
                error: function(error) {
                    console.error('Error toggling pause state:', error);
                }
            });
        });

        // Restart functionality
        const restartButton = document.getElementById('restartButton');
        const restartConfirm = document.getElementById('restartConfirm');
        const cancelRestart = document.getElementById('cancelRestart');
        const confirmRestart = document.getElementById('confirmRestart');

        restartButton.addEventListener('click', function() {
            // Show confirmation dialog
            restartConfirm.classList.remove('hidden');
        });

        cancelRestart.addEventListener('click', function() {
            // Hide confirmation dialog
            restartConfirm.classList.add('hidden');
        });

        confirmRestart.addEventListener('click', function() {
            // Send restart request to server
            $.ajax({
                url: '/restart_training',
                type: 'POST',
                success: function(response) {
                    console.log('Training restart requested');
                    // Hide confirmation dialog
                    restartConfirm.classList.add('hidden');
                },
                error: function(error) {
                    console.error('Error restarting training:', error);
                }
            });
        });
        
        // Set up toggle controls for reward chart
        document.getElementById('total-reward-toggle').addEventListener('change', function() {
            rewardsChart.data.datasets[0].hidden = !this.checked;
            rewardsChart.update();
        });
        
        document.getElementById('best-reward-toggle').addEventListener('change', function() {
            rewardsChart.data.datasets[1].hidden = !this.checked;
            rewardsChart.update();
        });
        
        document.getElementById('avg-reward-toggle').addEventListener('change', function() {
            rewardsChart.data.datasets[2].hidden = !this.checked;
            rewardsChart.update();
        });
        
        // Set up toggle controls for trials comparison chart
        document.getElementById('avg-reward-trial-toggle').addEventListener('change', function() {
            trialsComparisonChart.data.datasets[0].hidden = !this.checked;
            trialsComparisonChart.update();
        });
        
        document.getElementById('best-reward-trial-toggle').addEventListener('change', function() {
            trialsComparisonChart.data.datasets[1].hidden = !this.checked;
            trialsComparisonChart.update();
        });
        
        document.getElementById('episode-count-toggle').addEventListener('change', function() {
            trialsComparisonChart.data.datasets[2].hidden = !this.checked;
            trialsComparisonChart.update();
        });
        
        document.getElementById('score-toggle').addEventListener('change', function() {
            trialsComparisonChart.data.datasets[3].hidden = !this.checked;
            trialsComparisonChart.update();
        });
        
        // Find the best trial by score
        function findBestTrialIndex(scores) {
            if (!scores || scores.length === 0) return -1;
            let maxScore = Math.max(...scores);
            return scores.indexOf(maxScore);
        }
        
        // Function to update the dashboard
        function updateDashboard() {
            $.ajax({
                url: '/data',
                type: 'GET',
                dataType: 'json',
                success: function(data) {
                    // Update pause button state
                    isPaused = data.paused;
                    if (isPaused) {
                        pauseButton.textContent = 'Resume Training';
                        pauseButton.classList.add('paused');
                    } else {
                        pauseButton.textContent = 'Pause Training';
                        pauseButton.classList.remove('paused');
                    }
                    
                    // Update status
                    $('#status').text('Status: ' + data.status);
                    
                    // Update hyperparameters table
                    $('#hyperparameters').empty();
                    $('#hyperparameters').append('<tr><th>Parameter</th><th>Value</th></tr>');
                    for (const [key, value] of Object.entries(data.hyperparameters)) {
                        $('#hyperparameters').append(`<tr><td>${key}</td><td>${value}</td></tr>`);
                    }
                    
                    // Update episodes table
                    $('#episodes').empty();
                    $('#episodes').append('<tr><th>Episode</th><th>Total Reward</th><th>Best Reward</th><th>Policy Loss</th><th>Value Loss</th></tr>');
                    data.recent_episodes.forEach(episode => {
                        $('#episodes').append(`<tr>
                            <td>${episode.episode}</td>
                            <td>${episode.total_reward.toFixed(2)}</td>
                            <td>${episode.best_reward.toFixed(2)}</td>
                            <td>${episode.policy_loss.toFixed(4)}</td>
                            <td>${episode.value_loss.toFixed(4)}</td>
                        </tr>`);
                    });
                    
                    // Update current trial charts
                    rewardsChart.data.labels = data.episodes;
                    rewardsChart.data.datasets[0].data = data.total_rewards;
                    rewardsChart.data.datasets[1].data = data.best_rewards;
                    rewardsChart.data.datasets[2].data = data.average_rewards;
                    rewardsChart.update();
                    
                    policyLossChart.data.labels = data.episodes;
                    policyLossChart.data.datasets[0].data = data.policy_losses;
                    policyLossChart.update();
                    
                    valueLossChart.data.labels = data.episodes;
                    valueLossChart.data.datasets[0].data = data.value_losses;
                    valueLossChart.update();
                    
                    // Update trials comparison chart
                    trialsComparisonChart.data.labels = data.trial_numbers;
                    trialsComparisonChart.data.datasets[0].data = data.trial_avg_rewards;
                    trialsComparisonChart.data.datasets[1].data = data.trial_best_rewards;
                    trialsComparisonChart.data.datasets[2].data = data.trial_best_reward_episodes;
                    trialsComparisonChart.data.datasets[3].data = data.trial_scores;
                    trialsComparisonChart.update();
                    
                    // Find best trial by score
                    const bestTrialIndex = findBestTrialIndex(data.trial_scores);
                    let bestTrialNumber = 0;
                    
                    if (bestTrialIndex >= 0) {
                        bestTrialNumber = data.trial_numbers[bestTrialIndex];
                        
                        // Update best trial badges
                        $('#best-trial-badge').text(`Best: Trial #${bestTrialNumber}`);
                        $('#best-trial-number').text(bestTrialNumber);
                        
                        // Update best trial hyperparameters
                        $('#best-trial-hyperparameters').empty();
                        $('#best-trial-hyperparameters').append('<tr><th>Parameter</th><th>Value</th></tr>');
                        
                        // Get hyperparameters of best trial
                        const bestTrialData = data.trials_data[bestTrialNumber];
                        if (bestTrialData && bestTrialData.params) {
                            for (const [key, value] of Object.entries(bestTrialData.params)) {
                                $('#best-trial-hyperparameters').append(`<tr>
                                    <td>${key}</td>
                                    <td>${typeof value === 'number' ? value.toFixed(6) : value}</td>
                                </tr>`);
                            }
                        } else {
                            $('#best-trial-hyperparameters').append('<tr><td colspan="2">No hyperparameter data available</td></tr>');
                        }
                    } else {
                        $('#best-trial-badge').text(`Best: None`);
                        $('#best-trial-number').text('None');
                        $('#best-trial-hyperparameters').empty();
                        $('#best-trial-hyperparameters').append('<tr><th>Parameter</th><th>Value</th></tr>');
                        $('#best-trial-hyperparameters').append('<tr><td colspan="2">No hyperparameter data available</td></tr>');
                    }
                    
                    // Update trials table
                    $('#trials-table').empty();
                    $('#trials-table').append('<tr><th>Trial #</th><th>Avg Reward</th><th>Best Reward</th><th>Episodes to Best</th><th>Overall Score</th></tr>');
                    for (let i = 0; i < data.trial_numbers.length; i++) {
                        const isBest = (i === bestTrialIndex);
                        const rowClass = isBest ? 'highlight-row' : '';
                        $('#trials-table').append(`<tr class="${rowClass}">
                            <td>${data.trial_numbers[i]}${isBest ? ' 🏆' : ''}</td>
                            <td>${data.trial_avg_rewards[i].toFixed(2)}</td>
                            <td>${data.trial_best_rewards[i].toFixed(2)}</td>
                            <td>${data.trial_best_reward_episodes[i]}</td>
                            <td>${data.trial_scores[i].toFixed(4)}</td>
                        </tr>`);
                    }
                },
                error: function(error) {
                    console.error('Error fetching dashboard data:', error);
                }
            });
        }
        
        // Update dashboard every second
        setInterval(updateDashboard, 1000);
        
        // Initial update
        updateDashboard();
    </script>
</body>
</html>
            ''')
    
    def dashboard(self):
        """Render the dashboard template"""
        try:
            return render_template('dashboard.html')
        except Exception as e:
            error(f"Error rendering dashboard: {e}")
            # Fallback: serve the file directly
            return send_from_directory(self.templates_dir, 'dashboard.html')
    
    def toggle_pause(self):
        """Toggle the pause state for training"""
        self.data['paused'] = not self.data['paused']
        status = "Paused" if self.data['paused'] else "Resumed"
        info(f"Training {status}")
        return jsonify({'paused': self.data['paused']})
    
    def restart_training(self):
        """Request a training restart"""
        info("Training restart requested through UI")
        # Set restart flag to true
        self.data['restart_requested'] = True
        # Reset logged data
        self.reset_data()
        return jsonify({'status': 'restart_requested'})
    
    def reset_data(self):
        """Reset all logged data to start fresh but keep trial history"""
        # Keep hyperparameters, trial history, and statuses but clear episode data
        hyperparams = self.data['hyperparameters']
        trial_numbers = self.data['trial_numbers']
        trial_best_rewards = self.data['trial_best_rewards']
        trial_avg_rewards = self.data['trial_avg_rewards']
        trial_best_reward_episodes = self.data['trial_best_reward_episodes']
        trial_scores = self.data['trial_scores']
        trials_data = self.data['trials_data']
        current_trial = self.data['current_trial']
        
        self.data = {
            'episodes': [],
            'total_rewards': [],
            'best_rewards': [],
            'average_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'hyperparameters': hyperparams,
            'status': 'Restarting',
            'recent_episodes': deque(maxlen=5),
            'paused': False,
            'restart_requested': True,
            
            # Preserve trial data
            'trial_numbers': trial_numbers,
            'trial_best_rewards': trial_best_rewards,
            'trial_avg_rewards': trial_avg_rewards,
            'trial_best_reward_episodes': trial_best_reward_episodes,
            'trial_scores': trial_scores,
            'current_trial': current_trial,
            'trials_data': trials_data,
            
            # Reset current trial tracking
            'current_trial_best_reward': -float('inf'),
            'current_trial_best_reward_episode': 0
        }
        info("Training data reset while preserving trial history")
    
    def is_restart_requested(self):
        """Check if training restart was requested"""
        return self.data.get('restart_requested', False)
    
    def clear_restart_flag(self):
        """Clear the restart flag after handling it"""
        was_restart_requested = self.data.get('restart_requested', False)
        self.data['restart_requested'] = False
        if was_restart_requested:
            info("Restart flag cleared")
        return was_restart_requested
    
    def is_paused(self):
        """Check if training is paused"""
        return self.data.get('paused', False)
    
    def get_data(self):
        """Provide data for the dashboard via API"""
        # Convert deque to list for JSON serialization
        data_copy = self.data.copy()
        data_copy['recent_episodes'] = list(self.data['recent_episodes'])
        return jsonify(data_copy)
    
    def set_hyperparameters(self, hyperparams):
        """Set the hyperparameters for display"""
        self.data['hyperparameters'] = hyperparams
    
    def update_status(self, status):
        """Update the current status"""
        self.data['status'] = status
        info(f"Status: {status}")
    
    def log_episode(self, episode, total_reward, best_reward, policy_loss, value_loss, entropy):
        """Log episode data for visualization"""
        self.data['episodes'].append(episode)
        self.data['total_rewards'].append(total_reward)
        self.data['best_rewards'].append(best_reward)
        self.data['policy_losses'].append(policy_loss)
        self.data['value_losses'].append(value_loss)
        self.data['entropies'].append(entropy)
        
        # Calculate and update the average reward
        if len(self.data['total_rewards']) > 0:
            avg_reward = sum(self.data['total_rewards'][-min(10, len(self.data['total_rewards'])):]) / min(10, len(self.data['total_rewards']))
            self.data['average_rewards'].append(avg_reward)
        else:
            self.data['average_rewards'].append(total_reward)
        
        # Add to recent episodes for table display
        self.data['recent_episodes'].append({
            'episode': episode,
            'total_reward': total_reward,
            'best_reward': best_reward,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        })
        
        # Track when we achieve a new best reward in this trial
        if total_reward > self.data['current_trial_best_reward']:
            self.data['current_trial_best_reward'] = total_reward
            self.data['current_trial_best_reward_episode'] = episode
    
    def normalize_metric(self, values):
        """Normalize a list of values to 0-1 range"""
        if not values or len(values) < 2:  # Need at least 2 values for normalization
            return values
            
        min_val = min(values)
        max_val = max(values)
        
        # If all values are the same, return 0.5 for all
        if max_val == min_val:
            return [0.5] * len(values)
            
        # Otherwise, normalize to 0-1 range
        return [(x - min_val) / (max_val - min_val) for x in values]
    
    def calculate_trial_scores(self):
        """Calculate overall scores for all trials based on normalized metrics"""
        if not self.data['trial_numbers']:
            return []
            
        # Normalize each metric across all trials
        norm_avg_rewards = self.normalize_metric(self.data['trial_avg_rewards'])
        norm_best_rewards = self.normalize_metric(self.data['trial_best_rewards'])
        norm_episodes = self.normalize_metric(self.data['trial_best_reward_episodes'])
        
        # Calculate scores: higher avg and best rewards are good, lower episodes is good
        scores = []
        for i in range(len(self.data['trial_numbers'])):
            # Score = normalized_avg_reward + normalized_best_reward - normalized_episodes_to_best
            score = norm_avg_rewards[i] + norm_best_rewards[i] - norm_episodes[i]
            scores.append(score)
            
        return scores
    
    def log_trial_result(self, trial_number, avg_reward, best_reward, best_reward_episode=None, params=None):
        """Log the results of a completed trial"""
        # If no specific best_reward_episode provided, use the tracked one
        if best_reward_episode is None:
            best_reward_episode = self.data['current_trial_best_reward_episode']
        
        # Store the trial results
        self.data['trial_numbers'].append(trial_number)
        self.data['trial_best_rewards'].append(best_reward)
        self.data['trial_avg_rewards'].append(avg_reward)
        self.data['trial_best_reward_episodes'].append(best_reward_episode)
        self.data['current_trial'] = trial_number
        
        # Calculate and update all trial scores after adding new trial
        self.data['trial_scores'] = self.calculate_trial_scores()
        
        # Store detailed trial data
        self.data['trials_data'][str(trial_number)] = {
            'avg_reward': avg_reward,
            'best_reward': best_reward,
            'best_reward_episode': best_reward_episode,
            'score': self.data['trial_scores'][-1] if self.data['trial_scores'] else 0,
            'params': params if params else {}
        }
        
        info(f"Logged trial {trial_number} with avg_reward: {avg_reward:.4f}, best_reward: {best_reward:.4f}, " +
             f"reached at episode: {best_reward_episode}, score: {self.data['trial_scores'][-1]:.4f}")
        
        # Reset tracking for next trial
        self.data['current_trial_best_reward'] = -float('inf')
        self.data['current_trial_best_reward_episode'] = 0
    
    def get_best_trial_by_score(self):
        """Get the trial with the highest overall score"""
        if not self.data['trial_scores']:
            return None
            
        best_idx = self.data['trial_scores'].index(max(self.data['trial_scores']))
        return {
            'trial_number': self.data['trial_numbers'][best_idx],
            'avg_reward': self.data['trial_avg_rewards'][best_idx],
            'best_reward': self.data['trial_best_rewards'][best_idx],
            'episodes_to_best': self.data['trial_best_reward_episodes'][best_idx],
            'score': self.data['trial_scores'][best_idx]
        }
    
    def start(self):
        """Start the Flask server in a background thread"""
        thread = threading.Thread(target=self._run_flask)
        thread.daemon = True
        thread.start()
        
        # Give the server a moment to start
        time.sleep(1)
        
        # Open the dashboard in the browser
        dashboard_url = f'http://127.0.0.1:{self.port}'
        try:
            webbrowser.open(dashboard_url)
            info(f"Dashboard opened at {dashboard_url}")
        except Exception as e:
            warning(f"Could not open browser automatically: {e}")
            
        info(f"Dashboard available at: {dashboard_url}")
        info(f"If that doesn't work, try: http://localhost:{self.port}")
        info(f"For access from other computers on your network, try http://YOUR_IP_ADDRESS:{self.port}")
        info("If you're experiencing 403 errors, try opening the dashboard in a different browser")
    
    def _run_flask(self):
        """Run the Flask application"""
        try:
            # Change host from 127.0.0.1 to 0.0.0.0 to allow external connections
            self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False, threaded=True)
        except Exception as e:
            error(f"Flask server error: {e}")

