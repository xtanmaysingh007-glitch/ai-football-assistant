#!/usr/bin/env python3
"""
AI Scouting Assistant for Small Football Clubs
Phase-1: Intelligent Player Scouting → BUILD SQUAD
Phase-2: Team Tactics & Lineup Optimization ← ANALYZE SQUAD FROM PHASE-1
Complete system with ML/DL, feature engineering, and professional GUI
WITH SQUAD TRANSFER FEATURE (Load players from Phase-1 to Phase-2)
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
import pickle
import json
from datetime import datetime
from pathlib import Path
import os

# ML & DL Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import re
from collections import Counter

warnings.filterwarnings('ignore')

# ============================================================================
# SQUAD BUILDER - NEW FEATURE FOR TRANSFERRING PLAYERS FROM PHASE-1 TO PHASE-2
# ============================================================================

class SquadBuilder:
    """Manages custom squad for transfer between Phase-1 and Phase-2"""
    
    def __init__(self):
        self.squad = []
        self.squad_df = None
    
    def add_player(self, player_dict):
        """Add player to squad"""
        player_name = player_dict.get('Player', 'Unknown')
        if any(p.get('Player') == player_name for p in self.squad):
            return f"❌ {player_name} already in squad!"
        
        self.squad.append(player_dict)
        return f"✅ Added {player_name} to squad"
    
    def remove_player(self, player_name):
        """Remove player from squad"""
        self.squad = [p for p in self.squad if p.get('Player') != player_name]
    
    def get_squad_df(self):
        """Get squad as DataFrame"""
        if not self.squad:
            return pd.DataFrame()
        self.squad_df = pd.DataFrame(self.squad)
        return self.squad_df
    
    def clear_squad(self):
        """Clear entire squad"""
        self.squad = []
        self.squad_df = None
    
    def get_squad_size(self):
        """Get number of players in squad"""
        return len(self.squad)
    
    def save_squad(self, filepath):
        """Save squad to CSV"""
        if not self.squad or len(self.squad) == 0:
            return False
        df = self.get_squad_df()
        df.to_csv(filepath, index=False)
        return True

# ============================================================================
# PART 1: DATA EXPANSION & FEATURE ENGINEERING
# ============================================================================

class DataProcessor:
    """Handles dataset loading, expansion, and feature engineering"""
    
    def __init__(self):
        self.df = None
        self.df_expanded = None
        self.scaler = StandardScaler()
        self.position_patterns = {
            'GK': ['GK'],
            'DF': ['DF', 'CB', 'LB', 'RB', 'WB'],
            'MF': ['MF', 'CM', 'CAM', 'CDM'],
            'FW': ['FW', 'ST', 'CF', 'LW', 'RW']
        }
    
    def load_dataset(self, filepath):
        """Load and clean dataset"""
        print(f"Loading dataset from {filepath}...")
        self.df = pd.read_csv(filepath)
        self.df = self.df.fillna(0)
        print(f"Loaded {len(self.df)} players with {len(self.df.columns)} columns")
        return self.df
    
    def expand_dataset(self, target_size=6000):
        """Expand dataset intelligently to target_size players"""
        print(f"Expanding dataset from {len(self.df)} to {target_size} players...")
        
        base_df = self.df.copy()
        expansion_size = target_size - len(base_df)
        
        numeric_cols = base_df.select_dtypes(include=[np.number]).columns
        positions = base_df['Pos'].unique()
        new_rows = []
        
        for _ in range(expansion_size):
            pos = np.random.choice(positions)
            pos_data = base_df[base_df['Pos'] == pos]
            
            if len(pos_data) == 0:
                continue
            
            base_player = pos_data.sample(1).iloc[0].copy()
            new_player = base_player.copy()
            new_player['Player'] = f"Synthetic_{np.random.randint(10000, 99999)}"
            
            for col in numeric_cols:
                if col not in ['Rk', 'Age', 'Born']:
                    val = new_player[col]
                    if isinstance(val, (int, float)) and val > 0:
                        noise = np.random.normal(1, 0.15)
                        new_player[col] = max(0, val * noise)
            
            new_player['Age'] = max(16, min(38, new_player['Age'] + np.random.normal(0, 2)))
            new_rows.append(new_player)
        
        expanded_df = pd.concat([base_df, pd.DataFrame(new_rows)], ignore_index=True)
        expanded_df = expanded_df.fillna(0)
        self.df_expanded = expanded_df
        print(f"Dataset expanded to {len(expanded_df)} players")
        return expanded_df
    
    def engineer_features(self):
        """Create football-meaningful features"""
        df = self.df_expanded.copy()
        
        df['Finishing_Quality'] = 0
        if 'Gls' in df.columns and 'xG' in df.columns and 'SoT' in df.columns:
            goals_norm = pd.to_numeric(df['Gls'], errors='coerce').fillna(0)
            xg_norm = pd.to_numeric(df['xG'], errors='coerce').fillna(0)
            sot_norm = pd.to_numeric(df['SoT'], errors='coerce').fillna(0)
            
            goal_efficiency = np.where(xg_norm > 0, goals_norm / (xg_norm + 0.1), 0)
            shot_accuracy = np.where(sot_norm > 0, sot_norm / (pd.to_numeric(df['Sh'], errors='coerce').fillna(1) + 0.1), 0)
            
            df['Finishing_Quality'] = (goal_efficiency * 50 + shot_accuracy * 50)
        
        df['Creativity'] = 0
        if 'Ast' in df.columns and 'xAG' in df.columns:
            ast = pd.to_numeric(df['Ast'], errors='coerce').fillna(0)
            xag = pd.to_numeric(df['xAG'], errors='coerce').fillna(0)
            
            ast_norm = (ast / (ast.max() + 1)) * 100
            xag_norm = (xag / (xag.max() + 1)) * 100
            
            df['Creativity'] = (ast_norm * 0.6 + xag_norm * 0.4)
        
        df['Goal_Threat'] = 0
        if 'xG' in df.columns:
            xg = pd.to_numeric(df['xG'], errors='coerce').fillna(0)
            df['Goal_Threat'] = (xg / (xg.max() + 1)) * 100
        
        df['Defensive_Actions'] = 0
        if 'Tkl' in df.columns and 'Int' in df.columns:
            tkl = pd.to_numeric(df['Tkl'], errors='coerce').fillna(0)
            inter = pd.to_numeric(df['Int'], errors='coerce').fillna(0)
            
            tkl_norm = (tkl / (tkl.max() + 1)) * 100
            inter_norm = (inter / (inter.max() + 1)) * 100
            
            df['Defensive_Actions'] = (tkl_norm * 0.7 + inter_norm * 0.3)
        
        df['Discipline'] = 100
        if 'CrdY' in df.columns and 'CrdR' in df.columns:
            crdy = pd.to_numeric(df['CrdY'], errors='coerce').fillna(0)
            crdr = pd.to_numeric(df['CrdR'], errors='coerce').fillna(0)
            
            discipline_score = 100 - ((crdy * 5 + crdr * 25) / 10)
            df['Discipline'] = np.clip(discipline_score, 0, 100)
        
        df['Involvement'] = 0
        if 'Touches' in df.columns:
            touches = pd.to_numeric(df['Touches'], errors='coerce').fillna(0)
            df['Involvement'] = (touches / (touches.max() + 1)) * 100
        
        df['Pace'] = 50
        if 'Sh/90' in df.columns:
            sh90 = pd.to_numeric(df['Sh/90'], errors='coerce').fillna(0)
            df['Pace'] = (sh90 / (sh90.max() + 0.1)) * 100
        
        df['Market_Value'] = 1
        
        engineered_features = ['Finishing_Quality', 'Creativity', 'Goal_Threat', 
                              'Defensive_Actions', 'Discipline', 'Involvement', 'Pace']
        
        for feat in engineered_features:
            df[feat] = np.clip(df[feat], 0, 100)
        
        self.df_expanded = df
        print(f"Feature engineering complete")
        return df
    
    def get_numeric_features(self):
        """Get all numeric columns for ML"""
        numeric_cols = self.df_expanded.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
    def parse_position(self, pos_str):
        """Parse position string"""
        if pd.isna(pos_str) or pos_str == '':
            return 'MF'
        
        pos_str = str(pos_str).upper()
        
        for category, patterns in self.position_patterns.items():
            for pattern in patterns:
                if pattern in pos_str:
                    return category
        
        return 'MF'

# ============================================================================
# PART 2: ML & DL MODELS
# ============================================================================

class AIModels:
    """Hybrid ML + DL approach for player evaluation"""
    
    def __init__(self, numeric_features):
        self.numeric_features = numeric_features
        self.rf_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def prepare_data(self, df):
        """Prepare data for training"""
        X = df[self.numeric_features].fillna(0).values
        X = self.scaler.fit_transform(X)
        
        age_norm = (df['Age'].fillna(25).values - 16) / 22
        involvement = df['Involvement'].fillna(50).values / 100
        threat = df['Goal_Threat'].fillna(50).values / 100
        
        y = (age_norm * 0.3 + involvement * 0.35 + threat * 0.35) * 100
        y = np.clip(y, 1, 100)
        
        return X, y
    
    def train_random_forest(self, X, y):
        """Train Random Forest"""
        print("Training Random Forest...")
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)
        print(f"Random Forest trained with R² = {self.rf_model.score(X, y):.4f}")
        return self.rf_model
    
    def train_neural_network(self, X, y):
        """Train Neural Network"""
        print("Training Neural Network...")
        
        self.nn_model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=X.shape[1]),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.nn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.nn_model.fit(X, y/100, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        print("Neural Network training complete")
        return self.nn_model
    
    def predict_ensemble(self, X):
        """Ensemble prediction"""
        rf_pred = self.rf_model.predict(X) if self.rf_model else np.zeros(len(X))
        nn_pred = (self.nn_model.predict(X, verbose=0).flatten() * 100) if self.nn_model else np.zeros(len(X))
        
        ensemble = (rf_pred * 0.6 + nn_pred * 0.4)
        return np.clip(ensemble, 0, 100)

# ============================================================================
# PART 3: INTELLIGENT SEARCH & RANKING
# ============================================================================

class IntelligentSearchEngine:
    """NLP-based player search with intent detection"""
    
    def __init__(self, df, processor, ai_models):
        self.df = df
        self.processor = processor
        self.ai_models = ai_models
        
        self.role_keywords = {
            'striker': ['striker', 'forward', 'fw', 'cf', 'st'],
            'midfielder': ['midfielder', 'mf', 'cm'],
            'defender': ['defender', 'df', 'cb'],
            'goalkeeper': ['goalkeeper', 'gk', 'keeper']
        }
        
        self.attribute_keywords = {
            'young': ['young', 'youth'],
            'fast': ['fast', 'pace', 'quick'],
            'creative': ['creative', 'creativity', 'assists'],
            'prolific': ['prolific', 'scorer', 'goals']
        }
    
    def detect_intent(self, query):
        """Detect role and attributes"""
        query_lower = query.lower()
        
        detected_role = None
        detected_attrs = set()
        
        for role, keywords in self.role_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_role = role
                break
        
        for attr, keywords in self.attribute_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_attrs.add(attr)
        
        return detected_role, detected_attrs
    
    def search(self, query, limit=50):
        """Search players"""
        role, attrs = self.detect_intent(query)
        candidates = self.df.copy()
        
        if role:
            role_map = {
                'striker': ['FW'],
                'midfielder': ['MF'],
                'defender': ['DF'],
                'goalkeeper': ['GK']
            }
            positions = role_map.get(role, [])
            candidates = candidates[candidates['Position_Category'].isin(positions)]
        
        results = []
        
        for idx, player in candidates.iterrows():
            score = 50
            explanations = []
            
            if 'young' in attrs:
                if player['Age'] < 23:
                    score += 20
                    explanations.append(f"Young (Age: {player['Age']:.0f})")
            
            if 'fast' in attrs:
                if player['Pace'] > 70:
                    score += 15
                    explanations.append(f"Fast (Pace: {player['Pace']:.0f})")
            
            if 'creative' in attrs:
                if player['Creativity'] > 70:
                    score += 20
                    explanations.append(f"Creative ({player['Creativity']:.0f})")
            
            if 'prolific' in attrs:
                if player['Finishing_Quality'] > 70:
                    score += 20
                    explanations.append(f"Prolific ({player['Finishing_Quality']:.0f})")
            
            score = (score + player['AI_Score']) / 2
            
            results.append({
                'player': player['Player'],
                'position': player['Pos'],
                'age': player['Age'],
                'team': player['Squad'],
                'score': score,
                'market_value': player['Market_Value'],
                'explanations': explanations,
                'creativity': player['Creativity'],
                'finishing': player['Finishing_Quality'],
                'full_row': player  # Store full player data
            })
        
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:limit]
        return results

# ============================================================================
# PART 4: TEAM TACTICS (PHASE-2)
# ============================================================================

class TeamTacticsAnalyzer:
    """Team analysis"""
    
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
    
    def analyze_team(self, squad_df):
        """Analyze squad"""
        
        gls = pd.to_numeric(squad_df.get('Gls', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        xg = pd.to_numeric(squad_df.get('xG', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        sot = pd.to_numeric(squad_df.get('SoT', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        sh = pd.to_numeric(squad_df.get('Sh', pd.Series([1]*len(squad_df))), errors='coerce').fillna(1)
        
        goal_efficiency = np.where(xg > 0, gls / (xg + 0.1), 0)
        shot_accuracy = np.where(sot > 0, sot / (sh + 0.1), 0)
        finishing = (goal_efficiency * 50 + shot_accuracy * 50)
        finishing_score = np.clip(finishing.mean(), 0, 100)
        
        ast = pd.to_numeric(squad_df.get('Ast', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        xag = pd.to_numeric(squad_df.get('xAG', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        
        ast_norm = (ast / (ast.max() + 1)) * 100 if ast.max() > 0 else pd.Series([0] * len(squad_df))
        xag_norm = (xag / (xag.max() + 1)) * 100 if xag.max() > 0 else pd.Series([0] * len(squad_df))
        
        creativity = (ast_norm * 0.6 + xag_norm * 0.4)
        creativity_score = np.clip(creativity.mean(), 0, 100)
        
        tkl = pd.to_numeric(squad_df.get('Tkl', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        inter = pd.to_numeric(squad_df.get('Int', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        
        tkl_norm = (tkl / (tkl.max() + 1)) * 100 if tkl.max() > 0 else pd.Series([0] * len(squad_df))
        inter_norm = (inter / (inter.max() + 1)) * 100 if inter.max() > 0 else pd.Series([0] * len(squad_df))
        
        defense = (tkl_norm * 0.7 + inter_norm * 0.3)
        defense_score = np.clip(defense.mean(), 0, 100)
        
        crdy = pd.to_numeric(squad_df.get('CrdY', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        crdr = pd.to_numeric(squad_df.get('CrdR', pd.Series([0]*len(squad_df))), errors='coerce').fillna(0)
        
        discipline_score = 100 - ((crdy.mean() * 5 + crdr.mean() * 25) / 10)
        discipline_score = np.clip(discipline_score, 0, 100)
        
        overall = (finishing_score * 0.4 + creativity_score * 0.3 + defense_score * 0.3)
        
        return {
            'attack_score': finishing_score,
            'creativity_score': creativity_score,
            'defense_score': defense_score,
            'discipline_score': discipline_score,
            'overall_strength': overall
        }
    
    def detect_weaknesses(self, squad_df):
        """Detect weaknesses"""
        analysis = self.analyze_team(squad_df)
        weaknesses = []
        
        if analysis['attack_score'] < 50:
            weaknesses.append({'area': 'Attack', 'severity': 'High', 'description': f'Poor finishing ({analysis["attack_score"]:.0f}/100)'})
        if analysis['creativity_score'] < 50:
            weaknesses.append({'area': 'Creativity', 'severity': 'High', 'description': f'Limited creation ({analysis["creativity_score"]:.0f}/100)'})
        if analysis['defense_score'] < 50:
            weaknesses.append({'area': 'Defense', 'severity': 'High', 'description': f'Weak defense ({analysis["defense_score"]:.0f}/100)'})
        if analysis['discipline_score'] < 70:
            weaknesses.append({'area': 'Discipline', 'severity': 'Medium', 'description': f'Too many cards ({analysis["discipline_score"]:.0f}/100)'})
        
        if not weaknesses:
            weaknesses.append({'area': 'Overall', 'severity': 'Low', 'description': f'Well-balanced squad!'})
        
        return weaknesses
    
    def recommend_formation(self, squad_df):
        """Recommend formation"""
        analysis = self.analyze_team(squad_df)
        
        formations = {
            '4-3-3': {'description': 'Balanced', 'attack': 0.4, 'mid': 0.3, 'def': 0.3},
            '4-2-3-1': {'description': 'Defensive', 'attack': 0.35, 'mid': 0.25, 'def': 0.4},
            '3-5-2': {'description': 'Attacking', 'attack': 0.45, 'mid': 0.35, 'def': 0.2}
        }
        
        scores = {}
        for fmt, w in formations.items():
            score = (analysis['attack_score'] * w['attack'] + analysis['creativity_score'] * w['mid'] + analysis['defense_score'] * w['def'])
            scores[fmt] = score
        
        best = max(scores, key=scores.get)
        return {'recommended': best, 'description': formations[best]['description'], 'all_scores': scores}

# ============================================================================
# PART 5: TKINTER GUI WITH SQUAD TRANSFER
# ============================================================================

class ScoutingAssistantGUI:
    """GUI with Squad Transfer Feature (NEW)"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("⚽ AI Scouting Assistant - Phase 1 & 2 WITH SQUAD TRANSFER")
        self.root.geometry("1600x950")
        
        # Initialize
        self.processor = None
        self.ai_models = None
        self.search_engine = None
        self.tactics_analyzer = None
        self.current_squad = None
        self.squad_builder = SquadBuilder()  # NEW: Squad builder
        self.search_results = []  # Store search results
        
        self.create_main_layout()
    
    def create_main_layout(self):
        """Create main layout"""
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.phase1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.phase1_frame, text="Phase-1: Player Scouting")
        self.create_phase1_tab()
        
        self.phase2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.phase2_frame, text="Phase-2: Team Tactics")
        self.create_phase2_tab()
        
        self.status_var = tk.StringVar(value="Ready. Load dataset to begin.")
        ttk.Label(self.root, textvariable=self.status_var, foreground='green').pack(fill=tk.X, padx=5, pady=2)
    
    def create_phase1_tab(self):
        """Create Phase-1 with squad transfer"""
        
        # Control panel
        control_frame = ttk.LabelFrame(self.phase1_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Train Models", command=self.train_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="➕ ADD TO SQUAD (NEW)", command=self.add_to_squad_from_phase1).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📊 View Squad", command=self.view_squad).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🗑️ Clear Squad", command=self.clear_squad_from_phase1).pack(side=tk.LEFT, padx=5)
        
        # Search panel
        search_frame = ttk.LabelFrame(self.phase1_frame, text="Intelligent Search", padding=10)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="Enter query (e.g., 'young fast striker'):").pack(fill=tk.X)
        
        self.search_var = tk.StringVar()
        search_input = ttk.Entry(search_frame, textvariable=self.search_var, width=60)
        search_input.pack(fill=tk.X, pady=5)
        search_input.bind('<Return>', lambda e: self.perform_search())
        ttk.Button(search_frame, text="Search", command=self.perform_search).pack(side=tk.LEFT, padx=5)
        
        # Results panel
        results_frame = ttk.LabelFrame(self.phase1_frame, text="Search Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_tree = ttk.Treeview(results_frame, columns=(
            'Player', 'Pos', 'Age', 'Team', 'Score'
        ), height=20)
        
        for col in ['Player', 'Pos', 'Age', 'Team', 'Score']:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_phase2_tab(self):
        """Create Phase-2 for squad analysis"""
        
        control_frame = ttk.LabelFrame(self.phase2_frame, text="Squad Analysis", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load CSV Squad", command=self.load_squad_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📥 LOAD FROM PHASE-1 (NEW)", command=self.load_squad_from_phase1).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Analyze Team", command=self.analyze_team).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Get Best XI", command=self.get_best_xi).pack(side=tk.LEFT, padx=5)
        
        results_frame = ttk.LabelFrame(self.phase2_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tactics_text = scrolledtext.ScrolledText(results_frame, height=30, wrap=tk.WORD)
        self.tactics_text.pack(fill=tk.BOTH, expand=True)
    
    # ========================================================================
    # PHASE-1 METHODS
    # ========================================================================
    
    def load_dataset(self):
        """Load dataset"""
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        if not filepath:
            return
        
        try:
            self.status_var.set("Loading dataset...")
            self.root.update()
            
            self.processor = DataProcessor()
            self.processor.load_dataset(filepath)
            self.processor.expand_dataset(target_size=6000)
            self.processor.engineer_features()
            self.processor.df_expanded['Position_Category'] = self.processor.df_expanded['Pos'].apply(self.processor.parse_position)
            
            numeric_features = self.processor.get_numeric_features()
            self.ai_models = AIModels(numeric_features)
            X, y = self.ai_models.prepare_data(self.processor.df_expanded)
            
            self.ai_models.train_random_forest(X, y)
            self.ai_models.train_neural_network(X, y)
            
            predictions = self.ai_models.predict_ensemble(X)
            self.processor.df_expanded['AI_Score'] = predictions
            self.processor.df_expanded['Market_Value'] = predictions
            
            self.search_engine = IntelligentSearchEngine(self.processor.df_expanded, self.processor, self.ai_models)
            self.tactics_analyzer = TeamTacticsAnalyzer(self.processor.df_expanded, self.processor)
            
            self.status_var.set(f"✓ Ready: {len(self.processor.df_expanded)} players loaded")
            messagebox.showinfo("Success", f"Loaded {len(self.processor.df_expanded)} players!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def train_models(self):
        """Train models"""
        if self.processor is None:
            messagebox.showwarning("Warning", "Load dataset first")
            return
        
        try:
            numeric_features = self.processor.get_numeric_features()
            self.ai_models = AIModels(numeric_features)
            X, y = self.ai_models.prepare_data(self.processor.df_expanded)
            self.ai_models.train_random_forest(X, y)
            self.ai_models.train_neural_network(X, y)
            
            predictions = self.ai_models.predict_ensemble(X)
            self.processor.df_expanded['AI_Score'] = predictions
            self.processor.df_expanded['Market_Value'] = predictions
            
            self.search_engine = IntelligentSearchEngine(self.processor.df_expanded, self.processor, self.ai_models)
            
            self.status_var.set("✓ Models trained")
            messagebox.showinfo("Success", "Models trained!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def perform_search(self):
        """Search players"""
        if self.search_engine is None:
            messagebox.showwarning("Warning", "Load dataset first")
            return
        
        query = self.search_var.get()
        if not query:
            return
        
        try:
            results = self.search_engine.search(query, limit=50)
            self.search_results = results
            
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            for i, r in enumerate(results, 1):
                self.results_tree.insert('', 'end', text=str(i), values=(
                    r['player'], r['position'], f"{r['age']:.0f}", r['team'], f"{r['score']:.1f}"
                ))
            
            self.status_var.set(f"Found {len(results)} players")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def add_to_squad_from_phase1(self):
        """Add selected player to squad (NEW FEATURE)"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select a player first!")
            return
        
        item = selection[0]
        values = self.results_tree.item(item)['values']
        player_name = values[0]
        
        # Find player in search results
        player_dict = next((p['full_row'].to_dict() for p in self.search_results if p['player'] == player_name), None)
        
        if player_dict:
            msg = self.squad_builder.add_player(player_dict)
            messagebox.showinfo("Squad", msg)
            self.status_var.set(f"Squad: {self.squad_builder.get_squad_size()} players")
    
    def view_squad(self):
        """View current squad"""
        squad_df = self.squad_builder.get_squad_df()
        
        if squad_df.empty:
            messagebox.showwarning("Warning", "Squad is empty!")
            return
        
        info = f"SQUAD ({len(squad_df)} players):\n\n"
        for idx, (_, player) in enumerate(squad_df.iterrows(), 1):
            info += f"{idx}. {player.get('Player', 'N/A')} ({player.get('Pos', 'N/A')}) - {player.get('Squad', 'N/A')}\n"
        
        messagebox.showinfo("Squad", info)
    
    def clear_squad_from_phase1(self):
        """Clear squad"""
        if messagebox.askyesno("Confirm", "Clear squad?"):
            self.squad_builder.clear_squad()
            self.status_var.set("Squad cleared")
    
    # ========================================================================
    # PHASE-2 METHODS
    # ========================================================================
    
    def load_squad_from_phase1(self):
        """Load squad built in Phase-1 (NEW FEATURE)"""
        squad_df = self.squad_builder.get_squad_df()
        
        if squad_df.empty:
            messagebox.showwarning("Warning", "Build a squad in Phase-1 first!")
            return
        
        self.current_squad = squad_df
        messagebox.showinfo("Success", f"Loaded squad with {len(squad_df)} players from Phase-1!\n\nClick 'Analyze Team' to analyze")
    
    def load_squad_csv(self):
        """Load squad from CSV"""
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        if not filepath:
            return
        
        try:
            squad_df = pd.read_csv(filepath)
            squad_df = squad_df.fillna(0)
            self.current_squad = squad_df
            messagebox.showinfo("Success", f"Loaded {len(squad_df)} players")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def analyze_team(self):
        """Analyze team"""
        if self.current_squad is None or len(self.current_squad) == 0:
            messagebox.showwarning("Warning", "Load a squad first")
            return
        
        try:
            for col in ['Finishing_Quality', 'Creativity', 'Defensive_Actions', 'Discipline']:
                if col not in self.current_squad.columns:
                    self.current_squad[col] = 50
            
            analysis = self.tactics_analyzer.analyze_team(self.current_squad)
            weaknesses = self.tactics_analyzer.detect_weaknesses(self.current_squad)
            formation = self.tactics_analyzer.recommend_formation(self.current_squad)
            
            self.tactics_text.delete('1.0', tk.END)
            
            output = f"""
════════════════════════════════════════════════════════════
                    TEAM ANALYSIS (LOADED FROM PHASE-1)
════════════════════════════════════════════════════════════

Squad Size: {len(self.current_squad)} players

TEAM STRENGTH ASSESSMENT:
  Attack:      {analysis['attack_score']:.1f}/100
  Creativity:  {analysis['creativity_score']:.1f}/100
  Defense:     {analysis['defense_score']:.1f}/100
  Discipline:  {analysis['discipline_score']:.1f}/100

OVERALL:     {analysis['overall_strength']:.1f}/100

WEAKNESSES:
"""
            
            for w in weaknesses:
                output += f"\n  🔴 {w['area']}: {w['description']}"
            
            output += f"""

FORMATION RECOMMENDATION: {formation['recommended']}
  ({formation['description']})

════════════════════════════════════════════════════════════
"""
            
            self.tactics_text.insert('1.0', output)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def get_best_xi(self):
        """Get best XI"""
        if self.current_squad is None:
            messagebox.showwarning("Warning", "Load squad first")
            return
        
        try:
            self.tactics_text.delete('1.0', tk.END)
            self.tactics_text.insert('1.0', "Best XI functionality - coming soon!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ScoutingAssistantGUI(root)
    root.mainloop()

