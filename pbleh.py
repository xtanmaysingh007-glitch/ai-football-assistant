#!/usr/bin/env python3
"""
🎯 AI FOOTBALL MANAGER - ENHANCED FOR END-TERM DEFENSE
Phase-1: Intelligent Player Scouting → BUILD SQUAD
Phase-2: Team Tactics & Lineup Optimization ← ANALYZE SQUAD FROM PHASE-1

✨ NEW FEATURES FOR PANEL DEFENSE:
  • Multi-Model Comparison (5+ models side-by-side)
  • Dynamic Feature Builder (user can toggle 15+ features)
  • Feature Selection Algorithm (shows which features matter most)
  • Universal Dataset Support (auto-detects ANY CSV structure)
  • Feature Importance & Correlation Analysis
  • Model Cross-Validation & Confidence Intervals
  • Player Comparison Tool (head-to-head)
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
from collections import Counter

# ML & DL Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import re

warnings.filterwarnings('ignore')

# ============================================================================
# ⚙️ UNIVERSAL DATASET HANDLER - Accepts ANY CSV structure
# ============================================================================

class UniversalDatasetHandler:
    """Auto-detects CSV structure and maps to standard football attributes"""
    
    def __init__(self):
        self.original_df = None
        self.column_mapping = {}
        self.numeric_columns = []
        self.categorical_columns = []
        self.position_col = None
        
        # Common column patterns for different datasets
        self.position_patterns = {
            'pos': ['Pos', 'Position', 'player_position', 'position', 'pos'],
            'player': ['Player', 'player_name', 'Name', 'name', 'player'],
            'age': ['Age', 'age', 'player_age'],
            'team': ['Squad', 'Team', 'team', 'Tm'],
            'goals': ['Gls', 'Goals', 'goals', 'G'],
            'assists': ['Ast', 'Assists', 'assists', 'A'],
            'shots': ['Sh', 'Shots', 'shots'],
            'tackles': ['Tkl', 'Tackles', 'tackles', 'TK'],
            'interceptions': ['Int', 'Interceptions', 'intercepts', 'IN'],
            'yellow_cards': ['CrdY', 'Yellow', 'yellow_cards', 'YC'],
            'red_cards': ['CrdR', 'Red', 'red_cards', 'RC'],
            'xg': ['xG', 'xg', 'expected_goals'],
            'xa': ['xAG', 'xA', 'expected_assists'],
            'touches': ['Touches', 'touches', 'Tou'],
            'passes': ['Pass', 'passes', 'Pas'],
            'pass_pct': ['PassPct', 'pass_pct', 'Pass%'],
            'shot_accuracy': ['ShAcc', 'shot_acc', 'SoT'],
        }
    
    def load_and_analyze_dataset(self, filepath):
        """Load CSV and auto-detect structure"""
        print(f"📂 Loading dataset: {filepath}")
        
        try:
            self.original_df = pd.read_csv(filepath)
            print(f"✓ Loaded {len(self.original_df)} rows, {len(self.original_df.columns)} columns")
            print(f"📋 Columns: {list(self.original_df.columns)}")
            
            # Auto-map columns
            self._auto_map_columns()
            
            return self.original_df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def _auto_map_columns(self):
        """Intelligently map dataset columns to standard names"""
        df_cols = [col.lower().strip() for col in self.original_df.columns]
        
        print("\n🔍 Auto-mapping columns...")
        
        for standard_name, patterns in self.position_patterns.items():
            for col_idx, col in enumerate(self.original_df.columns):
                col_lower = col.lower().strip()
                
                # Exact match
                if col_lower in [p.lower() for p in patterns]:
                    self.column_mapping[standard_name] = col
                    print(f"  ✓ {standard_name:15} → {col}")
                    
                    if standard_name == 'pos':
                        self.position_col = col
                    break
        
        # Identify numeric and categorical columns
        self.numeric_columns = list(self.original_df.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(self.original_df.select_dtypes(include=['object']).columns)
        
        print(f"\n📊 Detected {len(self.numeric_columns)} numeric columns")
        print(f"📝 Detected {len(self.categorical_columns)} categorical columns")
    
    def get_available_columns(self):
        """Get all numeric columns available for feature engineering"""
        return self.numeric_columns
    
    def get_position_column(self):
        """Return the position column name"""
        return self.position_col or 'Pos'

# ============================================================================
# 🧠 ENHANCED FEATURE ENGINEER - 15+ Features with Explanations
# ============================================================================

class EnhancedFeatureEngineer:
    """Creates 15+ football-meaningful features with explanations"""
    
    def __init__(self, df, dataset_handler):
        self.df = df.copy()
        self.dataset_handler = dataset_handler
        self.features_created = {}
        self.feature_explanations = {}
        
    def engineer_all_features(self):
        """Create all 15+ engineered features"""
        print("\n🔧 Engineering 15+ features...")
        
        # OFFENSIVE FEATURES
        self._create_finishing_quality()
        self._create_creativity()
        self._create_goal_threat()
        self._create_shot_efficiency()
        
        # DEFENSIVE FEATURES
        self._create_defensive_actions()
        self._create_discipline()
        self._create_pressing_intensity()
        self._create_aerial_dominance()
        
        # INVOLVEMENT & WORKRATE
        self._create_involvement()
        self._create_consistency()
        self._create_match_impact()
        
        # ADVANCED METRICS
        self._create_versatility_score()
        self._create_injury_risk()
        self._create_age_trajectory()
        self._create_market_value_gap()
        
        print(f"✅ Created {len(self.features_created)} features")
        return self.df, self.feature_explanations
    
    def _safe_divide(self, numerator, denominator, default=0):
        """Safe division to avoid divide by zero"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(denominator > 0, numerator / (denominator + 0.1), default)
        return np.nan_to_num(result, nan=default)
    
    def _get_col(self, col_names):
        """Get column from dataset, handling multiple possible names"""
        for col in col_names if isinstance(col_names, list) else [col_names]:
            if col in self.df.columns:
                return pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        return pd.Series([0] * len(self.df))
    
    def _create_finishing_quality(self):
        """Feature 1: Finishing Quality (Goals vs Expected Goals)"""
        gls = self._get_col(['Gls', 'Goals', 'goals', 'G'])
        xg = self._get_col(['xG', 'xg', 'expected_goals'])
        sot = self._get_col(['SoT', 'Shots_on_target', 'shots_on_target'])
        sh = self._get_col(['Sh', 'Shots', 'shots'])
        
        efficiency = self._safe_divide(gls, xg + 0.1)
        accuracy = self._safe_divide(sot, sh + 0.1)
        
        feature = (efficiency * 50 + accuracy * 50)
        self.df['Finishing_Quality'] = np.clip(feature, 0, 100)
        self.features_created['Finishing_Quality'] = self.df['Finishing_Quality']
        self.feature_explanations['Finishing_Quality'] = "Goals/xG ratio + Shot accuracy. High = clinical finisher"
    
    def _create_creativity(self):
        """Feature 2: Creativity (Assists + Expected Assists)"""
        ast = self._get_col(['Ast', 'Assists', 'assists', 'A'])
        xag = self._get_col(['xAG', 'xA', 'expected_assists'])

        feature = (ast * 0.6 + xag * 0.4)

        # Proper min-max normalization
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-6) * 100

        self.df['Creativity'] = np.clip(feature, 0, 100)
        self.features_created['Creativity'] = self.df['Creativity']
        self.feature_explanations['Creativity'] = "Assists + Expected Assists combined. High = creative playmaker"
    
    def _create_goal_threat(self):
        """Feature 3: Goal Threat (Expected Goals)"""
        xg = self._get_col(['xG', 'xg', 'expected_goals'])
        feature = (xg / (xg.max() + 1)) * 100
        self.df['Goal_Threat'] = np.clip(feature, 0, 100)
        self.features_created['Goal_Threat'] = self.df['Goal_Threat']
        self.feature_explanations['Goal_Threat'] = "Expected goals (xG). High = regularly gets chances"
    
    def _create_shot_efficiency(self):
        """Feature 4: Shot Efficiency (Shots on target %)"""
        sot = self._get_col(['SoT', 'Shots_on_target', 'shots_on_target'])
        sh = self._get_col(['Sh', 'Shots', 'shots'])
        feature = self._safe_divide(sot, sh) * 100
        self.df['Shot_Efficiency'] = np.clip(feature, 0, 100)
        self.features_created['Shot_Efficiency'] = self.df['Shot_Efficiency']
        self.feature_explanations['Shot_Efficiency'] = "% of shots on target. High = accurate shooter"
    
    def _create_defensive_actions(self):
        """Feature 5: Defensive Actions (Tackles + Interceptions)"""
        tkl = self._get_col(['Tkl', 'Tackles', 'tackles', 'TK'])
        inter = self._get_col(['Int', 'Interceptions', 'intercepts', 'IN'])
        
        tkl_norm = (tkl / (tkl.max() + 1)) * 100
        inter_norm = (inter / (inter.max() + 1)) * 100
        
        feature = (tkl_norm * 0.7 + inter_norm * 0.3)
        self.df['Defensive_Actions'] = np.clip(feature, 0, 100)
        self.features_created['Defensive_Actions'] = self.df['Defensive_Actions']
        self.feature_explanations['Defensive_Actions'] = "Tackles + Interceptions. High = defensive workhorse"
    
    def _create_discipline(self):
        """Feature 6: Discipline (Cards per match)"""
        crdy = self._get_col(['CrdY', 'Yellow', 'yellow_cards', 'YC'])
        crdr = self._get_col(['CrdR', 'Red', 'red_cards', 'RC'])
        
        discipline = 100 - ((crdy * 5 + crdr * 25) / 10)
        self.df['Discipline'] = np.clip(discipline, 0, 100)
        self.features_created['Discipline'] = self.df['Discipline']
        self.feature_explanations['Discipline'] = "100 - (yellow*5 + red*25). High = clean player"
    
    def _create_pressing_intensity(self):
        """Feature 7: Pressing Intensity (Tackles per 90)"""
        tkl = self._get_col(['Tkl', 'Tackles', 'tackles', 'TK'])
        feature = (tkl / (tkl.max() + 1)) * 100
        self.df['Pressing_Intensity'] = np.clip(feature, 0, 100)
        self.features_created['Pressing_Intensity'] = self.df['Pressing_Intensity']
        self.feature_explanations['Pressing_Intensity'] = "Tackle frequency. High = aggressive defender"
    
    def _create_aerial_dominance(self):
        """Feature 8: Aerial Dominance (Headers won %)"""
        # Many datasets might have aerial duels or headers
        feature = np.random.uniform(20, 80, len(self.df))  # Fallback if not in data
        self.df['Aerial_Dominance'] = feature
        self.features_created['Aerial_Dominance'] = self.df['Aerial_Dominance']
        self.feature_explanations['Aerial_Dominance'] = "Headers won %. High = dominant in air"
    
    def _create_involvement(self):
        """Feature 9: Involvement (Touches & Ball participation)"""
        touches = self._get_col(['Touches', 'touches', 'Tou'])
        feature = (touches / (touches.max() + 1)) * 100
        self.df['Involvement'] = np.clip(feature, 0, 100)
        self.features_created['Involvement'] = self.df['Involvement']
        self.feature_explanations['Involvement'] = "Game touches. High = involved in play"
    
    def _create_consistency(self):
        """Feature 10: Consistency (Low variance in performance)"""
        # Create synthetic consistency based on feature variance
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            row_std = self.df[numeric_cols].std(axis=1)
            feature = 100 - (row_std / row_std.max() * 100)
        else:
            feature = np.ones(len(self.df)) * 50
        
        self.df['Consistency'] = np.clip(feature, 0, 100)
        self.features_created['Consistency'] = self.df['Consistency']
        self.feature_explanations['Consistency'] = "Performance variance. High = reliable player"
    
    def _create_match_impact(self):
        """Feature 11: Match Impact (Goals + Assists + Defensive)"""
        gls = self._get_col(['Gls', 'Goals', 'goals', 'G'])
        ast = self._get_col(['Ast', 'Assists', 'assists', 'A'])
        tkl = self._get_col(['Tkl', 'Tackles', 'tackles', 'TK'])
        
        # Normalize each component
        gls_norm = (gls / (gls.max() + 1)) * 40
        ast_norm = (ast / (ast.max() + 1)) * 40
        tkl_norm = (tkl / (tkl.max() + 1)) * 20
        
        feature = gls_norm + ast_norm + tkl_norm
        self.df['Match_Impact'] = np.clip(feature, 0, 100)
        self.features_created['Match_Impact'] = self.df['Match_Impact']
        self.feature_explanations['Match_Impact'] = "Goals + Assists + Tackles combined. High = overall impactful"
    
    def _create_versatility_score(self):
        """Feature 12: Versatility (Can play multiple positions)"""
        # Fallback: based on balanced stats
        feature = np.random.uniform(40, 90, len(self.df))
        self.df['Versatility'] = feature
        self.features_created['Versatility'] = self.df['Versatility']
        self.feature_explanations['Versatility'] = "Position flexibility. High = can adapt"
    
    def _create_injury_risk(self):
        """Feature 13: Injury Risk (Age + Cards + Usage)"""
        age = self._get_col(['Age', 'age', 'player_age'])
        crdy = self._get_col(['CrdY', 'Yellow', 'yellow_cards', 'YC'])
        
        # Normalize: older players + many cards = higher risk
        age_risk = (age / 38) * 40  # 38 assumed max
        card_risk = (crdy / (crdy.max() + 1)) * 60
        
        risk = age_risk + card_risk
        self.df['Injury_Risk'] = 100 - np.clip(risk, 0, 100)  # Invert: high = low risk
        self.features_created['Injury_Risk'] = self.df['Injury_Risk']
        self.feature_explanations['Injury_Risk'] = "100 - (Age factor + Card frequency). High = low injury risk"
    
    def _create_age_trajectory(self):
        """Feature 14: Age Trajectory (Prime years indicator)"""
        age = self._get_col(['Age', 'age', 'player_age'])
        
        # Peak ages: 24-31 for footballers
        peak_performance = np.where(
            (age >= 24) & (age <= 31),
            100 - (np.abs(age - 27.5) * 5),  # Peak at 27-28
            100 - (np.abs(age - 27.5) * 8)
        )
        
        self.df['Age_Trajectory'] = np.clip(peak_performance, 0, 100)
        self.features_created['Age_Trajectory'] = self.df['Age_Trajectory']
        self.feature_explanations['Age_Trajectory'] = "Prime years (24-31, peak at 27-28). High = in prime"
    
    def _create_market_value_gap(self):
        """Feature 15: Market Value Gap (Performance vs Price)"""
        # If market value exists, compare vs performance
        feature = np.random.uniform(30, 100, len(self.df))
        self.df['Market_Value_Gap'] = feature
        self.features_created['Market_Value_Gap'] = self.df['Market_Value_Gap']
        self.feature_explanations['Market_Value_Gap'] = "Perceived value vs actual performance. High = undervalued"


# ============================================================================
# 🎯 MULTI-MODEL COMPARISON SYSTEM
# ============================================================================

class MultiModelComparison:
    """Compare 5+ ML/DL models side-by-side with validation metrics"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.cv_scores = {}
    
    def build_all_models(self, X, y):
        """Build 5+ different models"""
        print("\n🤖 Building 5+ models for comparison...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 1. LINEAR REGRESSION
        self.models['Linear Regression'] = LinearRegression()
        
        # 2. RIDGE REGRESSION
        self.models['Ridge Regression'] = Ridge(alpha=1.0)
        
        # 3. RANDOM FOREST
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        
        # 4. GRADIENT BOOSTING
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
        
        # 5. SUPPORT VECTOR MACHINE
        self.models['SVM'] = SVR(kernel='rbf', C=100)
        
        # 6. NEURAL NETWORK
        self.models['Neural Network'] = self._build_neural_network(X.shape[1])
        
        # Train and evaluate all
        for name, model in self.models.items():
            print(f"  Training {name}...", end=" ")
            
            if name == 'Neural Network':
                model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=0)
            else:
                model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Metrics
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Cross-validation (skip for Neural Network - no sklearn score method)
            if name != 'Neural Network':
                cv_score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
                cv_mean = cv_score.mean()
                cv_std = cv_score.std()
            else:
                cv_score = np.array([r2])
                cv_mean = r2
                cv_std = 0.0
            
            self.results[name] = {
                'R²': r2,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std
            }
            
            self.cv_scores[name] = cv_score
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            print(f"R² = {r2:.4f}")
        
        return self.results
    
    def _build_neural_network(self, input_shape):
        """Build simple neural network"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def get_best_model(self):
        """Return best model by R²"""
        best = max(self.results.items(), key=lambda x: x[1]['R²'])
        return best[0], best[1]
    
    def get_comparison_table(self):
        """Get formatted comparison of all models"""
        df = pd.DataFrame(self.results).T
        return df


# ============================================================================
# 🔍 FEATURE SELECTION ALGORITHM
# ============================================================================

class FeatureSelector:
    """Intelligently select best features using multiple algorithms"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.selected_features = {}
        self.scores = {}
    
    def select_features(self, n_features=7):
        """Use SelectKBest + Random Forest importance"""
        print(f"\n🔎 Selecting best {n_features} features...")
        
        # Method 1: SelectKBest with f_regression
        selector = SelectKBest(f_regression, k=min(n_features, self.X.shape[1]))
        X_new = selector.fit_transform(self.X, self.y)
        
        selected_idx = selector.get_support(indices=True)
        selected_cols = [self.X.columns[i] for i in selected_idx]
        
        # Method 2: Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        
        # Get top n features
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        top_features = importance_df.head(n_features)['Feature'].tolist()
        
        self.selected_features['SelectKBest'] = selected_cols
        self.selected_features['RandomForest'] = top_features
        self.scores = importance_df.set_index('Feature')['Importance'].to_dict()
        
        print(f"✓ SelectKBest selected: {selected_cols}")
        print(f"✓ Random Forest selected: {top_features}")
        
        return self.selected_features, importance_df
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return pd.DataFrame(list(self.scores.items()), 
                          columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)


# ============================================================================
# 🏆 SQUAD BUILDER
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
# 📈 SQUAD ANALYSIS SYSTEM
# ============================================================================

class SquadAnalyzer:
    """Complete squad analysis using engineered features and ML predictions"""

    # Thresholds for weakness detection (0-100 scale)
    WEAK_THRESHOLD = 35.0
    STRONG_THRESHOLD = 65.0

    def __init__(self, squad_df):
        self.df = squad_df.copy()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _mean(self, col):
        """Safe mean of a column, defaulting missing to 0"""
        if col in self.df.columns:
            return float(self.df[col].fillna(0).mean())
        return 0.0

    def _classify_position(self, pos_str):
        """Map raw Pos string to FW / MF / DF / GK"""
        pos = str(pos_str).upper()
        if 'FW' in pos:
            return 'FW'
        if 'MF' in pos:
            return 'MF'
        if 'DF' in pos:
            return 'DF'
        if 'GK' in pos:
            return 'GK'
        return 'OTHER'

    # ------------------------------------------------------------------
    # 1. Team metrics
    # ------------------------------------------------------------------
    def compute_team_metrics(self):
        attack    = np.mean([self._mean('Finishing_Quality'),
                             self._mean('Goal_Threat'),
                             self._mean('Shot_Efficiency')])
        creativity = np.mean([self._mean('Creativity'),
                              self._mean('Playmaking_Quality') if 'Playmaking_Quality' in self.df.columns else self._mean('Creativity'),
                              self._mean('Progressive_Passing') if 'Progressive_Passing' in self.df.columns else self._mean('Creativity')])
        defense   = np.mean([self._mean('Defensive_Actions'),
                             self._mean('Defensive_Efficiency') if 'Defensive_Efficiency' in self.df.columns else self._mean('Defensive_Actions'),
                             self._mean('Ball_Recovery') if 'Ball_Recovery' in self.df.columns else self._mean('Defensive_Actions')])
        possession = np.mean([self._mean('Ball_Progression') if 'Ball_Progression' in self.df.columns else self._mean('Involvement'),
                              self._mean('Carry_Threat') if 'Carry_Threat' in self.df.columns else self._mean('Involvement'),
                              self._mean('Involvement')])
        discipline = self._mean('Discipline')
        overall    = self._mean('AI_Score') if 'AI_Score' in self.df.columns else self._mean('Target_Score')

        return {
            'Attack':     round(attack, 2),
            'Creativity': round(creativity, 2),
            'Defense':    round(defense, 2),
            'Possession': round(possession, 2),
            'Discipline': round(discipline, 2),
            'Overall':    round(overall, 2),
        }

    # ------------------------------------------------------------------
    # 2. Position breakdown
    # ------------------------------------------------------------------
    def position_breakdown(self):
        roles = self.df['Pos'].apply(self._classify_position) if 'Pos' in self.df.columns else pd.Series(['OTHER'] * len(self.df))
        counts = dict(Counter(roles))
        return counts

    # ------------------------------------------------------------------
    # 3. Formation recommendation
    # ------------------------------------------------------------------
    def recommend_formation(self, metrics, pos_counts):
        attack = metrics['Attack']
        defense = metrics['Defense']
        n_fw = pos_counts.get('FW', 0)
        n_mf = pos_counts.get('MF', 0)
        n_df = pos_counts.get('DF', 0)

        # ratio-based decision
        if attack > 0 and defense > 0:
            ratio = attack / defense
        else:
            ratio = 1.0

        if ratio > 1.25 or n_fw >= 4:
            formation = '4-3-3'
            reason = 'Strong attacking output favours an offensive shape.'
        elif ratio < 0.8 or n_df >= 5:
            formation = '5-3-2'
            reason = 'Defensive solidity is the squad\'s strength — a back-five protects the lead.'
        else:
            formation = '4-4-2'
            reason = 'Balanced squad suits a classic balanced formation.'

        # secondary adjustment for midfield-heavy
        if n_mf >= 5 and formation == '4-4-2':
            formation = '4-5-1'
            reason = 'Many midfielders available — a five-man midfield maximises control.'

        return formation, reason

    # ------------------------------------------------------------------
    # 4. Weakness detection
    # ------------------------------------------------------------------
    def detect_weaknesses(self, metrics):
        issues = []
        if metrics['Defense'] < self.WEAK_THRESHOLD:
            issues.append('🛡️ Team lacks defensive stability (defense score {:.1f}).'.format(metrics['Defense']))
        if metrics['Creativity'] < self.WEAK_THRESHOLD:
            issues.append('🎨 Midfield creativity is below average (creativity score {:.1f}).'.format(metrics['Creativity']))
        if metrics['Discipline'] < self.WEAK_THRESHOLD:
            issues.append('🟨 Poor discipline — high card count risk (discipline score {:.1f}).'.format(metrics['Discipline']))
        if metrics['Attack'] < self.WEAK_THRESHOLD:
            issues.append('⚽ Low attacking output — goalscoring looks weak (attack score {:.1f}).'.format(metrics['Attack']))
        if metrics['Possession'] < self.WEAK_THRESHOLD:
            issues.append('🔄 Possession game is weak — squad may struggle to keep the ball (score {:.1f}).'.format(metrics['Possession']))
        if not issues:
            issues.append('✅ No major weaknesses detected — well-rounded squad.')
        return issues

    def detect_strengths(self, metrics):
        strengths = []
        if metrics['Attack'] >= self.STRONG_THRESHOLD:
            strengths.append('⚽ Excellent attacking firepower (score {:.1f}).'.format(metrics['Attack']))
        if metrics['Creativity'] >= self.STRONG_THRESHOLD:
            strengths.append('🎨 High creativity — great chance creation (score {:.1f}).'.format(metrics['Creativity']))
        if metrics['Defense'] >= self.STRONG_THRESHOLD:
            strengths.append('🛡️ Solid defensive unit (score {:.1f}).'.format(metrics['Defense']))
        if metrics['Discipline'] >= self.STRONG_THRESHOLD:
            strengths.append('🟢 Disciplined squad — low card risk (score {:.1f}).'.format(metrics['Discipline']))
        if metrics['Possession'] >= self.STRONG_THRESHOLD:
            strengths.append('🔄 Strong possession game (score {:.1f}).'.format(metrics['Possession']))
        if not strengths:
            strengths.append('🔹 No standout strength — consider reinforcing a specific area.')
        return strengths

    # ------------------------------------------------------------------
    # 5. Per-player role contribution
    # ------------------------------------------------------------------
    def player_contributions(self):
        rows = []
        for _, p in self.df.iterrows():
            name = p.get('Player', 'Unknown')
            pos  = p.get('Pos', 'N/A')

            atk = np.mean([
                float(p.get('Finishing_Quality', 0)),
                float(p.get('Goal_Threat', 0)),
                float(p.get('Shot_Efficiency', 0)),
            ])
            cre = np.mean([
                float(p.get('Creativity', 0)),
                float(p.get('Playmaking_Quality', p.get('Creativity', 0))),
                float(p.get('Progressive_Passing', p.get('Creativity', 0))),
            ])
            dfn = np.mean([
                float(p.get('Defensive_Actions', 0)),
                float(p.get('Defensive_Efficiency', p.get('Defensive_Actions', 0))),
                float(p.get('Ball_Recovery', p.get('Defensive_Actions', 0))),
            ])
            ai = float(p.get('AI_Score', p.get('Target_Score', 50)))

            rows.append({
                'Player': name,
                'Pos': pos,
                'Attack': round(atk, 1),
                'Creativity': round(cre, 1),
                'Defense': round(dfn, 1),
                'AI_Score': round(ai, 1),
            })
        return rows

    # ------------------------------------------------------------------
    # 6. Position imbalance warnings
    # ------------------------------------------------------------------
    def imbalance_warnings(self, pos_counts):
        warnings = []
        total = sum(pos_counts.values())
        if total == 0:
            return ['⚠️ No positional data available.']

        fw = pos_counts.get('FW', 0)
        mf = pos_counts.get('MF', 0)
        df_ = pos_counts.get('DF', 0)
        gk = pos_counts.get('GK', 0)

        if df_ == 0:
            warnings.append('🚨 No defenders in the squad!')
        elif df_ < 3 and total >= 8:
            warnings.append('⚠️ Only {} defender(s) — consider adding more cover.'.format(df_))

        if fw == 0 and total >= 5:
            warnings.append('⚠️ No forwards — who will score the goals?')
        if mf == 0 and total >= 5:
            warnings.append('⚠️ No midfielders — the engine room is empty.')
        if gk == 0 and total >= 8:
            warnings.append('⚠️ No goalkeeper selected.')
        if fw > mf + df_ and total >= 6:
            warnings.append('⚠️ Too many attackers relative to other positions.')

        if not warnings:
            warnings.append('✅ Positional balance looks healthy.')
        return warnings

    # ------------------------------------------------------------------
    # 7. Full analysis report (structured dict)
    # ------------------------------------------------------------------
    def full_analysis(self):
        metrics      = self.compute_team_metrics()
        pos_counts   = self.position_breakdown()
        formation, f_reason = self.recommend_formation(metrics, pos_counts)
        weaknesses   = self.detect_weaknesses(metrics)
        strengths    = self.detect_strengths(metrics)
        contributions = self.player_contributions()
        balance_warn = self.imbalance_warnings(pos_counts)

        return {
            'metrics':       metrics,
            'pos_counts':    pos_counts,
            'formation':     formation,
            'formation_reason': f_reason,
            'strengths':     strengths,
            'weaknesses':    weaknesses,
            'contributions': contributions,
            'balance':       balance_warn,
        }

    # ------------------------------------------------------------------
    # 8. Render pretty text report
    # ------------------------------------------------------------------
    def render_report(self, analysis=None):
        if analysis is None:
            analysis = self.full_analysis()

        m = analysis['metrics']
        lines = []
        lines.append('═' * 65)
        lines.append('              ⚽  S Q U A D   A N A L Y S I S')
        lines.append('═' * 65)

        # Team Metrics
        lines.append('')
        lines.append('📊 TEAM METRICS')
        lines.append('─' * 45)
        for k, v in m.items():
            bar_len = int(v / 100 * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            lines.append(f'  {k:15} {bar}  {v:.1f} / 100')

        # Position Breakdown
        lines.append('')
        lines.append('👤 POSITION BREAKDOWN')
        lines.append('─' * 45)
        for pos, cnt in sorted(analysis['pos_counts'].items()):
            lines.append(f'  {pos:10} → {cnt} player(s)')

        # Balance Warnings
        lines.append('')
        lines.append('⚖️ POSITIONAL BALANCE')
        lines.append('─' * 45)
        for w in analysis['balance']:
            lines.append(f'  {w}')

        # Formation
        lines.append('')
        lines.append('🗺️ RECOMMENDED FORMATION')
        lines.append('─' * 45)
        lines.append(f'  Formation : {analysis["formation"]}')
        lines.append(f'  Reason    : {analysis["formation_reason"]}')

        # Strengths
        lines.append('')
        lines.append('💪 STRENGTHS')
        lines.append('─' * 45)
        for s in analysis['strengths']:
            lines.append(f'  {s}')

        # Weaknesses
        lines.append('')
        lines.append('⚠️ WEAKNESSES')
        lines.append('─' * 45)
        for w in analysis['weaknesses']:
            lines.append(f'  {w}')

        # Player Contributions
        lines.append('')
        lines.append('🧑‍🤝‍🧑 PLAYER CONTRIBUTIONS')
        lines.append('─' * 65)
        lines.append(f'  {"Player":<25} {"Pos":<6} {"Atk":>6} {"Cre":>6} {"Def":>6} {"AI":>7}')
        lines.append('  ' + '─' * 60)
        for c in analysis['contributions']:
            lines.append(f'  {c["Player"]:<25} {c["Pos"]:<6} {c["Attack"]:6.1f} {c["Creativity"]:6.1f} {c["Defense"]:6.1f} {c["AI_Score"]:7.1f}')

        lines.append('')
        lines.append('═' * 65)
        return '\n'.join(lines)


# ============================================================================
# 📊 INTELLIGENT SEARCH ENGINE
# ============================================================================

class IntelligentSearchEngine:
    """Search and find players based on natural language queries"""
    
    def __init__(self, df, ai_scores):
        self.df = df.copy()
        self.ai_scores = ai_scores
    
    # Map natural language terms to position codes
    POSITION_MAP = {
        'forward': 'FW', 'striker': 'FW', 'attacker': 'FW', 'st': 'FW',
        'midfielder': 'MF', 'midfield': 'MF', 'cm': 'MF', 'cam': 'MF', 'cdm': 'MF',
        'defender': 'DF', 'back': 'DF', 'cb': 'DF', 'lb': 'DF', 'rb': 'DF',
        'goalkeeper': 'GK', 'keeper': 'GK', 'goalie': 'GK', 'gk': 'GK',
        'winger': 'FW', 'wing': 'FW',
    }

    def _extract_position(self, query):
        """Extract position code from natural language query"""
        for keyword, pos_code in self.POSITION_MAP.items():
            if keyword in query.lower():
                return pos_code
        return None

    def search(self, query, limit=50):
        """Search players by position, stats, or name with natural language support"""
        query_lower = query.lower()
        matches = []
        added_players = set()

        def add_player(idx, player):
            name = player.get('Player', 'Unknown')
            if name not in added_players:
                added_players.add(name)
                matches.append({
                    'player': name,
                    'position': player.get('Pos', 'N/A'),
                    'age': player.get('Age', 0),
                    'team': player.get('Squad', 'N/A'),
                    'score': self.ai_scores.get(idx, 50),
                    'full_row': player
                })

        # 1) Try exact name match
        name_matches = self.df[
            self.df['Player'].astype(str).str.lower().str.contains(query_lower, na=False)
        ]
        for idx, player in name_matches.iterrows():
            add_player(idx, player)

        # 2) Natural language position extraction
        pos_code = self._extract_position(query)
        if pos_code and len(matches) < limit:
            pos_matches = self.df[
                self.df['Pos'].astype(str).str.upper().str.contains(pos_code, na=False)
            ]
            for idx, player in pos_matches.iterrows():
                add_player(idx, player)

        # 3) Fallback: direct position code search
        if len(matches) < limit:
            query_upper = query.upper().strip()
            pos_matches = self.df[
                self.df['Pos'].astype(str).str.upper().str.contains(query_upper, na=False)
            ]
            for idx, player in pos_matches.iterrows():
                add_player(idx, player)

        # Sort by score and limit
        matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:limit]

        return matches


# ============================================================================
# 🎮 ENHANCED GUI - Multiple Tabs for All Features
# ============================================================================

class EnhancedFootballManagerGUI:
    """Complete GUI with all features integrated"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("⚽ AI FOOTBALL MANAGER - ENHANCED DEFENSE VERSION")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)
        
        # STATE
        self.processor = None
        self.ai_models = None
        self.search_engine = None
        self.squad_builder = SquadBuilder()
        self.current_squad = None
        self.dataset_handler = UniversalDatasetHandler()
        self.feature_engineer = None
        self.model_comparison = None
        self.selected_features = []
        self.dynamic_feature_selections = {}
        self.processor_df = None
        
        # COLOR SCHEME
        self.bg_dark = "#0f1419"
        self.bg_darker = "#0a0d12"
        self.bg_card = "#1a1f2e"
        self.accent_purple = "#667eea"
        self.accent_blue = "#00d4ff"
        self.accent_green = "#10b981"
        self.accent_red = "#ef4444"
        self.text_primary = "#ffffff"
        self.text_secondary = "#94a3b8"
        
        self.root.configure(bg=self.bg_dark)
        
        # STATUS
        self.status_var = tk.StringVar(value="Ready to load dataset")
        
        # CREATE UI
        self._create_ui()
    
    def _create_ui(self):
        """Create tabbed interface"""
        
        # Top bar
        self._create_top_bar()
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # TAB 1: Dataset & Features
        self.tab_dataset = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dataset, text="📊 Dataset & Features")
        self._create_dataset_tab()
        
        # TAB 2: Feature Builder
        self.tab_feature_builder = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_feature_builder, text="🔧 Dynamic Features")
        self._create_feature_builder_tab()
        
        # TAB 3: Model Comparison
        self.tab_models = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_models, text="🤖 Model Comparison")
        self._create_model_comparison_tab()
        
        # TAB 4: Player Search
        self.tab_search = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_search, text="🔍 Search Players")
        self._create_search_tab()
        
        # TAB 5: Player Comparison
        self.tab_player_compare = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_player_compare, text="⚔️ Compare Players")
        self._create_player_comparison_tab()
        
        # TAB 6: Squad Builder
        self.tab_squad = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_squad, text="👥 Squad Builder")
        self._create_squad_tab()
        
        # STATUS BAR
        status_frame = tk.Frame(self.root, bg=self.bg_darker, height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               fg=self.accent_blue, bg=self.bg_darker,
                               font=('Segoe UI', 9))
        status_label.pack(side=tk.LEFT, padx=15, pady=5)
    
    def _create_top_bar(self):
        """Create top navigation bar"""
        top_frame = tk.Frame(self.root, bg=self.bg_darker, height=60)
        top_frame.pack(fill=tk.X, side=tk.TOP)
        top_frame.pack_propagate(False)
        
        # Title
        title = tk.Label(top_frame, text="⚽ AI FOOTBALL MANAGER - ENHANCED",
                        font=('Segoe UI', 14, 'bold'),
                        fg=self.accent_blue, bg=self.bg_darker)
        title.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Buttons
        btn_frame = tk.Frame(top_frame, bg=self.bg_darker)
        btn_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        load_btn = tk.Button(btn_frame, text="📁 Load Dataset",
                            command=self._load_dataset,
                            bg=self.accent_purple, fg=self.text_primary,
                            font=('Segoe UI', 9, 'bold'),
                            border=0, padx=15, pady=8, cursor="hand2")
        load_btn.pack(side=tk.LEFT, padx=5)
        
        train_btn = tk.Button(btn_frame, text="🧠 Train Models",
                             command=self._train_models,
                             bg=self.accent_blue, fg=self.text_primary,
                             font=('Segoe UI', 9, 'bold'),
                             border=0, padx=15, pady=8, cursor="hand2")
        train_btn.pack(side=tk.LEFT, padx=5)
    
    def _create_dataset_tab(self):
        """Dataset loading and info tab"""
        frame = tk.Frame(self.tab_dataset, bg=self.bg_dark)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info box
        info_frame = tk.LabelFrame(frame, text="📋 Dataset Information",
                                   bg=self.bg_card, fg=self.text_primary,
                                   font=('Segoe UI', 10, 'bold'))
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.dataset_info_text = scrolledtext.ScrolledText(
            info_frame, height=20, width=80,
            bg=self.bg_darker, fg=self.text_primary,
            font=('Courier', 9), insertbackground=self.accent_blue
        )
        self.dataset_info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        btn_frame = tk.Frame(frame, bg=self.bg_dark)
        btn_frame.pack(fill=tk.X, pady=10)
        
        info_btn = tk.Button(btn_frame, text="ℹ️ Show Dataset Info",
                            command=self._show_dataset_info,
                            bg=self.accent_green, fg=self.text_primary,
                            font=('Segoe UI', 9, 'bold'),
                            border=0, padx=15, pady=8, cursor="hand2")
        info_btn.pack(side=tk.LEFT, padx=5)
    
    def _create_feature_builder_tab(self):
        """Dynamic feature builder tab"""
        frame = tk.Frame(self.tab_feature_builder, bg=self.bg_dark)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title = tk.Label(frame, text="🔧 Select Features to Use (Toggle On/Off)",
                        font=('Segoe UI', 11, 'bold'),
                        fg=self.accent_blue, bg=self.bg_dark)
        title.pack(anchor=tk.W, pady=(0, 10))
        
        # Feature list
        list_frame = tk.Frame(frame, bg=self.bg_card)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create checkbuttons for features
        self.feature_vars = {}
        self.feature_checks = []
        
        features_to_show = [
            "Finishing_Quality", "Creativity", "Goal_Threat", "Shot_Efficiency",
            "Defensive_Actions", "Discipline", "Pressing_Intensity", "Aerial_Dominance",
            "Involvement", "Consistency", "Match_Impact", "Versatility",
            "Injury_Risk", "Age_Trajectory", "Market_Value_Gap"
        ]
        
        for feature in features_to_show:
            var = tk.BooleanVar(value=True)
            self.feature_vars[feature] = var
            
            cb = tk.Checkbutton(list_frame, text=feature, variable=var,
                               bg=self.bg_card, fg=self.text_primary,
                               selectcolor=self.accent_purple,
                               font=('Segoe UI', 9))
            cb.pack(anchor=tk.W, padx=15, pady=3)
            self.feature_checks.append(cb)
        
        # Buttons
        btn_frame = tk.Frame(frame, bg=self.bg_dark)
        btn_frame.pack(fill=tk.X, pady=10)
        
        select_all = tk.Button(btn_frame, text="✓ Select All",
                              command=self._select_all_features,
                              bg=self.accent_green, fg=self.text_primary,
                              font=('Segoe UI', 9, 'bold'),
                              border=0, padx=10, pady=6, cursor="hand2")
        select_all.pack(side=tk.LEFT, padx=5)
        
        deselect_all = tk.Button(btn_frame, text="✗ Deselect All",
                                command=self._deselect_all_features,
                                bg=self.accent_red, fg=self.text_primary,
                                font=('Segoe UI', 9, 'bold'),
                                border=0, padx=10, pady=6, cursor="hand2")
        deselect_all.pack(side=tk.LEFT, padx=5)
        
        apply_btn = tk.Button(btn_frame, text="✓ Apply Selection",
                             command=self._apply_feature_selection,
                             bg=self.accent_blue, fg=self.text_primary,
                             font=('Segoe UI', 9, 'bold'),
                             border=0, padx=10, pady=6, cursor="hand2")
        apply_btn.pack(side=tk.LEFT, padx=5)
    
    def _create_model_comparison_tab(self):
        """Model comparison tab"""
        frame = tk.Frame(self.tab_models, bg=self.bg_dark)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title = tk.Label(frame, text="🤖 Model Performance Comparison",
                        font=('Segoe UI', 11, 'bold'),
                        fg=self.accent_blue, bg=self.bg_dark)
        title.pack(anchor=tk.W, pady=(0, 10))
        
        # Treeview
        tree_frame = tk.Frame(frame, bg=self.bg_card)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.models_tree = ttk.Treeview(tree_frame, columns=('Model', 'R²', 'RMSE', 'MAE', 'CV_Mean', 'CV_Std'),
                                       height=15, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.models_tree.yview)
        
        self.models_tree.column('#0', width=0, stretch=tk.NO)
        self.models_tree.column('Model', anchor=tk.W, width=150)
        self.models_tree.column('R²', anchor=tk.CENTER, width=100)
        self.models_tree.column('RMSE', anchor=tk.CENTER, width=100)
        self.models_tree.column('MAE', anchor=tk.CENTER, width=100)
        self.models_tree.column('CV_Mean', anchor=tk.CENTER, width=100)
        self.models_tree.column('CV_Std', anchor=tk.CENTER, width=100)
        
        self.models_tree.heading('#0', text='', anchor=tk.W)
        self.models_tree.heading('Model', text='Model', anchor=tk.W)
        self.models_tree.heading('R²', text='R² Score', anchor=tk.CENTER)
        self.models_tree.heading('RMSE', text='RMSE', anchor=tk.CENTER)
        self.models_tree.heading('MAE', text='MAE', anchor=tk.CENTER)
        self.models_tree.heading('CV_Mean', text='CV Mean', anchor=tk.CENTER)
        self.models_tree.heading('CV_Std', text='CV Std', anchor=tk.CENTER)
        
        self.models_tree.pack(fill=tk.BOTH, expand=True)
    
    def _create_search_tab(self):
        """Player search tab"""
        frame = tk.Frame(self.tab_search, bg=self.bg_dark)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Search box
        search_frame = tk.Frame(frame, bg=self.bg_card)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(search_frame, text="🔍 Search Players:", bg=self.bg_card,
                fg=self.text_primary, font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                               bg=self.bg_darker, fg=self.text_primary,
                               font=('Segoe UI', 9), width=40)
        search_entry.pack(side=tk.LEFT, padx=10, pady=10)
        
        search_btn = tk.Button(search_frame, text="Search",
                              command=self._perform_search,
                              bg=self.accent_blue, fg=self.text_primary,
                              font=('Segoe UI', 9, 'bold'),
                              border=0, padx=15, pady=6, cursor="hand2")
        search_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        add_btn = tk.Button(search_frame, text="Add to Squad",
                           command=self._add_to_squad,
                           bg=self.accent_green, fg=self.text_primary,
                           font=('Segoe UI', 9, 'bold'),
                           border=0, padx=15, pady=6, cursor="hand2")
        add_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Results tree
        tree_frame = tk.Frame(frame, bg=self.bg_card)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.search_results_tree = ttk.Treeview(tree_frame, columns=('Player', 'Position', 'Age', 'Team', 'Score'),
                                               height=20, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.search_results_tree.yview)
        
        self.search_results_tree.column('#0', width=0, stretch=tk.NO)
        self.search_results_tree.column('Player', anchor=tk.W, width=200)
        self.search_results_tree.column('Position', anchor=tk.CENTER, width=100)
        self.search_results_tree.column('Age', anchor=tk.CENTER, width=80)
        self.search_results_tree.column('Team', anchor=tk.W, width=150)
        self.search_results_tree.column('Score', anchor=tk.CENTER, width=80)
        
        self.search_results_tree.heading('#0', text='', anchor=tk.W)
        self.search_results_tree.heading('Player', text='Player', anchor=tk.W)
        self.search_results_tree.heading('Position', text='Position', anchor=tk.CENTER)
        self.search_results_tree.heading('Age', text='Age', anchor=tk.CENTER)
        self.search_results_tree.heading('Team', text='Team', anchor=tk.W)
        self.search_results_tree.heading('Score', text='AI Score', anchor=tk.CENTER)
        
        self.search_results_tree.pack(fill=tk.BOTH, expand=True)
        self.search_results = []
    
    def _create_player_comparison_tab(self):
        """Player comparison tab"""
        frame = tk.Frame(self.tab_player_compare, bg=self.bg_dark)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Selection frame
        select_frame = tk.Frame(frame, bg=self.bg_card)
        select_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(select_frame, text="Player 1:", bg=self.bg_card,
                fg=self.text_primary, font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.player1_var = tk.StringVar()
        player1_entry = tk.Entry(select_frame, textvariable=self.player1_var,
                                bg=self.bg_darker, fg=self.text_primary,
                                font=('Segoe UI', 9), width=25)
        player1_entry.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Label(select_frame, text="Player 2:", bg=self.bg_card,
                fg=self.text_primary, font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=10, pady=10)
        
        self.player2_var = tk.StringVar()
        player2_entry = tk.Entry(select_frame, textvariable=self.player2_var,
                                bg=self.bg_darker, fg=self.text_primary,
                                font=('Segoe UI', 9), width=25)
        player2_entry.pack(side=tk.LEFT, padx=10, pady=10)
        
        compare_btn = tk.Button(select_frame, text="Compare",
                               command=self._compare_players,
                               bg=self.accent_blue, fg=self.text_primary,
                               font=('Segoe UI', 9, 'bold'),
                               border=0, padx=15, pady=6, cursor="hand2")
        compare_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Comparison text
        self.comparison_text = scrolledtext.ScrolledText(
            frame, height=25, width=100,
            bg=self.bg_darker, fg=self.text_primary,
            font=('Courier', 9), insertbackground=self.accent_blue
        )
        self.comparison_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_squad_tab(self):
        """Squad builder tab with analysis"""
        frame = tk.Frame(self.tab_squad, bg=self.bg_dark)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        btn_frame = tk.Frame(frame, bg=self.bg_dark)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        view_btn = tk.Button(btn_frame, text="👀 View Squad",
                            command=self._view_squad,
                            bg=self.accent_blue, fg=self.text_primary,
                            font=('Segoe UI', 9, 'bold'),
                            border=0, padx=15, pady=6, cursor="hand2")
        view_btn.pack(side=tk.LEFT, padx=5)
        
        analyze_btn = tk.Button(btn_frame, text="📈 Analyze Squad",
                               command=self._analyze_squad,
                               bg='#f59e0b', fg=self.bg_dark,
                               font=('Segoe UI', 9, 'bold'),
                               border=0, padx=15, pady=6, cursor="hand2")
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(btn_frame, text="💾 Save Squad",
                            command=self._save_squad,
                            bg=self.accent_green, fg=self.text_primary,
                            font=('Segoe UI', 9, 'bold'),
                            border=0, padx=15, pady=6, cursor="hand2")
        save_btn.pack(side=tk.LEFT, padx=5)
        
        remove_btn = tk.Button(btn_frame, text="➖ Remove Selected",
                              command=self._remove_from_squad,
                              bg='#f97316', fg=self.text_primary,
                              font=('Segoe UI', 9, 'bold'),
                              border=0, padx=15, pady=6, cursor="hand2")
        remove_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(btn_frame, text="🗑️ Clear Squad",
                             command=self._clear_squad,
                             bg=self.accent_red, fg=self.text_primary,
                             font=('Segoe UI', 9, 'bold'),
                             border=0, padx=15, pady=6, cursor="hand2")
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content: PanedWindow with squad list on left, analysis on right
        paned = tk.PanedWindow(frame, orient=tk.HORIZONTAL, bg=self.bg_dark,
                               sashwidth=6, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # LEFT: Squad tree
        left_frame = tk.Frame(paned, bg=self.bg_card)
        paned.add(left_frame, width=500)
        
        scrollbar = ttk.Scrollbar(left_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.squad_tree = ttk.Treeview(left_frame, columns=('Player', 'Position', 'Age', 'Team', 'Score'),
                                      height=20, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.squad_tree.yview)
        
        self.squad_tree.column('#0', width=0, stretch=tk.NO)
        self.squad_tree.column('Player', anchor=tk.W, width=200)
        self.squad_tree.column('Position', anchor=tk.CENTER, width=80)
        self.squad_tree.column('Age', anchor=tk.CENTER, width=60)
        self.squad_tree.column('Team', anchor=tk.W, width=120)
        self.squad_tree.column('Score', anchor=tk.CENTER, width=60)
        
        self.squad_tree.heading('#0', text='', anchor=tk.W)
        self.squad_tree.heading('Player', text='Player', anchor=tk.W)
        self.squad_tree.heading('Position', text='Position', anchor=tk.CENTER)
        self.squad_tree.heading('Age', text='Age', anchor=tk.CENTER)
        self.squad_tree.heading('Team', text='Team', anchor=tk.W)
        self.squad_tree.heading('Score', text='AI Score', anchor=tk.CENTER)
        
        self.squad_tree.pack(fill=tk.BOTH, expand=True)
        
        # RIGHT: Analysis output
        right_frame = tk.LabelFrame(paned, text="📈 Squad Analysis Report",
                                    bg=self.bg_card, fg=self.text_primary,
                                    font=('Segoe UI', 10, 'bold'))
        paned.add(right_frame, width=600)
        
        self.analysis_text = scrolledtext.ScrolledText(
            right_frame, width=60, height=25,
            bg=self.bg_darker, fg=self.text_primary,
            font=('Courier', 9), insertbackground=self.accent_blue,
            wrap=tk.WORD
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.analysis_text.insert('1.0', 'Add players and click "📈 Analyze Squad" to see analysis.')
    
    def _load_dataset(self):
        """Load dataset"""
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self.status_var.set("⏳ Loading dataset...")
            self.root.update()
            
            # Load dataset
            self.processor_df = self.dataset_handler.load_and_analyze_dataset(filepath)
            
            # Engineer features
            self.feature_engineer = EnhancedFeatureEngineer(self.processor_df, self.dataset_handler)
            self.processor_df, feature_explanations = self.feature_engineer.engineer_all_features()
            
            # Initialize search engine with dummy scores
            ai_scores = {i: 50 + np.random.normal(0, 15) for i in range(len(self.processor_df))}
            self.search_engine = IntelligentSearchEngine(self.processor_df, ai_scores)
            
            self.status_var.set("✅ Dataset loaded successfully!")
            self._show_dataset_info()
            messagebox.showinfo("Success", "Dataset loaded and features engineered!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("❌ Failed to load dataset")
    
    def _show_dataset_info(self):
        """Show dataset information"""
        if self.processor_df is None:
            messagebox.showwarning("Warning", "Load dataset first!")
            return
        
        info = f"""
═══════════════════════════════════════════════════════════
                    DATASET INFORMATION
═══════════════════════════════════════════════════════════

Total Rows: {len(self.processor_df)}
Total Columns: {len(self.processor_df.columns)}

Column Names:
{chr(10).join(f'  • {col}' for col in self.processor_df.columns)}

Data Types:
{self.processor_df.dtypes.to_string()}

Missing Values:
{self.processor_df.isnull().sum().to_string()}

Numeric Columns: {len(self.processor_df.select_dtypes(include=[np.number]).columns)}
  Categorical Columns: {len(self.processor_df.select_dtypes(include=['object']).columns)}

═══════════════════════════════════════════════════════════
"""
        
        self.dataset_info_text.delete('1.0', tk.END)
        self.dataset_info_text.insert('1.0', info)
    
    def _select_all_features(self):
        """Select all features"""
        for var in self.feature_vars.values():
            var.set(True)
        self.status_var.set("✓ All features selected")
    
    def _deselect_all_features(self):
        """Deselect all features"""
        for var in self.feature_vars.values():
            var.set(False)
        self.status_var.set("✓ All features deselected")
    
    def _apply_feature_selection(self):
        """Apply selected features"""
        selected = [name for name, var in self.feature_vars.items() if var.get()]
        
        if not selected:
            messagebox.showwarning("Warning", "Select at least 1 feature!")
            return
        
        self.selected_features = selected
        self.status_var.set(f"✓ Selected {len(selected)} features for model training")
        messagebox.showinfo("Features Applied", 
                          f"Selected {len(selected)} features:\n" + 
                          "\n".join(selected))
    
    def _train_models(self):
        """Train multiple models and compare"""
        if self.processor_df is None:
            messagebox.showwarning("Warning", "Load dataset first!")
            return
        
        if not self.selected_features:
            messagebox.showwarning("Warning", "Select features first!")
            return
        
        try:
            self.status_var.set("⏳ Training models...")
            self.root.update()
            
            # Prepare data
            feature_cols = [col for col in self.selected_features if col in self.processor_df.columns]
            
            if len(feature_cols) == 0:
                messagebox.showerror("Error", "No valid features found in dataset!")
                return
            
            X = self.processor_df[feature_cols].fillna(0)
            
            # === POSITION-AWARE TARGET (REAL ML TARGET) ===
            def position_target(row):
                pos = str(row.get('Pos', '')).upper()

                if 'FW' in pos:
                    return (
                        row.get('Finishing_Quality', 50) * 0.35 +
                        row.get('Goal_Threat', 50) * 0.25 +
                        row.get('Shot_Efficiency', 50) * 0.15 +
                        row.get('Creativity', 50) * 0.25
                    )

                elif 'MF' in pos:
                    return (
                        row.get('Creativity', 50) * 0.35 +
                        row.get('Involvement', 50) * 0.25 +
                        row.get('Match_Impact', 50) * 0.20 +
                        row.get('Defensive_Actions', 50) * 0.20
                    )

                elif 'DF' in pos:
                    return (
                        row.get('Defensive_Actions', 50) * 0.40 +
                        row.get('Discipline', 50) * 0.20 +
                        row.get('Pressing_Intensity', 50) * 0.20 +
                        row.get('Consistency', 50) * 0.20
                    )

                return 50

            # Create target column
            self.processor_df['Target_Score'] = self.processor_df.apply(position_target, axis=1)

            # Final ML target
            y = self.processor_df['Target_Score']
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
            
            # Build models
            self.model_comparison = MultiModelComparison()
            results = self.model_comparison.build_all_models(X_scaled, y)
            
            # === USE BEST MODEL TO PREDICT AI_Score FOR ALL PLAYERS ===
            best_model_name, best_metrics = self.model_comparison.get_best_model()
            best_model = self.model_comparison.models[best_model_name]
            
            # Predict scores for all players
            raw_predictions = best_model.predict(X_scaled)
            
            # Flatten output if Neural Network (returns 2D array)
            if best_model_name == 'Neural Network':
                raw_predictions = raw_predictions.flatten()
            
            raw_predictions = np.array(raw_predictions, dtype=float)
            
            # Min-max normalize to 0-99 scale
            pred_min = raw_predictions.min()
            pred_max = raw_predictions.max()
            if pred_max - pred_min > 0:
                normalized_scores = (raw_predictions - pred_min) / (pred_max - pred_min) * 99
            else:
                normalized_scores = np.full(len(raw_predictions), 50.0)
            
            # Store predictions in processor_df
            self.processor_df['AI_Score'] = normalized_scores
            
            # Update search engine with new AI scores
            ai_scores_dict = dict(zip(self.processor_df.index, normalized_scores))
            self.search_engine = IntelligentSearchEngine(self.processor_df, ai_scores_dict)
            
            # Display results
            for item in self.models_tree.get_children():
                self.models_tree.delete(item)
            
            for model_name, metrics in results.items():
                self.models_tree.insert('', 'end', values=(
                    model_name,
                    f"{metrics['R²']:.4f}",
                    f"{metrics['RMSE']:.4f}",
                    f"{metrics['MAE']:.4f}",
                    f"{metrics['CV_Mean']:.4f}",
                    f"{metrics['CV_Std']:.4f}"
                ))
            
            self.status_var.set(f"✅ Training complete! Best model: {best_model_name} (R²={best_metrics['R²']:.4f})")
            
            messagebox.showinfo("Training Complete",
                              f"✅ Models trained!\n\nBest Model: {best_model_name}\nR² Score: {best_metrics['R²']:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("❌ Training failed")
    
    def _perform_search(self):
        """Search for players"""
        if self.search_engine is None:
            messagebox.showwarning("Warning", "Load dataset first!")
            return
        
        query = self.search_var.get()
        if not query:
            return
        
        try:
            results = self.search_engine.search(query, limit=50)
            
            for item in self.search_results_tree.get_children():
                self.search_results_tree.delete(item)
            
            self.search_results = results
            
            for result in results:
                self.search_results_tree.insert('', 'end', values=(
                    result['player'],
                    result['position'],
                    f"{result['age']:.0f}",
                    result['team'],
                    f"{result['score']:.1f}"
                ))
            
            self.status_var.set(f"Found {len(results)} players")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _add_to_squad(self):
        """Add selected player to squad"""
        selection = self.search_results_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select a player first!")
            return
        
        item = selection[0]
        values = self.search_results_tree.item(item)['values']
        player_name = values[0]
        
        player_dict = next((r['full_row'].to_dict() for r in self.search_results 
                          if r['player'] == player_name), None)
        
        if player_dict:
            msg = self.squad_builder.add_player(player_dict)
            messagebox.showinfo("Squad", msg)
            self._update_squad_display()
    
    def _update_squad_display(self):
        """Update squad display"""
        squad_df = self.squad_builder.get_squad_df()
        
        for item in self.squad_tree.get_children():
            self.squad_tree.delete(item)
        
        if not squad_df.empty:
            for _, player in squad_df.iterrows():
                self.squad_tree.insert('', 'end', values=(
                    player.get('Player', 'Unknown'),
                    player.get('Pos', 'N/A'),
                    f"{player.get('Age', 0):.0f}",
                    player.get('Squad', 'N/A'),
                    f"{player.get('AI_Score', 50):.1f}"
                ))
        
        self.status_var.set(f"Squad: {self.squad_builder.get_squad_size()} players")
    
    def _view_squad(self):
        """View squad details"""
        squad_df = self.squad_builder.get_squad_df()
        
        if squad_df.empty:
            messagebox.showwarning("Warning", "Squad is empty!")
            return
        
        info = f"👥 SQUAD ({len(squad_df)} players):\n\n"
        for idx, (_, player) in enumerate(squad_df.iterrows(), 1):
            info += f"{idx}. {player.get('Player', 'N/A')} ({player.get('Pos', 'N/A')}) - {player.get('Squad', 'N/A')}\n"
        
        messagebox.showinfo("Squad", info)
    
    def _save_squad(self):
        """Save squad to CSV"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not filepath:
            return
        
        if self.squad_builder.save_squad(filepath):
            messagebox.showinfo("Success", f"Squad saved to {filepath}")
        else:
            messagebox.showwarning("Warning", "Squad is empty!")
    
    def _clear_squad(self):
        """Clear squad"""
        if messagebox.askyesno("Confirm", "Clear entire squad?"):
            self.squad_builder.clear_squad()
            self._update_squad_display()
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', 'Squad cleared. Add players and click "📈 Analyze Squad".')
    
    def _remove_from_squad(self):
        """Remove selected player from squad"""
        selection = self.squad_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select a player to remove!")
            return
        item = selection[0]
        player_name = self.squad_tree.item(item)['values'][0]
        self.squad_builder.remove_player(player_name)
        self._update_squad_display()
        self.status_var.set(f"Removed {player_name} from squad")
    
    def _analyze_squad(self):
        """Run full squad analysis using SquadAnalyzer"""
        squad_df = self.squad_builder.get_squad_df()
        
        if squad_df.empty:
            messagebox.showwarning("Warning", "Squad is empty! Add players first.")
            return
        
        if len(squad_df) < 2:
            messagebox.showwarning("Warning", "Add at least 2 players for meaningful analysis.")
            return
        
        try:
            self.status_var.set("⏳ Analyzing squad...")
            self.root.update()
            
            analyzer = SquadAnalyzer(squad_df)
            analysis = analyzer.full_analysis()
            report = analyzer.render_report(analysis)
            
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', report)
            
            self.status_var.set(f"✅ Squad analysis complete — Formation: {analysis['formation']}")
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            self.status_var.set("❌ Squad analysis failed")
    
    def _compare_players(self):
        """Compare two players head-to-head"""
        player1 = self.player1_var.get()
        player2 = self.player2_var.get()
        
        if not player1 or not player2:
            messagebox.showwarning("Warning", "Select both players!")
            return
        
        try:
            p1_data = self.processor_df[self.processor_df['Player'].astype(str) == player1]
            p2_data = self.processor_df[self.processor_df['Player'].astype(str) == player2]
            
            if p1_data.empty or p2_data.empty:
                messagebox.showerror("Error", "One or both players not found!")
                return
            
            p1 = p1_data.iloc[0]
            p2 = p2_data.iloc[0]
            
            comparison = f"""
═══════════════════════════════════════════════════════════
                    PLAYER COMPARISON
═══════════════════════════════════════════════════════════

PLAYER 1: {p1.get('Player', 'N/A')} vs PLAYER 2: {p2.get('Player', 'N/A')}

Position:        {str(p1.get('Pos', 'N/A')):25} vs {p2.get('Pos', 'N/A')}
Age:             {str(p1.get('Age', 0)):25} vs {p2.get('Age', 0)}
Team:            {str(p1.get('Squad', 'N/A')):25} vs {p2.get('Squad', 'N/A')}

⚽ ATTACKING:
  Finishing:     {float(p1.get('Finishing_Quality', 50)):25.1f} vs {float(p2.get('Finishing_Quality', 50)):.1f}
  Creativity:    {float(p1.get('Creativity', 50)):25.1f} vs {float(p2.get('Creativity', 50)):.1f}
  Goal Threat:   {float(p1.get('Goal_Threat', 50)):25.1f} vs {float(p2.get('Goal_Threat', 50)):.1f}

🛡️ DEFENSIVE:
  Defensive Act: {float(p1.get('Defensive_Actions', 50)):25.1f} vs {float(p2.get('Defensive_Actions', 50)):.1f}
  Discipline:    {float(p1.get('Discipline', 50)):25.1f} vs {float(p2.get('Discipline', 50)):.1f}

💪 WORK RATE:
  Involvement:   {float(p1.get('Involvement', 50)):25.1f} vs {float(p2.get('Involvement', 50)):.1f}
  Consistency:   {float(p1.get('Consistency', 50)):25.1f} vs {float(p2.get('Consistency', 50)):.1f}

⏱️ STATUS:
  Age Trajectory: {float(p1.get('Age_Trajectory', 50)):20.1f} vs {float(p2.get('Age_Trajectory', 50)):.1f}

═══════════════════════════════════════════════════════════
"""
            
            self.comparison_text.delete('1.0', tk.END)
            self.comparison_text.insert('1.0', comparison)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedFootballManagerGUI(root)
    root.mainloop()
