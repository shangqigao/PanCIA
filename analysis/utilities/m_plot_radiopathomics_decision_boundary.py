import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from pathlib import Path
import json
import argparse
import warnings
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
warnings.filterwarnings('ignore')

def parse_model_name(folder_name, file_name):
    """Parse model name from folder and file structure"""
    parts = folder_name.split("+")
    radio_model = parts[0]
    patho_model = parts[1] if len(parts) > 1 else ""

    base = file_name.replace("_results.json", "")
    tokens = base.split("_")
    omics = tokens[0]

    radio_aggr = None
    patho_aggr = None

    for t in tokens:
        if t.startswith("radio+"):
            radio_aggr = t.split("+")[1]
        if t.startswith("patho+"):
            patho_aggr = t.split("+")[1]

    if omics == "radiomics":
        model = radio_model
        aggr = radio_aggr
        omics_type = "Radiomics"

    elif omics == "pathomics":
        model = patho_model
        aggr = patho_aggr
        omics_type = "Pathomics"

    elif omics == "radiopathomics":
        model = f"{radio_model}+{patho_model}"
        aggr = radio_aggr
        omics_type = "Radiopathomics"

    else:
        model = folder_name
        aggr = ""
        omics_type = "Other"

    if aggr == "None":
        aggr = "MEAN"

    name = f"{omics_type}: {model} ({aggr})" if aggr else model
    return name, omics_type

def load_model_data(root_parent, target_model_names=None, target_omics=None):
    """Load model data from JSON files"""
    all_models_data = {}
    root_path = Path(root_parent)
    
    for model_dir in root_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        json_files = list(model_dir.glob("*_results.json"))
        
        for json_file in json_files:
            model_name, omics_type = parse_model_name(model_dir.name, json_file.name)

            if target_model_names and model_name not in target_model_names:
                continue
            
            if target_omics and omics_type != target_omics:
                continue
                
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            key = model_name
            all_models_data[key] = {
                'data': data,
                'omics': omics_type,
                'model': model_name
            }
    
    return all_models_data

def prepare_data_for_models(model_data, immune_df, model_keys=None):
    """
    Prepare common data for multiple models with immune subtypes and event status.
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing model data with structure {model_key: {'data': {...}}}
    immune_df : pandas.DataFrame
        DataFrame containing immune subtype information with columns 'ID3' and 'Subtype'
    model_keys : list, optional
        List of model keys to process. If None, uses all available models.
    
    Returns:
    --------
    tuple : (risks_dict, events, durations, subtypes, subject_ids, common_subjects)
        - risks_dict: dict mapping model_key to risk scores array
        - events: numpy array of event status
        - durations: numpy array of survival times
        - subtypes: numpy array of immune subtypes
        - subject_ids: list of subject IDs
        - common_subjects: set of subjects common to all selected models
    """
    import numpy as np
    import pandas as pd
    
    # Validate input
    if not model_data:
        print("Error: model_data is empty")
        return None, None, None, None, None, None
    
    # Determine which models to use
    if model_keys is None:
        model_keys = list(model_data.keys())
    elif isinstance(model_keys, str):
        model_keys = [model_keys]
    
    # Check all models exist
    missing_keys = [key for key in model_keys if key not in model_data]
    if missing_keys:
        print(f"Models not found: {missing_keys}. Available: {list(model_data.keys())}")
        return None, None, None, None, None, None
    
    # Extract risk dictionaries for each model
    risk_dicts = {}
    event_dict = None
    duration_dict = None
    
    for key in model_keys:
        data = model_data[key]['data']
        
        # Validate required columns
        required_cols = ['subject', 'risk', 'event', 'duration']
        missing_cols = [col for col in required_cols if col not in data]
        if missing_cols:
            print(f"Error: Model '{key}' missing columns: {missing_cols}")
            return None, None, None, None, None, None
        
        # Create risk dictionary
        risk_dicts[key] = dict(zip(data['subject'], np.array(data['risk']).flatten()))
        
        # Use first model for event/duration info (assumed consistent across models)
        if event_dict is None:
            event_dict = dict(zip(data['subject'], np.array(data['event']).flatten().astype(int)))
            duration_dict = dict(zip(data['subject'], np.array(data['duration']).flatten()))
    
    # Find subjects common to ALL models
    common_subjects = set(risk_dicts[model_keys[0]].keys())
    for risk_dict in risk_dicts.values():
        common_subjects &= set(risk_dict.keys())
    
    if len(common_subjects) == 0:
        print("No common subjects found across all models!")
        return None, None, None, None, None, None
    
    # Prepare data arrays
    risks_by_model = {key: [] for key in model_keys}
    events = []
    durations = []
    subtypes = []
    subject_ids = []
    
    # Create immune subtype lookup dictionary for efficiency
    if immune_df is not None and 'ID3' in immune_df.columns and 'Subtype' in immune_df.columns:
        immune_dict = dict(zip(immune_df['ID3'], immune_df['Subtype']))
    else:
        immune_dict = {}
        print("Warning: immune_df missing required columns 'ID3' or 'Subtype'")
    
    for subject in common_subjects:
        # Collect risks for each model
        for key in model_keys:
            risks_by_model[key].append(risk_dicts[key][subject])
        
        # Event and duration (use first model's values, verify consistency)
        current_event = event_dict[subject]
        current_duration = duration_dict[subject]
        events.append(current_event)
        durations.append(current_duration)
        subject_ids.append(subject)
        
        # Get immune subtype
        subtype = immune_dict.get(subject, 'Unknown')
        if pd.isna(subtype):
            subtype = 'Unknown'
        subtypes.append(subtype)
    
    # Convert to numpy arrays
    for key in model_keys:
        risks_by_model[key] = np.array(risks_by_model[key])
    
    # Return results
    return (risks_by_model, 
            np.array(events), 
            np.array(durations), 
            np.array(subtypes), 
            subject_ids, 
            common_subjects)

class CrossValidatedSurvivalModels:
    """Perform cross-validation for survival models without data leakage using KFold"""
    
    def __init__(self, risk_dict, events, durations, n_folds=5, random_state=42):
        """
        Parameters:
        -----------
        risk_dict : dict
            Dictionary where keys are risk score names and values are arrays of risk scores
            Example: {'risk_score_1': array1, 'risk_score_2': array2, ...}
        events : array-like
            Binary event indicators (1=event, 0=censored)
        durations : array-like
            Survival times
        n_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.risk_dict = risk_dict
        self.risk_names = list(risk_dict.keys())
        self.events = events
        self.durations = durations
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.cv_results = {
            'CoxPH': {'cindices': [], 'models': [], 'scalers': [], 
                      'predictions': [], 'fold_indices': []},
            'RSF': {'cindices': [], 'models': [], 'scalers': [], 
                    'predictions': [], 'fold_indices': []}
        }

    def _create_feature_dataframe(self):
        """Create DataFrame from risk_dict"""
        X = pd.DataFrame(self.risk_dict)
        return X

    def fit_cross_validate(self):
        """Perform nested cross-validation with hyperparameter tuning for both models"""
        
        X = self._create_feature_dataframe()
        
        # Use standard KFold for outer loop
        outer_kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Define hyperparameter grids
        cox_param_grid = {
            'penalizer': [0.01, 0.05, 0.1, 0.5, 1.0]
        }
        
        rsf_param_grid = {
            'n_estimators': [100],
            'min_samples_split': [10],
            'min_samples_leaf': [5],
            'max_depth': [3, 5, 7, None]
        }
        
        # Store results
        self.cv_results = {
            'CoxPH': {'cindices': [], 'models': [], 'best_params': [], 
                    'predictions': [], 'scalers': [], 'test_indices': []},
            'RSF': {'cindices': [], 'models': [], 'best_params': [], 
                    'predictions': [], 'scalers': [], 'test_indices': []}
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X)):
            print(f"\n{'='*60}")
            print(f"Outer Fold {fold_idx + 1}/{self.n_folds}")
            print(f"{'='*60}")
            print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            print(f"  Train events: {np.sum(self.events[train_idx])} ({np.mean(self.events[train_idx])*100:.1f}%)")
            print(f"  Test events: {np.sum(self.events[test_idx])} ({np.mean(self.events[test_idx])*100:.1f}%)")
            
            # Split into outer train and test
            X_outer_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_outer_train = self.events[train_idx]
            y_test = self.events[test_idx]
            durations_outer_train = self.durations[train_idx]
            durations_test = self.durations[test_idx]
            
            # Standardize features based on outer training data
            scaler_outer = StandardScaler()
            X_outer_train_scaled = scaler_outer.fit_transform(X_outer_train)
            X_test_scaled = scaler_outer.transform(X_test)
            
            # Convert back to DataFrame for CoxPH
            X_outer_train_scaled_df = pd.DataFrame(X_outer_train_scaled, 
                                                columns=X_outer_train.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, 
                                        columns=X_test.columns)
            
            # ===== COX PH WITH INNER CV =====
            print(f"\n  Tuning CoxPH with inner CV...")
            best_cox_params, best_cox_model, cox_inner_results = self._tune_coxph(
                X_outer_train_scaled_df, y_outer_train, durations_outer_train, 
                cox_param_grid
            )
            
            # Evaluate best CoxPH on outer test set
            cox_pred_test = best_cox_model.predict_partial_hazard(X_test_scaled_df).values
            cindex_cox = concordance_index(durations_test, -cox_pred_test, y_test)
            
            print(f"  Best CoxPH params: {best_cox_params}")
            print(f"  CoxPH Test C-index: {cindex_cox:.3f}")
            
            # ===== RSF WITH INNER CV =====
            print(f"\n  Tuning RSF with inner CV...")
            best_rsf_params, best_rsf_model, rsf_inner_results = self._tune_rsf(
                X_outer_train_scaled, y_outer_train, durations_outer_train,
                rsf_param_grid
            )
            
            # Evaluate best RSF on outer test set
            rsf_pred_test = best_rsf_model.predict(X_test_scaled)
            cindex_rsf = concordance_index(durations_test, -rsf_pred_test, y_test)
            
            print(f"  Best RSF params: {best_rsf_params}")
            print(f"  RSF Test C-index: {cindex_rsf:.3f}")
            
            # Store predictions with all risk scores
            test_risk_values = {name: X_test[name].values for name in self.risk_names}
            
            # Store results
            self.cv_results['CoxPH']['cindices'].append(cindex_cox)
            self.cv_results['CoxPH']['models'].append(best_cox_model)
            self.cv_results['CoxPH']['best_params'].append(best_cox_params)
            self.cv_results['CoxPH']['scalers'].append(scaler_outer)
            self.cv_results['CoxPH']['test_indices'].append(test_idx)
            self.cv_results['CoxPH']['predictions'].append({
                'test_idx': test_idx,
                'predictions': cox_pred_test,
                'true_events': y_test,
                'true_durations': durations_test,
                **{f'risk_{name}': test_risk_values[name] for name in self.risk_names}
            })
            
            self.cv_results['RSF']['cindices'].append(cindex_rsf)
            self.cv_results['RSF']['models'].append(best_rsf_model)
            self.cv_results['RSF']['best_params'].append(best_rsf_params)
            self.cv_results['RSF']['scalers'].append(scaler_outer)
            self.cv_results['RSF']['test_indices'].append(test_idx)
            self.cv_results['RSF']['predictions'].append({
                'test_idx': test_idx,
                'predictions': rsf_pred_test,
                'true_events': y_test,
                'true_durations': durations_test,
                **{f'risk_{name}': test_risk_values[name] for name in self.risk_names}
            })
        
        # Calculate summary statistics
        self._print_nested_cv_summary()
        
        return self.cv_results

    def _tune_coxph(self, X_train, y_train, durations_train, param_grid):
        """Inner CV loop for CoxPH hyperparameter tuning"""
        inner_kf = KFold(n_splits=min(5, len(X_train)-1), shuffle=True, 
                        random_state=self.random_state)
        
        results = []
        for penalizer in param_grid['penalizer']:
            fold_scores = []
            
            for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
                X_inner_train = X_train.iloc[inner_train_idx]
                X_inner_val = X_train.iloc[inner_val_idx]
                y_inner_train = y_train[inner_train_idx]
                y_inner_val = y_train[inner_val_idx]
                durations_inner_train = durations_train[inner_train_idx]
                durations_inner_val = durations_train[inner_val_idx]
                
                # Train CoxPH
                cox_data = X_inner_train.copy()
                cox_data['event'] = y_inner_train
                cox_data['time'] = durations_inner_train
                
                coxph = CoxPHFitter(penalizer=penalizer)
                coxph.fit(cox_data, duration_col='time', event_col='event')
                
                # Evaluate
                cox_pred_val = coxph.predict_partial_hazard(X_inner_val).values
                cindex = concordance_index(durations_inner_val, -cox_pred_val, y_inner_val)
                fold_scores.append(cindex)
            
            avg_score = np.mean(fold_scores)
            results.append({
                'penalizer': penalizer,
                'mean_cindex': avg_score,
                'std_cindex': np.std(fold_scores),
                'fold_scores': fold_scores
            })
            
            print(f"    penalizer={penalizer}: {avg_score:.3f} ± {np.std(fold_scores):.3f}")
        
        # Select best parameters
        best_params = max(results, key=lambda x: x['mean_cindex'])
        best_penalizer = best_params['penalizer']
        
        # Train final model on all training data
        cox_data_full = X_train.copy()
        cox_data_full['event'] = y_train
        cox_data_full['time'] = durations_train
        best_model = CoxPHFitter(penalizer=best_penalizer)
        best_model.fit(cox_data_full, duration_col='time', event_col='event')
        
        return {'penalizer': best_penalizer}, best_model, results

    def _tune_rsf(self, X_train, y_train, durations_train, param_grid):
        """Inner CV loop for RSF hyperparameter tuning"""
        from sklearn.model_selection import ParameterGrid
        
        inner_kf = KFold(n_splits=min(5, len(X_train)-1), shuffle=True, 
                        random_state=self.random_state)
        
        # Convert to structured array for RSF
        train_survival = np.array([(e, d) for e, d in zip(y_train, durations_train)],
                                dtype=[('event', bool), ('time', float)])
        
        results = []
        param_grid_list = list(ParameterGrid(param_grid))
        
        for params in param_grid_list:
            fold_scores = []
            
            for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
                X_inner_train = X_train[inner_train_idx]
                X_inner_val = X_train[inner_val_idx]
                survival_inner_train = train_survival[inner_train_idx]
                y_inner_val = y_train[inner_val_idx]
                durations_inner_val = durations_train[inner_val_idx]
                
                # Train RSF
                rsf = RandomSurvivalForest(
                    n_estimators=params['n_estimators'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_depth=params['max_depth'],
                    random_state=self.random_state,
                    n_jobs=-1
                )
                rsf.fit(X_inner_train, survival_inner_train)
                
                # Evaluate
                rsf_pred_val = rsf.predict(X_inner_val)
                cindex = concordance_index(durations_inner_val, -rsf_pred_val, y_inner_val)
                fold_scores.append(cindex)
            
            avg_score = np.mean(fold_scores)
            results.append({
                'params': params,
                'mean_cindex': avg_score,
                'std_cindex': np.std(fold_scores),
                'fold_scores': fold_scores
            })
            
            print(f"    {params}: {avg_score:.3f} ± {np.std(fold_scores):.3f}")
        
        # Select best parameters
        best_result = max(results, key=lambda x: x['mean_cindex'])
        best_params = best_result['params']
        
        # Train final model on all training data
        best_model = RandomSurvivalForest(
            n_estimators=best_params['n_estimators'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_depth=best_params['max_depth'],
            random_state=self.random_state,
            n_jobs=-1
        )
        best_model.fit(X_train, train_survival)
        
        return best_params, best_model, results

    def _print_nested_cv_summary(self):
        """Print detailed nested cross-validation results"""
        print("\n" + "="*60)
        print("NESTED CROSS-VALIDATION RESULTS SUMMARY")
        print("="*60)
        
        for model_name in ['CoxPH', 'RSF']:
            cindices = self.cv_results[model_name]['cindices']
            best_params_list = self.cv_results[model_name]['best_params']
            
            print(f"\n{model_name}:")
            print(f"  Outer CV Performance:")
            print(f"    Mean C-index: {np.mean(cindices):.3f} ± {np.std(cindices):.3f}")
            print(f"    95% CI: [{np.percentile(cindices, 2.5):.3f}, {np.percentile(cindices, 97.5):.3f}]")
            print(f"    Individual folds: {[f'{x:.3f}' for x in cindices]}")
            
            print(f"  Hyperparameter Stability:")
            # Count parameter frequencies
            if model_name == 'CoxPH':
                penalizers = [p['penalizer'] for p in best_params_list]
                unique, counts = np.unique(penalizers, return_counts=True)
                for param, count in zip(unique, counts):
                    print(f"    penalizer={param}: selected in {count}/{len(best_params_list)} folds ({count/len(best_params_list)*100:.0f}%)")
            else:
                # For RSF, show most common parameter combinations
                from collections import Counter
                param_strings = [str(p) for p in best_params_list]
                common_params = Counter(param_strings).most_common(3)
                for params_str, count in common_params:
                    print(f"    {params_str}: selected in {count}/{len(best_params_list)} folds ({count/len(best_params_list)*100:.0f}%)")
        
        # Compare models
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        cox_scores = self.cv_results['CoxPH']['cindices']
        rsf_scores = self.cv_results['RSF']['cindices']
        
        # Paired t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(cox_scores, rsf_scores)
        
        print(f"CoxPH:  {np.mean(cox_scores):.3f} ± {np.std(cox_scores):.3f}")
        print(f"RSF:    {np.mean(rsf_scores):.3f} ± {np.std(rsf_scores):.3f}")
        print(f"Difference: {np.mean(rsf_scores) - np.mean(cox_scores):.3f}")
        print(f"Paired t-test p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            winner = "RSF" if np.mean(rsf_scores) > np.mean(cox_scores) else "CoxPH"
            print(f"Significant difference (p<0.05): {winner} performs better")
        else:
            print("No significant difference between models")
    
    def get_subtype_cindices_cv(self, subtypes):
        """Calculate subtype-specific C-indices using cross-validated predictions"""
        
        subtype_cindices = {'CoxPH': {}, 'RSF': {}}
        subtype_details = {'CoxPH': {}, 'RSF': {}}
        
        for model_name in ['CoxPH', 'RSF']:
            # Collect all out-of-fold predictions
            all_predictions = []
            all_events = []
            all_durations = []
            all_subtypes = []
            all_risk_values = {name: [] for name in self.risk_names}
            
            for fold_result in self.cv_results[model_name]['predictions']:
                test_idx = fold_result['test_idx']
                all_predictions.extend(fold_result['predictions'])
                all_events.extend(fold_result['true_events'])
                all_durations.extend(fold_result['true_durations'])
                all_subtypes.extend(subtypes[test_idx])
                
                # Collect all risk values
                for name in self.risk_names:
                    all_risk_values[name].extend(fold_result[f'risk_{name}'])
            
            all_predictions = np.array(all_predictions)
            all_events = np.array(all_events)
            all_durations = np.array(all_durations)
            all_subtypes = np.array(all_subtypes)
            
            for name in self.risk_names:
                all_risk_values[name] = np.array(all_risk_values[name])
            
            # Calculate C-index for each subtype
            unique_subtypes = np.unique(all_subtypes)
            for subtype in unique_subtypes:
                if subtype == 'Unknown':
                    continue
                    
                mask = all_subtypes == subtype
                n_samples = np.sum(mask)
                if n_samples >= 5:  # Minimum samples for reliable estimate
                    try:
                        cindex = concordance_index(all_durations[mask], -all_predictions[mask], all_events[mask])
                        subtype_cindices[model_name][subtype] = cindex
                        
                        # Build details dictionary
                        details = {
                            'n_samples': n_samples,
                            'n_events': np.sum(all_events[mask]),
                            'mean_risk': np.mean(all_predictions[mask])
                        }
                        
                        # Add mean for each risk score
                        for name in self.risk_names:
                            details[f'mean_risk_{name}'] = np.mean(all_risk_values[name][mask])
                        
                        subtype_details[model_name][subtype] = details
                        
                    except Exception as e:
                        print(f"Warning: Could not compute C-index for {subtype} ({model_name}): {e}")
                        subtype_cindices[model_name][subtype] = np.nan
                else:
                    print(f"Warning: {subtype} has only {n_samples} samples, skipping C-index calculation")
                    subtype_cindices[model_name][subtype] = np.nan
        
        return subtype_cindices, subtype_details
    
    def get_aggregated_predictions(self, subtypes):
        """Get aggregated out-of-fold predictions for all samples"""
        
        aggregated_preds = {'CoxPH': {'predictions': None, 'events': None, 
                                    'durations': None, 'subtypes': None,
                                    'fold_ids': None, 'risk_dict': None},
                        'RSF': {'predictions': None, 'events': None,
                                'durations': None, 'subtypes': None,
                                'fold_ids': None, 'risk_dict': None}}
        
        for model_name in ['CoxPH', 'RSF']:
            # Initialize arrays
            n_samples = len(self.events)
            all_preds = np.zeros(n_samples)
            all_events = np.zeros(n_samples)
            all_durations = np.zeros(n_samples)
            all_subtypes = np.array([''] * n_samples, dtype=object)
            all_fold_ids = np.zeros(n_samples, dtype=int)
            
            # Initialize risk_dict as a dictionary of arrays
            risk_dict = {name: np.zeros(n_samples) for name in self.risk_names}
            
            # Fill with out-of-fold predictions
            for fold_idx, fold_result in enumerate(self.cv_results[model_name]['predictions']):
                test_idx = fold_result['test_idx']
                all_preds[test_idx] = fold_result['predictions']
                all_events[test_idx] = fold_result['true_events']
                all_durations[test_idx] = fold_result['true_durations']
                all_subtypes[test_idx] = subtypes[test_idx]
                all_fold_ids[test_idx] = fold_idx
                
                # Fill risk scores in risk_dict
                for name in self.risk_names:
                    risk_dict[name][test_idx] = fold_result[f'risk_{name}']
            
            aggregated_preds[model_name]['predictions'] = all_preds
            aggregated_preds[model_name]['events'] = all_events
            aggregated_preds[model_name]['durations'] = all_durations
            aggregated_preds[model_name]['subtypes'] = all_subtypes
            aggregated_preds[model_name]['fold_ids'] = all_fold_ids
            aggregated_preds[model_name]['risk_dict'] = risk_dict
        
        return aggregated_preds

def plot_km_curves_risk_groups(predictions, events, durations, subtypes, model_name, 
                                output_root, n_groups=2, save=True):
    """
    Plot Kaplan-Meier curves for risk groups (2 or 3 groups)
    
    Parameters:
    -----------
    predictions : array-like
        Predicted risk scores
    events : array-like
        Event indicators (1 for event, 0 for censored)
    durations : array-like
        Survival times
    subtypes : array-like
        Immune subtype labels
    model_name : str
        Name of the model (CoxPH or RSF)
    output_root : Path
        Output directory path
    n_groups : int
        Number of risk groups (2 for low/high, 3 for low/middle/high)
    save : bool
        Whether to save the plots
    """
    
    # Get unique subtypes (excluding Unknown)
    unique_subtypes = np.unique(subtypes)
    unique_subtypes = [s for s in unique_subtypes if s != 'Unknown']
    
    # Create figure with subplots: one for overall + one for each subtype
    n_subplots = 1 + len(unique_subtypes)
    n_cols = min(3, n_subplots)
    n_rows = (n_subplots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_subplots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Define risk group labels based on n_groups
    if n_groups == 2:
        risk_labels = ['Low Risk', 'High Risk']
        colors = ['blue', 'red']
        linestyles = ['-', '-']
    else:  # n_groups == 3
        risk_labels = ['Low Risk', 'Middle Risk', 'High Risk']
        colors = ['green', 'orange', 'red']
        linestyles = ['-', '-', '-']
    
    # Plot overall survival curves
    ax = axes[0]
    plot_km_for_group(ax, predictions, events, durations, None, risk_labels, 
                     colors, linestyles, n_groups, f"Overall Survival - {model_name}")
    
    # Plot subtype-specific survival curves
    for idx, subtype in enumerate(unique_subtypes):
        ax = axes[idx + 1]
        mask = subtypes == subtype
        plot_km_for_group(ax, predictions[mask], events[mask], durations[mask], 
                         subtype, risk_labels, colors, linestyles, n_groups,
                         f"{subtype} - {model_name}")
    
    # Hide any unused subplots
    for idx in range(len(unique_subtypes) + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Kaplan-Meier Curves by Risk Group ({n_groups}-Group Stratification)\n{model_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"KM_curves_{n_groups}groups_{model_name}.png"
        plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        print(f"KM curves saved to {output_path / filename}")
    
    plt.close()
    return fig

def plot_km_for_group(ax, predictions, events, durations, group_name, risk_labels, 
                     colors, linestyles, n_groups, title):
    """Helper function to plot KM curves for a specific group"""
    
    # Determine risk group thresholds
    if n_groups == 2:
        # Split by median
        median_risk = np.median(predictions)
        risk_groups = (predictions > median_risk).astype(int)
        unique_groups = [0, 1]
    else:  # n_groups == 3
        # Split by tertiles
        tertiles = np.percentile(predictions, [33.33, 66.67])
        risk_groups = np.zeros(len(predictions), dtype=int)
        risk_groups[predictions > tertiles[1]] = 2
        risk_groups[(predictions > tertiles[0]) & (predictions <= tertiles[1])] = 1
        unique_groups = [0, 1, 2]
    
    # Fit and plot KM curves for each risk group
    kmf = KaplanMeierFitter()
    
    for group_idx in unique_groups:
        mask = risk_groups == group_idx
        if np.sum(mask) == 0:
            continue
            
        kmf.fit(durations[mask], events[mask], label=f"{risk_labels[group_idx]} (n={np.sum(mask)})")
        kmf.plot_survival_function(ax=ax, color=colors[group_idx], 
                                  linestyle=linestyles[group_idx], linewidth=2)
    
    # Perform log-rank test between groups
    if n_groups == 2:
        group0_mask = risk_groups == 0
        group1_mask = risk_groups == 1
        if np.sum(group0_mask) > 0 and np.sum(group1_mask) > 0:
            results = logrank_test(durations[group0_mask], durations[group1_mask],
                                  events[group0_mask], events[group1_mask])
            p_value = results.p_value
            p_text = f'Log-rank p = {p_value:.4f}' if p_value >= 0.0001 else 'Log-rank p < 0.0001'
            ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:  # n_groups == 3
        # Perform pairwise log-rank tests
        group0_mask = risk_groups == 0
        group1_mask = risk_groups == 1
        group2_mask = risk_groups == 2
        
        p_values = []
        comparisons = []
        
        if np.sum(group0_mask) > 0 and np.sum(group1_mask) > 0:
            res01 = logrank_test(durations[group0_mask], durations[group1_mask],
                                events[group0_mask], events[group1_mask])
            p_values.append(res01.p_value)
            comparisons.append('Low vs Mid')
        
        if np.sum(group0_mask) > 0 and np.sum(group2_mask) > 0:
            res02 = logrank_test(durations[group0_mask], durations[group2_mask],
                                events[group0_mask], events[group2_mask])
            p_values.append(res02.p_value)
            comparisons.append('Low vs High')
        
        if np.sum(group1_mask) > 0 and np.sum(group2_mask) > 0:
            res12 = logrank_test(durations[group1_mask], durations[group2_mask],
                                events[group1_mask], events[group2_mask])
            p_values.append(res12.p_value)
            comparisons.append('Mid vs High')
        
        # Display p-values
        p_text = '\n'.join([f'{comp}: p = {p:.4f}' if p >= 0.0001 else f'{comp}: p < 0.0001' 
                           for comp, p in zip(comparisons, p_values)])
        ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Add event counts
    n_events = np.sum(events)
    n_censored = len(events) - n_events
    ax.text(0.95, 0.95, f'Total: n={len(events)}\nEvents: {int(n_events)}\nCensored: {int(n_censored)}', 
           transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def plot_survival_decision_boundary_cv(risks1, risks2, subtypes,
                                       cv_results, aggregated_preds, subtype_details,
                                       model1_name, model2_name,
                                       output_root, save=True):
    """Create scatter plots using cross-validated predictions"""
    
    # Count frequency of each subtype and get top 4
    subtype_counts = pd.Series(subtypes).value_counts()
    top4_subtypes = subtype_counts.head(4).index.tolist()
    
    print(f"\nTop 4 immune subtypes: {top4_subtypes}")
    print(f"Frequencies: {subtype_counts.head(4).to_dict()}")
    
    # Calculate global medians for the dashed lines
    global_median_risks1 = np.median(risks1)
    global_median_risks2 = np.median(risks2)
    
    print(f"Global median {model1_name} risk score: {global_median_risks1:.3f}")
    print(f"Global median {model2_name} risk score: {global_median_risks2:.3f}")
    
    # Get subtype-specific C-indices from CV
    subtype_cindices_cv, _ = cv_results.get_subtype_cindices_cv(subtypes)
    
    # List to store all figures
    figs = []
    
    # ========================================================================
    # FIRST: Plot for ALL SAMPLES (all subtypes combined)
    # ========================================================================
    print("\n" + "="*60)
    print("Creating plot for ALL SAMPLES (all subtypes combined)")
    print("="*60)
    
    fig_all, axes_all = plt.subplots(1, 2, figsize=(18, 7))
    
    for model_idx, model_name in enumerate(['CoxPH', 'RSF']):
        ax = axes_all[model_idx]
        
        # Get aggregated predictions for this model
        pred_data = aggregated_preds[model_name]
        
        # Use ALL samples (no subtype filtering)
        risks1_all = pred_data['risk_dict'][model1_name]
        risks2_all = pred_data['risk_dict'][model2_name]
        events_all = pred_data['events']
        durations_all = pred_data['durations']
        predictions_all = pred_data['predictions']
        
        # Get global CV statistics
        global_cindex_mean = np.mean(cv_results.cv_results[model_name]['cindices'])
        global_cindex_std = np.std(cv_results.cv_results[model_name]['cindices'])
        
        # Calculate overall survival rate
        n_total_all = len(events_all)
        n_events_all = np.sum(events_all)
        survival_rate_all = (1 - n_events_all/n_total_all) * 100 if n_total_all > 0 else 0
        
        # Create mesh grid for decision boundary
        x_min, x_max = risks1.min() - 0.1, risks1.max() + 0.1
        y_min, y_max = risks2.min() - 0.1, risks2.max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                            np.linspace(y_min, y_max, 100))
        
        # Predict risk surface using all models and average
        risk_grid_sum = np.zeros(xx.shape)
        for fold_idx in range(len(cv_results.cv_results[model_name]['models'])):
            fold_model = cv_results.cv_results[model_name]['models'][fold_idx]
            fold_scaler = cv_results.cv_results[model_name]['scalers'][fold_idx]
            
            grid = np.c_[xx.ravel(), yy.ravel()]
            grid_scaled = fold_scaler.transform(grid)
            grid_df = pd.DataFrame(grid_scaled, columns=[model1_name, model2_name])
            
            if model_name == 'CoxPH':
                risk_grid_fold = fold_model.predict_partial_hazard(grid_df)
                risk_grid_fold = risk_grid_fold.values.reshape(xx.shape)
            else:
                risk_grid_fold = fold_model.predict(grid_scaled)
                risk_grid_fold = risk_grid_fold.reshape(xx.shape)
            
            risk_grid_sum += risk_grid_fold
        
        risk_grid = risk_grid_sum / len(cv_results.cv_results[model_name]['models'])
        
        # Plot risk surface
        contour = ax.contourf(xx, yy, risk_grid, levels=20, cmap='RdBu_r', alpha=0.3)
        median_risk = np.median(risk_grid)
        ax.contour(xx, yy, risk_grid, levels=[median_risk], 
                  colors='black', linewidths=2, linestyles='-', 
                  label='Median Risk Boundary')
        
        # Add dashed lines for median risk scores
        ax.axvline(x=global_median_risks1, color='yellow', linestyle='--', 
                  linewidth=2, alpha=0.7, 
                  label=f'Median: {global_median_risks1:.3f}')
        
        ax.axhline(y=global_median_risks2, color='green', linestyle='--', 
                  linewidth=2, alpha=0.7,
                  label=f'Median: {global_median_risks2:.3f}')
        
        # Plot data points - consistent with subtype plots: colored by event status
        event_colors = {1: 'red', 0: 'blue'}
        for event_val in [0, 1]:
            event_mask = events_all == event_val
            if np.sum(event_mask) > 0:
                size = 80 if event_val == 1 else 60
                alpha = 0.8 if event_val == 1 else 0.6
                marker = 'o' if model_name == 'CoxPH' else 's'
                label = f"{'Event' if event_val==1 else 'Censored'} (n={np.sum(event_mask)})"
                
                ax.scatter(risks1_all[event_mask], risks2_all[event_mask],
                          c=event_colors[event_val], marker=marker, s=size,
                          alpha=alpha, edgecolors='black', linewidth=1.5,
                          label=label)
        
        ax.set_xlabel(f'{model1_name}\nRisk Score', fontsize=13)
        ax.set_ylabel(f'{model2_name}\nRisk Score', fontsize=13)
        
        # Create title with CV metrics
        title = (f'{model_name} - ALL SAMPLES\n'
                f'Global C-index (CV): {global_cindex_mean:.3f} ± {global_cindex_std:.3f}\n'
                f'(n={n_total_all}, events={n_events_all}, survival={survival_rate_all:.1f}%)')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Add colorbar for all samples figure
    cbar_ax_all = fig_all.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar_all = fig_all.colorbar(contour, cax=cbar_ax_all)
    cbar_all.set_label('Predicted Risk Score\n(Higher = Worse Prognosis)', fontsize=11)
    
    plt.subplots_adjust(right=0.85, wspace=0.3)
    fig_all.tight_layout(rect=[0, 0, 0.85, 0.98])
    
    if save:
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        filename_all = f"survival_boundary_CV_ALL_SAMPLES_{model1_name.split()[0]}_vs_{model2_name.split()[0]}.png"
        fig_all.savefig(output_path / filename_all, dpi=150, bbox_inches='tight')
        print(f"Plot saved for ALL SAMPLES -> {output_path / filename_all}")
    
    plt.close()
    figs.append(fig_all)
    
    # ========================================================================
    # SECOND: Plot for each top subtype separately
    # ========================================================================
    for subtype in top4_subtypes:
        print(f"\nCreating plot for subtype: {subtype}")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        for model_idx, model_name in enumerate(['CoxPH', 'RSF']):
            ax = axes[model_idx]
            
            # Get aggregated predictions for this model
            pred_data = aggregated_preds[model_name]
            
            # Filter for current subtype
            subtype_mask = pred_data['subtypes'] == subtype
            risks1_sub = pred_data['risk_dict'][model1_name][subtype_mask]
            risks2_sub = pred_data['risk_dict'][model2_name][subtype_mask]
            events_sub = pred_data['events'][subtype_mask]
            durations_sub = pred_data['durations'][subtype_mask]
            predictions_sub = pred_data['predictions'][subtype_mask]
            
            # Calculate subtype-specific C-index (already from CV)
            cindex_sub = subtype_cindices_cv[model_name].get(subtype, np.nan)
            
            # Get details for this subtype
            details = subtype_details[model_name].get(subtype, {})
            n_subtype = details.get('n_samples', len(events_sub))
            n_events_sub = details.get('n_events', np.sum(events_sub))
            
            # Get global CV statistics
            global_cindex_mean = np.mean(cv_results.cv_results[model_name]['cindices'])
            global_cindex_std = np.std(cv_results.cv_results[model_name]['cindices'])
            
            # Create mesh grid for decision boundary (using same limits as before)
            x_min, x_max = risks1.min() - 0.1, risks1.max() + 0.1
            y_min, y_max = risks2.min() - 0.1, risks2.max() + 0.1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                                np.linspace(y_min, y_max, 100))
            
            # Predict risk surface using all models and average
            risk_grid_sum = np.zeros(xx.shape)
            for fold_idx in range(len(cv_results.cv_results[model_name]['models'])):
                fold_model = cv_results.cv_results[model_name]['models'][fold_idx]
                fold_scaler = cv_results.cv_results[model_name]['scalers'][fold_idx]
                
                grid = np.c_[xx.ravel(), yy.ravel()]
                grid_scaled = fold_scaler.transform(grid)
                grid_df = pd.DataFrame(grid_scaled, columns=[model1_name, model2_name])
                
                if model_name == 'CoxPH':
                    risk_grid_fold = fold_model.predict_partial_hazard(grid_df)
                    risk_grid_fold = risk_grid_fold.values.reshape(xx.shape)
                else:
                    risk_grid_fold = fold_model.predict(grid_scaled)
                    risk_grid_fold = risk_grid_fold.reshape(xx.shape)
                
                risk_grid_sum += risk_grid_fold
            
            risk_grid = risk_grid_sum / len(cv_results.cv_results[model_name]['models'])
            
            # Plot risk surface
            contour = ax.contourf(xx, yy, risk_grid, levels=20, cmap='RdBu_r', alpha=0.3)
            median_risk = np.median(risk_grid)
            ax.contour(xx, yy, risk_grid, levels=[median_risk], 
                      colors='black', linewidths=2, linestyles='-', 
                      label='Median Risk Boundary')
            
            # Add dashed lines for median risk scores
            ax.axvline(x=global_median_risks1, color='yellow', linestyle='--', 
                    linewidth=2, alpha=0.7, 
                    label=f'Median: {global_median_risks1:.3f}')
            
            ax.axhline(y=global_median_risks2, color='green', linestyle='--', 
                    linewidth=2, alpha=0.7,
                    label=f'Median: {global_median_risks2:.3f}')
            
            # Plot data points - colored by event status
            event_colors = {1: 'red', 0: 'blue'}
            for event_val in [0, 1]:
                event_mask = events_sub == event_val
                if np.sum(event_mask) > 0:
                    size = 80 if event_val == 1 else 60
                    alpha = 0.8 if event_val == 1 else 0.6
                    marker = 'o' if model_name == 'CoxPH' else 's'
                    label = f"{'Event' if event_val==1 else 'Censored'} (n={np.sum(event_mask)})"
                    
                    ax.scatter(risks1_sub[event_mask], risks2_sub[event_mask],
                              c=event_colors[event_val], marker=marker, s=size,
                              alpha=alpha, edgecolors='black', linewidth=1.5,
                              label=label)
            
            ax.set_xlabel(f'{model1_name}\nRisk Score', fontsize=13)
            ax.set_ylabel(f'{model2_name}\nRisk Score', fontsize=13)
            
            # Calculate survival rate
            survival_rate = (1 - n_events_sub/n_subtype) * 100 if n_subtype > 0 else 0
            
            # Create title with CV metrics
            title = (f'{model_name} - {subtype}\n'
                    f'Global C-index (CV): {global_cindex_mean:.3f} ± {global_cindex_std:.3f}\n'
                    f'Subtype C-index (CV): {cindex_sub:.3f} ')
            
            if not np.isnan(cindex_sub):
                # Add comparison to global performance
                diff = cindex_sub - global_cindex_mean
                comparison = f"({'better' if diff > 0 else 'worse' if diff < 0 else 'similar'})"
                title += comparison
            
            title += f'\n(n={n_subtype}, events={n_events_sub}, survival={survival_rate:.1f}%)'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label('Predicted Risk Score\n(Higher = Worse Prognosis)', fontsize=11)
        
        plt.subplots_adjust(right=0.85, wspace=0.3)
        plt.tight_layout(rect=[0, 0, 0.85, 0.98])
        
        if save:
            output_path = Path(output_root)
            output_path.mkdir(parents=True, exist_ok=True)
            clean_subtype = str(subtype).replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            filename = f"survival_boundary_CV_{clean_subtype}_{model1_name.split()[0]}_vs_{model2_name.split()[0]}.png"
            plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved for {subtype} -> {output_path / filename}")
        
        plt.close()
        figs.append(fig)
    
    # Print summary of subtype-specific C-indices
    print("\n" + "="*60)
    print("Cross-Validated Subtype-Specific C-index Summary (KFold):")
    print("="*60)
    for subtype in top4_subtypes:
        cox_cindex = subtype_cindices_cv['CoxPH'].get(subtype, np.nan)
        rsf_cindex = subtype_cindices_cv['RSF'].get(subtype, np.nan)
        n_samples = subtype_details['CoxPH'].get(subtype, {}).get('n_samples', 0)
        print(f"{subtype:20s} | n={n_samples:3d} | CoxPH: {cox_cindex:.3f} | RSF: {rsf_cindex:.3f}")
    print("="*60)
    
    return figs

def plot_cv_performance_summary(cv_results, model1_name, model2_name, output_root, save=True):
    """Plot cross-validation performance summary"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Individual fold C-indices
    ax1 = axes[0, 0]
    folds = np.arange(1, cv_results.n_folds + 1)
    cox_cindices = cv_results.cv_results['CoxPH']['cindices']
    rsf_cindices = cv_results.cv_results['RSF']['cindices']
    
    ax1.plot(folds, cox_cindices, 'o-', label='CoxPH', linewidth=2, markersize=8, color='blue')
    ax1.plot(folds, rsf_cindices, 's-', label='RSF', linewidth=2, markersize=8, color='red')
    ax1.axhline(y=np.mean(cox_cindices), color='blue', linestyle='--', alpha=0.5, label=f"CoxPH mean: {np.mean(cox_cindices):.3f}")
    ax1.axhline(y=np.mean(rsf_cindices), color='red', linestyle='--', alpha=0.5, label=f"RSF mean: {np.mean(rsf_cindices):.3f}")
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('C-index', fontsize=12)
    ax1.set_title('Cross-Validation Performance per Fold', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(folds)
    ax1.set_ylim([0.4, 1.0])
    
    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    bp = ax2.boxplot([cox_cindices, rsf_cindices], labels=['CoxPH', 'RSF'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    bp['medians'][0].set_color('darkblue')
    bp['medians'][1].set_color('darkred')
    ax2.set_ylabel('C-index', fontsize=12)
    ax2.set_title('Performance Distribution Across Folds', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.4, 1.0])
    
    # Add mean values as text
    ax2.text(1, np.mean(cox_cindices) + 0.1, f'Mean: {np.mean(cox_cindices):.3f}\nStd: {np.std(cox_cindices):.3f}', 
             ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    ax2.text(2, np.mean(rsf_cindices) + 0.1, f'Mean: {np.mean(rsf_cindices):.3f}\nStd: {np.std(rsf_cindices):.3f}', 
             ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
    
    # Plot 3: C-index histogram comparison
    ax3 = axes[1, 0]
    ax3.hist(cox_cindices, bins=5, alpha=0.5, label='CoxPH', color='blue', edgecolor='black')
    ax3.hist(rsf_cindices, bins=5, alpha=0.5, label='RSF', color='red', edgecolor='black')
    ax3.set_xlabel('C-index', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of C-indices Across Folds', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Performance comparison with confidence intervals
    ax4 = axes[1, 1]
    models = ['CoxPH', 'RSF']
    means = [np.mean(cox_cindices), np.mean(rsf_cindices)]
    stds = [np.std(cox_cindices), np.std(rsf_cindices)]
    
    x_pos = np.arange(len(models))
    ax4.bar(x_pos, means, yerr=stds, capsize=10, color=['lightblue', 'lightcoral'], 
            edgecolor='black', linewidth=2, alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models)
    ax4.set_ylabel('C-index', fontsize=12)
    ax4.set_title('Mean Performance with 1 Std Dev', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0.4, 1.0])
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax4.text(i, mean + std + 0.02, f'{mean:.3f}\n±{std:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Cross-Validation Performance Summary (5-fold KFold)\n{model1_name} vs {model2_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        output_path = Path(output_root)
        filename = f"cv_performance_summary_{model1_name.split()[0]}_vs_{model2_name.split()[0]}.png"
        plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        print(f"CV performance summary saved to {output_path / filename}")
    
    plt.close()
    return fig

def plot_risk_group_km_curves(aggregated_preds, model_name, output_root, save=True):
    """
    Plot Kaplan-Meier curves for low-risk and high-risk groups (2 groups)
    Figure includes:
    1. Overall population
    2. Combined plot with all top 4 subtypes
    3-6. Individual plots for each top 4 subtype
    """
    
    predictions = aggregated_preds[model_name]['predictions']
    events = aggregated_preds[model_name]['events']
    durations = aggregated_preds[model_name]['durations']
    subtypes = aggregated_preds[model_name]['subtypes']
    
    # Get top 4 subtypes (excluding Unknown)
    subtype_counts = pd.Series(subtypes).value_counts()
    top4_subtypes = [s for s in subtype_counts.head(4).index if s != 'Unknown']
    
    print(f"\n{model_name} - Top 4 subtypes for KM curves: {top4_subtypes}")
    print(f"Subtype frequencies: {subtype_counts[top4_subtypes].to_dict()}")
    
    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Colors for risk groups
    colors = ['#2E86AB', '#A23B72']  # Blue for low risk, purple/red for high risk
    
    # Colors for different subtypes (for the combined plot)
    subtype_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # Red, Blue, Green, Purple
    
    # Plot 1: Overall
    ax = axes[0]
    plot_km_risk_groups(ax, predictions, events, durations, None, colors, 
                        "Overall Population", model_name)
    
    # Plot 2: Combined plot with all top 4 subtypes (4 curves showing risk group stratification)
    ax = axes[1]
    plot_combined_subtypes_km(ax, predictions, events, durations, subtypes, 
                              top4_subtypes, subtype_colors, model_name)
    
    # Plots 3-6: Individual plots for each top 4 subtype
    for idx, subtype in enumerate(top4_subtypes):
        ax = axes[idx + 2]  # +2 because first two plots are used
        mask = subtypes == subtype
        n_samples = np.sum(mask)
        print(f"  {subtype}: n={n_samples} samples, events={np.sum(events[mask])}")
        
        if n_samples >= 5:  # Only plot if enough samples
            plot_km_risk_groups(ax, predictions[mask], events[mask], durations[mask], 
                               subtype, colors, f"{subtype}", model_name)
        else:
            # Show message if insufficient samples
            ax.text(0.5, 0.5, f'Insufficient samples\n{subtype}: n={n_samples}\n(minimum 5 required)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax.set_title(f"{subtype}\n{model_name}", fontsize=11, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Survival Probability', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Kaplan-Meier Curves: Low-Risk vs High-Risk Groups (2-Group Stratification)\n{model_name}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"KM_curves_2groups_{model_name}.png"
        plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        print(f"2-group KM curves saved to {output_path / filename}")
    
    plt.close()
    return fig


def plot_risk_group_km_curves_3groups(aggregated_preds, model_name, output_root, save=True):
    """
    Plot Kaplan-Meier curves for low-risk, middle-risk, and high-risk groups (3 groups)
    Figure includes:
    1. Overall population
    2. Combined plot with all top 4 subtypes
    3-6. Individual plots for each top 4 subtype
    """
    
    predictions = aggregated_preds[model_name]['predictions']
    events = aggregated_preds[model_name]['events']
    durations = aggregated_preds[model_name]['durations']
    subtypes = aggregated_preds[model_name]['subtypes']
    
    # Get top 4 subtypes (excluding Unknown)
    subtype_counts = pd.Series(subtypes).value_counts()
    top4_subtypes = [s for s in subtype_counts.head(4).index if s != 'Unknown']
    
    print(f"\n{model_name} - Top 4 subtypes for 3-group KM curves: {top4_subtypes}")
    print(f"Subtype frequencies: {subtype_counts[top4_subtypes].to_dict()}")
    
    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Colors for risk groups (green, orange, red)
    colors = ['#2E8B57', '#FF8C00', '#DC143C']
    
    # Colors for different subtypes (for the combined plot)
    subtype_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # Red, Blue, Green, Purple
    
    # Plot 1: Overall
    ax = axes[0]
    plot_km_risk_groups_3groups(ax, predictions, events, durations, None, colors,
                                "Overall Population", model_name)
    
    # Plot 2: Combined plot with all top 4 subtypes (showing 3-group stratification for each subtype)
    ax = axes[1]
    plot_combined_subtypes_km(ax, predictions, events, durations, subtypes, 
                                      top4_subtypes, subtype_colors, model_name)
    
    # Plots 3-6: Individual plots for each top 4 subtype
    for idx, subtype in enumerate(top4_subtypes):
        ax = axes[idx + 2]  # +2 because first two plots are used
        mask = subtypes == subtype
        n_samples = np.sum(mask)
        
        if n_samples >= 10:  # Need more samples for 3-group stratification
            plot_km_risk_groups_3groups(ax, predictions[mask], events[mask], durations[mask],
                                        subtype, colors, f"{subtype}", model_name)
        else:
            # Show message if insufficient samples
            ax.text(0.5, 0.5, f'Insufficient samples\n{subtype}: n={n_samples}\n(minimum 10 required for 3 groups)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax.set_title(f"{subtype}\n{model_name}", fontsize=11, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Survival Probability', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Kaplan-Meier Curves: Low-Risk vs Middle-Risk vs High-Risk Groups (3-Group Stratification)\n{model_name}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"KM_curves_3groups_{model_name}.png"
        plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        print(f"3-group KM curves saved to {output_path / filename}")
    
    plt.close()
    return fig


def plot_combined_subtypes_km(ax, predictions, events, durations, subtypes, 
                              top4_subtypes, subtype_colors, model_name):
    """
    Helper function to plot combined KM curves for top 4 subtypes
    Shows overall survival curve for each subtype (without risk group stratification)
    Performs log-rank tests between specific subtype pairs
    """
    
    # Define abbreviation mapping
    subtype_abbrev = {
        "Wound Healing": "C1",
        "IFN-gamma Dominant": "C2", 
        "Inflammatory": "C3",
        "Lymphocyte Depleted": "C4"
    }
    
    # Store KM fit objects for log-rank tests
    kmf_dict = {}
    subtype_data = {}
    
    for idx, subtype in enumerate(top4_subtypes):
        mask = subtypes == subtype
        events_sub = events[mask]
        durations_sub = durations[mask]
        n_samples = np.sum(mask)
        
        if n_samples >= 5:
            # Get abbreviated name or use original if not in mapping
            abbrev_name = subtype_abbrev.get(subtype, subtype)
            display_name = f'{abbrev_name} ({subtype})' if subtype in subtype_abbrev else subtype
            
            # Fit KM curve for this subtype
            kmf = KaplanMeierFitter()
            kmf.fit(durations_sub, events_sub, 
                   label=f'{display_name} (n={n_samples})')
            kmf.plot_survival_function(ax=ax, color=subtype_colors[idx], 
                                      linewidth=2, linestyle='-')
            
            # Store for log-rank tests
            kmf_dict[subtype] = kmf
            subtype_data[subtype] = {
                'durations': durations_sub,
                'events': events_sub,
                'n_samples': n_samples,
                'abbrev': abbrev_name
            }
    
    # Perform log-rank tests for specific comparisons
    # C3 vs C2, C2 vs C1, C1 vs C4
    comparisons = [
        ("Inflammatory", "IFN-gamma Dominant"),  # C3 vs C2
        ("IFN-gamma Dominant", "Wound Healing"),  # C2 vs C1
        ("Wound Healing", "Lymphocyte Depleted"),  # C1 vs C4
        ("IFN-gamma Dominant", "Lymphocyte Depleted")  # C2 vs C4
    ]
    
    p_values = []
    valid_comparisons = []
    
    for subtype1, subtype2 in comparisons:
        if subtype1 in subtype_data and subtype2 in subtype_data:
            data1 = subtype_data[subtype1]
            data2 = subtype_data[subtype2]
            
            # Perform log-rank test
            results = logrank_test(data1['durations'], data2['durations'],
                                  data1['events'], data2['events'])
            p_value = results.p_value
            
            # Get abbreviated names
            abbrev1 = subtype_abbrev.get(subtype1, subtype1)
            abbrev2 = subtype_abbrev.get(subtype2, subtype2)
            
            p_values.append(p_value)
            valid_comparisons.append(f'{abbrev1} vs {abbrev2}')
    
    # Format p-value text
    if valid_comparisons:
        p_text_lines = []
        for comp, p in zip(valid_comparisons, p_values):
            if p < 0.0001:
                p_text_lines.append(f'{comp}: p < 0.0001')
            else:
                p_text_lines.append(f'{comp}: p = {p:.4f}')
        
        p_text = '\n'.join(p_text_lines)
        
        # Add p-values to plot
        ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='bottom')
    
    ax.set_title(f'Overall Survival by Immune Subtype\n{model_name}', 
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    # Add summary statistics for all subtypes combined
    n_total = len(events)
    n_events = np.sum(events)
    n_censored = n_total - n_events
    ax.text(0.05, 0.2, f'Total: n={int(n_total)}\nEvents: {int(n_events)}\nCensored: {int(n_censored)}', 
           transform=ax.transAxes, fontsize=8, verticalalignment='bottom', 
           horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


def plot_combined_risk_km_curves(aggregated_preds,
                                  output_root, save=True):
    """
    Create combined figure with 2-group, 3-group, and 4-group KM curves for both models
    for overall population only (since subtypes are shown in individual plots)
    """
    
    # Create a large figure with 2 rows (CoxPH and RSF) and 3 columns (2-group, 3-group, 4-group)
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    
    models = ['CoxPH', 'RSF']
    
    for row_idx, model_name in enumerate(models):
        predictions = aggregated_preds[model_name]['predictions']
        events = aggregated_preds[model_name]['events']
        durations = aggregated_preds[model_name]['durations']
        
        # 2-group plot (left column)
        ax = axes[row_idx, 0]
        plot_km_risk_groups_simple(ax, predictions, events, durations, 
                                   f"{model_name} - Low vs High Risk")
        ax.set_title(f"{model_name} - Low vs High Risk", fontsize=12, fontweight='bold')
        
        # 3-group plot (middle column)
        ax = axes[row_idx, 1]
        plot_km_risk_groups_3groups_simple(ax, predictions, events, durations,
                                           f"{model_name} - Low vs Middle vs High Risk")
        ax.set_title(f"{model_name} - Low vs Middle vs High Risk", fontsize=12, fontweight='bold')
        
        # 4-group plot (right column)
        ax = axes[row_idx, 2]
        plot_km_risk_groups_4groups_simple(ax, predictions, events, durations,
                                           f"{model_name} - Quartile Risk Groups")
        ax.set_title(f"{model_name} - Quartile Risk Groups", fontsize=12, fontweight='bold')
    
    plt.suptitle('Kaplan-Meier Curves: Risk Group Comparisons\nOverall Population', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = "KM_curves_combined_overall.png"
        plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        print(f"Combined KM curves saved to {output_path / filename}")
    
    plt.close()
    return fig

def plot_km_risk_groups(ax, predictions, events, durations, group_name, colors, title, model_name):
    """Helper function to plot 2-group KM curves"""
    
    # Split by median
    median_risk = np.median(predictions)
    low_risk_mask = predictions <= median_risk
    high_risk_mask = predictions > median_risk
    
    # Check if we have both groups
    if np.sum(low_risk_mask) == 0 or np.sum(high_risk_mask) == 0:
        ax.text(0.5, 0.5, f'Insufficient risk group separation\nLow risk: {np.sum(low_risk_mask)}\nHigh risk: {np.sum(high_risk_mask)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax.set_title(f'{title}\n{model_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Survival Probability', fontsize=10)
        return
    
    # Fit KM curves
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    
    kmf_low.fit(durations[low_risk_mask], events[low_risk_mask], 
                label=f'Low Risk (n={np.sum(low_risk_mask)})')
    kmf_high.fit(durations[high_risk_mask], events[high_risk_mask], 
                 label=f'High Risk (n={np.sum(high_risk_mask)})')
    
    # Plot
    kmf_low.plot_survival_function(ax=ax, color=colors[0], linewidth=2)
    kmf_high.plot_survival_function(ax=ax, color=colors[1], linewidth=2)
    
    # Perform log-rank test
    results = logrank_test(durations[low_risk_mask], durations[high_risk_mask],
                          events[low_risk_mask], events[high_risk_mask])
    p_value = results.p_value
    p_text = f'Log-rank p = {p_value:.4f}' if p_value >= 0.0001 else 'Log-rank p < 0.0001'
    
    # Add p-value to plot
    ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add title and labels
    if group_name:
        ax.set_title(f'{title}\n{model_name}', fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'{title}\n{model_name}', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Add summary statistics
    n_total = len(events)
    n_events = np.sum(events)
    n_censored = n_total - n_events
    ax.text(0.05, 0.2, f'Total: n={int(n_total)}\nEvents: {int(n_events)}\nCensored: {int(n_censored)}', 
           transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def plot_km_risk_groups_3groups(ax, predictions, events, durations, group_name, colors, title, model_name):
    """Helper function to plot 3-group KM curves"""
    
    # Split by tertiles
    tertiles = np.percentile(predictions, [33.33, 66.67])
    low_risk_mask = predictions <= tertiles[0]
    mid_risk_mask = (predictions > tertiles[0]) & (predictions <= tertiles[1])
    high_risk_mask = predictions > tertiles[1]
    
    # Check if we have all three groups
    if np.sum(low_risk_mask) == 0 or np.sum(mid_risk_mask) == 0 or np.sum(high_risk_mask) == 0:
        ax.text(0.5, 0.5, f'Insufficient risk group separation\nLow: {np.sum(low_risk_mask)}\nMid: {np.sum(mid_risk_mask)}\nHigh: {np.sum(high_risk_mask)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax.set_title(f'{title}\n{model_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Survival Probability', fontsize=10)
        return
    
    # Fit KM curves
    kmf_low = KaplanMeierFitter()
    kmf_mid = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    
    kmf_low.fit(durations[low_risk_mask], events[low_risk_mask], 
                label=f'Low Risk (n={np.sum(low_risk_mask)})')
    kmf_mid.fit(durations[mid_risk_mask], events[mid_risk_mask], 
                label=f'Middle Risk (n={np.sum(mid_risk_mask)})')
    kmf_high.fit(durations[high_risk_mask], events[high_risk_mask], 
                 label=f'High Risk (n={np.sum(high_risk_mask)})')
    
    # Plot
    kmf_low.plot_survival_function(ax=ax, color=colors[0], linewidth=2)
    kmf_mid.plot_survival_function(ax=ax, color=colors[1], linewidth=2)
    kmf_high.plot_survival_function(ax=ax, color=colors[2], linewidth=2)
    
    # Perform pairwise log-rank tests
    p_values = []
    
    res_lm = logrank_test(durations[low_risk_mask], durations[mid_risk_mask],
                         events[low_risk_mask], events[mid_risk_mask])
    p_lm = res_lm.p_value
    p_values.append(('Low vs Mid', p_lm))
    
    res_mh = logrank_test(durations[mid_risk_mask], durations[high_risk_mask],
                         events[mid_risk_mask], events[high_risk_mask])
    p_mh = res_mh.p_value
    p_values.append(('Mid vs High', p_mh))
    
    # Format p-values
    p_text = '\n'.join([f'{comp}: {p:.4f}' if p >= 0.0001 else f'{comp}: p < 0.0001' 
                       for comp, p in p_values])
    
    # Add p-values to plot
    ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=8,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add title and labels
    if group_name:
        ax.set_title(f'{title}\n{model_name}', fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'{title}\n{model_name}', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Add summary statistics
    n_total = len(events)
    n_events = np.sum(events)
    n_censored = n_total - n_events
    ax.text(0.05, 0.2, f'Total: n={int(n_total)}\nEvents: {int(n_events)}\nCensored: {int(n_censored)}', 
           transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def plot_km_risk_groups_simple(ax, predictions, events, durations, title):
    """Simplified 2-group KM plot for combined figure (overall only)"""
    
    # Split by median
    median_risk = np.median(predictions)
    low_risk_mask = predictions <= median_risk
    high_risk_mask = predictions > median_risk
    
    # Fit KM curves
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    
    kmf_low.fit(durations[low_risk_mask], events[low_risk_mask], 
                label=f'Low Risk (n={np.sum(low_risk_mask)})')
    kmf_high.fit(durations[high_risk_mask], events[high_risk_mask], 
                 label=f'High Risk (n={np.sum(high_risk_mask)})')
    
    # Plot
    kmf_low.plot_survival_function(ax=ax, color='#2E86AB', linewidth=2)
    kmf_high.plot_survival_function(ax=ax, color='#A23B72', linewidth=2)
    
    # Log-rank test
    results = logrank_test(durations[low_risk_mask], durations[high_risk_mask],
                          events[low_risk_mask], events[high_risk_mask])
    p_value = results.p_value
    p_text = f'p = {p_value:.4f}' if p_value >= 0.0001 else 'p < 0.0001'
    ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

def plot_km_risk_groups_3groups_simple(ax, predictions, events, durations, title):
    """Simplified 3-group KM plot for combined figure (overall only)"""
    
    # Split by tertiles
    tertiles = np.percentile(predictions, [33.33, 66.67])
    low_risk_mask = predictions <= tertiles[0]
    mid_risk_mask = (predictions > tertiles[0]) & (predictions <= tertiles[1])
    high_risk_mask = predictions > tertiles[1]
    
    # Fit KM curves
    kmf_low = KaplanMeierFitter()
    kmf_mid = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    
    kmf_low.fit(durations[low_risk_mask], events[low_risk_mask], 
                label=f'Low Risk (n={np.sum(low_risk_mask)})')
    kmf_mid.fit(durations[mid_risk_mask], events[mid_risk_mask], 
                label=f'Middle Risk (n={np.sum(mid_risk_mask)})')
    kmf_high.fit(durations[high_risk_mask], events[high_risk_mask], 
                 label=f'High Risk (n={np.sum(high_risk_mask)})')
    
    # Plot
    kmf_low.plot_survival_function(ax=ax, color='#2E8B57', linewidth=2)
    kmf_mid.plot_survival_function(ax=ax, color='#FF8C00', linewidth=2)
    kmf_high.plot_survival_function(ax=ax, color='#DC143C', linewidth=2)
    
    # Pairwise log-rank tests
    p_values = []
    
    res_lm = logrank_test(durations[low_risk_mask], durations[mid_risk_mask],
                         events[low_risk_mask], events[mid_risk_mask])
    p_values.append(('Low vs Mid', res_lm.p_value))
    
    res_lh = logrank_test(durations[low_risk_mask], durations[high_risk_mask],
                         events[low_risk_mask], events[high_risk_mask])
    p_values.append(('Low vs High', res_lh.p_value))
    
    res_mh = logrank_test(durations[mid_risk_mask], durations[high_risk_mask],
                         events[mid_risk_mask], events[high_risk_mask])
    p_values.append(('Mid vs High', res_mh.p_value))
    
    # Format p-values
    p_text = '\n'.join([f'{comp}: {p:.4f}' if p >= 0.0001 else f'{comp}: p < 0.0001' 
                       for comp, p in p_values])
    ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

def plot_km_risk_groups_4groups_simple(ax, predictions, events, durations, title):
    """Simplified 4-group KM plot for combined figure (overall only)"""
    
    # Split by quartiles (25%, 50%, 75%)
    quartiles = np.percentile(predictions, [25, 50, 75])
    q1_mask = predictions <= quartiles[0]  # Lowest risk (0-25%)
    q2_mask = (predictions > quartiles[0]) & (predictions <= quartiles[1])  # Low-mid risk (25-50%)
    q3_mask = (predictions > quartiles[1]) & (predictions <= quartiles[2])  # Mid-high risk (50-75%)
    q4_mask = predictions > quartiles[2]  # Highest risk (75-100%)
    
    # Fit KM curves
    kmf_q1 = KaplanMeierFitter()
    kmf_q2 = KaplanMeierFitter()
    kmf_q3 = KaplanMeierFitter()
    kmf_q4 = KaplanMeierFitter()
    
    kmf_q1.fit(durations[q1_mask], events[q1_mask], 
               label=f'Q1 (Lowest Risk) (n={np.sum(q1_mask)})')
    kmf_q2.fit(durations[q2_mask], events[q2_mask], 
               label=f'Q2 (Low-Mid Risk) (n={np.sum(q2_mask)})')
    kmf_q3.fit(durations[q3_mask], events[q3_mask], 
               label=f'Q3 (Mid-High Risk) (n={np.sum(q3_mask)})')
    kmf_q4.fit(durations[q4_mask], events[q4_mask], 
               label=f'Q4 (Highest Risk) (n={np.sum(q4_mask)})')
    
    # Plot with color gradient from green (lowest risk) to red (highest risk)
    # Colors: Q1=dark green, Q2=light green, Q3=orange, Q4=dark red
    kmf_q1.plot_survival_function(ax=ax, color='#006400', linewidth=2)  # Dark green
    kmf_q2.plot_survival_function(ax=ax, color='#8FBC8F', linewidth=2)  # Dark sea green
    kmf_q3.plot_survival_function(ax=ax, color='#FF8C00', linewidth=2)  # Dark orange
    kmf_q4.plot_survival_function(ax=ax, color='#8B0000', linewidth=2)  # Dark red
    
    # Pairwise log-rank tests (comparing adjacent groups and extreme groups)
    p_values = []
    
    # Adjacent comparisons
    res_q1_q2 = logrank_test(durations[q1_mask], durations[q2_mask],
                            events[q1_mask], events[q2_mask])
    p_values.append(('Q1 vs Q2', res_q1_q2.p_value))
    
    res_q2_q3 = logrank_test(durations[q2_mask], durations[q3_mask],
                            events[q2_mask], events[q3_mask])
    p_values.append(('Q2 vs Q3', res_q2_q3.p_value))
    
    res_q3_q4 = logrank_test(durations[q3_mask], durations[q4_mask],
                            events[q3_mask], events[q4_mask])
    p_values.append(('Q3 vs Q4', res_q3_q4.p_value))
    
    # Extreme groups comparison (Q2 vs Q4)
    res_q2_q4 = logrank_test(durations[q2_mask], durations[q4_mask],
                            events[q2_mask], events[q4_mask])
    p_values.append(('Q2 vs Q4', res_q2_q4.p_value))
    
    # Format p-values
    p_text = '\n'.join([f'{comp}: {p:.4f}' if p >= 0.0001 else f'{comp}: p < 0.0001' 
                       for comp, p in p_values])
    
    # Position text box in top-left corner to avoid overlap with curves
    ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

def main():
    parser = argparse.ArgumentParser(description='Compare two models with cross-validated CoxPH and RSF using KFold')
    parser.add_argument('--model1', type=str, default="Radiomics: BiomedParse (SPARRA)",
                       help='First model key')
    parser.add_argument('--model2', type=str, default="Pathomics: CONCH (ABMIL)",
                       help='Second model key')
    parser.add_argument('--omics', type=str, default=None,
                       help='Filter by omics type')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--no-cv-summary', action='store_true',
                       help='Skip CV summary plot')
    parser.add_argument('--no-km-curves', action='store_true',
                       help='Skip KM curves plots')
    
    args = parser.parse_args()
    
    # Set paths
    task = "TCGA_survival_PFI"
    root_parent = f"/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/outcomes_slice+tumor/{task}"
    immune_csv = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/TCGA_Pan-Cancer_outcomes/phenotypes/immune_subtype/immune_subtype.csv"
    output_root = f"/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/plots/Survival_ImmuneSubtype/{task}"
    
    # Load immune data
    print("Loading immune subtype data...")
    immune_df = pd.read_csv(immune_csv)
    immune_df["ID3"] = immune_df["SampleID"].apply(
        lambda x: "-".join(str(x).split("-")[:3])
    )
    immune_df["Subtype"] = immune_df["Subtype_Immune_Model_Based"].str.replace(
        r"\s*\(.*\)", "", regex=True
    )
    
    # Load model data
    print("Loading model data...")
    target_model_names = [args.model1, args.model2]
    # target_model_names += [
    #     'Radiomics: pyradiomics (MEAN)', 
    #     'Radiomics: FMCIB (MEAN)',
    #     'Radiomics: BiomedParse (MEAN)',
    #     'Radiomics: BiomedParse (ABMIL)',
    #     'Radiomics: BiomedParse (SPARRA)',
    #     'Radiomics: LVMMed (MEAN)', 
    #     'Radiomics: LVMMed (ABMIL)', 
    #     'Radiomics: LVMMed (SPARRA)', 
    # ]
    # target_model_names += [
    #     'Pathomics: UNI (MEAN)', 
    #     'Pathomics: UNI (ABMIL)',
    #     'Pathomics: UNI (SPARRA)',
    #     'Pathomics: CHIEF (MEAN)',
    #     'Pathomics: CHIEF (ABMIL)',
    #     'Pathomics: CHIEF (SPARRA)',
    #     'Pathomics: CONCH (MEAN)',
    #     # 'Pathomics: CONCH (ABMIL)',
    #     'Pathomics: CONCH (SPARRA)',
    # ]
    model_data = load_model_data(root_parent, target_model_names, target_omics=args.omics)
    print(f"Loaded {len(model_data)} models")
    
    # Prepare data
    print(f"Preparing data for {target_model_names}...")
    risk_dict, events, durations, subtypes, _, _ = prepare_data_for_models(
        model_data, immune_df, target_model_names
    )
    
    if risk_dict[args.model1] is None:
        return
    
    print(f"\nFound {len(risk_dict[args.model1])} common samples")
    print(f"Events: {np.sum(events)} ({np.mean(events)*100:.1f}%)")
    
    # Show subtype distribution
    subtype_counts = pd.Series(subtypes).value_counts()
    print(f"\nImmune subtype distribution:")
    for subtype, count in subtype_counts.items():
        if subtype != 'Unknown':
            print(f"  {subtype}: {count} samples ({count/len(subtypes)*100:.1f}%)")
    print(f"  Unknown: {subtype_counts.get('Unknown', 0)} samples")
    
    # Perform cross-validated survival analysis with KFold
    print(f"\nPerforming {args.n_folds}-fold cross-validation (KFold)...")
    cv_models = CrossValidatedSurvivalModels(
        risk_dict, events, durations, 
        n_folds=args.n_folds, random_state=42
    )
    cv_results = cv_models.fit_cross_validate()
    
    # Get aggregated out-of-fold predictions
    aggregated_preds = cv_models.get_aggregated_predictions(subtypes)

    risks1 = aggregated_preds['CoxPH']['risk_dict'][args.model1]
    risks2 = aggregated_preds['CoxPH']['risk_dict'][args.model2]
    raw_preds = {
        args.model1: {
            'predictions': risks1,
            'events': aggregated_preds['CoxPH']['events'],
            'durations': aggregated_preds['CoxPH']['durations'],
            'subtypes': aggregated_preds['CoxPH']['subtypes']
        },
        args.model2: {
            'predictions': risks2,
            'events': aggregated_preds['CoxPH']['events'],
            'durations': aggregated_preds['CoxPH']['durations'],
            'subtypes': aggregated_preds['CoxPH']['subtypes']
        }
    }
    
    # Get subtype details
    subtype_cindices_cv, subtype_details = cv_models.get_subtype_cindices_cv(subtypes)
    
    # Generate cross-validated decision boundary plots
    print("\nGenerating cross-validated survival decision boundary plots...")
    if set(target_model_names) == {args.model1, args.model2}:
        plot_survival_decision_boundary_cv(
            risks1, risks2, subtypes,
            cv_models, aggregated_preds, subtype_details,
            args.model1, args.model2,
            output_root
        )
    
    # Generate CV performance summary
    if not args.no_cv_summary:
        print("\nGenerating CV performance summary plot...")
        plot_cv_performance_summary(
            cv_models, args.model1, args.model2, output_root
        )
    
    # Generate KM curves for risk groups (only top 4 subtypes)
    if not args.no_km_curves:
        print("\n" + "="*60)
        print("Generating Kaplan-Meier Curves for Risk Groups (Top 4 Subtypes Only)")
        print("="*60)
        
        # Plot 2-group KM curves (Low vs High Risk) for both models
        print("\n1. Generating 2-group KM curves (Low vs High Risk)...")
        for model_name in [args.model1, args.model2]:
            print(f"   - {model_name}")
            plot_risk_group_km_curves(
                raw_preds,
                model_name, output_root, save=True
            )

        for model_name in ['CoxPH', 'RSF']:
            print(f"   - {model_name}")
            plot_risk_group_km_curves(
                aggregated_preds,
                model_name, output_root, save=True
            )
        
        # Plot 3-group KM curves (Low vs Middle vs High Risk) for both models
        print("\n2. Generating 3-group KM curves (Low vs Middle vs High Risk)...")
        for model_name in [args.model1, args.model2]:
            print(f"   - {model_name}")
            plot_risk_group_km_curves_3groups(
                raw_preds,
                model_name, output_root, save=True
            )

        for model_name in ['CoxPH', 'RSF']:
            print(f"   - {model_name}")
            plot_risk_group_km_curves_3groups(
                aggregated_preds,
                model_name, output_root, save=True
            )
        
        # Plot combined figure for overall population
        print("\n3. Generating combined KM curves figure (overall population)...")
        plot_combined_risk_km_curves(
            aggregated_preds,
            output_root, save=True
        )
        
        # Print summary of risk group sizes for top 4 subtypes
        print("\n" + "="*60)
        print("Risk Group Sizes Summary (Top 4 Subtypes):")
        print("="*60)
        
        top4_subtypes = [s for s in subtype_counts.head(4).index if s != 'Unknown']
        
        for model_name in ['CoxPH', 'RSF']:
            predictions = aggregated_preds[model_name]['predictions']
            print(f"\n{model_name}:")
            
            # Overall
            median_risk = np.median(predictions)
            tertiles = np.percentile(predictions, [33.33, 66.67])
            print(f"  Overall: 2-group (L/H): {np.sum(predictions <= median_risk)}/{np.sum(predictions > median_risk)}")
            print(f"          3-group (L/M/H): {np.sum(predictions <= tertiles[0])}/{np.sum((predictions > tertiles[0]) & (predictions <= tertiles[1]))}/{np.sum(predictions > tertiles[1])}")
            
            # Each top subtype
            for subtype in top4_subtypes:
                mask = subtypes == subtype
                pred_sub = predictions[mask]
                if len(pred_sub) >= 5:
                    median_sub = np.median(pred_sub)
                    tertiles_sub = np.percentile(pred_sub, [33.33, 66.67])
                    print(f"  {subtype:20s}: 2-group: {np.sum(pred_sub <= median_sub)}/{np.sum(pred_sub > median_sub)}")
                    print(f"  {'':20s}  3-group: {np.sum(pred_sub <= tertiles_sub[0])}/{np.sum((pred_sub > tertiles_sub[0]) & (pred_sub <= tertiles_sub[1]))}/{np.sum(pred_sub > tertiles_sub[1])}")
    
    print("\n✅ Done! All results are based on cross-validated predictions using KFold (no data leakage).")
    print(f"Results saved to: {output_root}")

if __name__ == "__main__":
    main()