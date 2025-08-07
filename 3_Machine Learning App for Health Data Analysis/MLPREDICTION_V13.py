import matplotlib
matplotlib.use('Agg')  # Streamlit Cloud용 필수 설정

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from io import BytesIO
import itertools
import base64

# PDF 관련 import를 조건부로 처리
try:
    from xhtml2pdf import pisa
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.title("📊 Health Data ML Prediction 📊")

# 세션 상태 초기화
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None
if 'data' not in st.session_state:
    st.session_state.data = None

st.sidebar.header("Upload Your CSV Data")
st.sidebar.markdown(f"**Progress:** Step {st.session_state.step} / 3")

# 앱 리셋 함수
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 1
    st.session_state.features = []
    st.session_state.target = None
    st.session_state.data = None

st.sidebar.button("Reset App", on_click=reset_app)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # 데이터 로드 및 세션에 저장
    if st.session_state.data is None:
        st.session_state.data = pd.read_csv(uploaded_file)
    
    data = st.session_state.data
    
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    
    # 데이터 기본 정보 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())

    # Step 1: EDA
    if st.session_state.step == 1:
        st.subheader("Step 1: Exploratory Data Analysis")
        
        # 데이터 타입 정보
        st.write("**Data Types:**")
        st.dataframe(pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Unique Values': [data[col].nunique() for col in data.columns]
        }))
        
        if st.checkbox("Show Feature Distribution Plot"):
            selected_feature = st.selectbox("Select Feature to Visualize", data.columns)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if data[selected_feature].dtype in ['object', 'category'] or data[selected_feature].nunique() < 10:
                sns.countplot(x=selected_feature, data=data, ax=ax)
                ax.set_title(f'Distribution of {selected_feature}')
                plt.xticks(rotation=45)
            else:
                data[selected_feature].hist(bins=30, ax=ax)
                ax.set_title(f'Distribution of {selected_feature}')
                ax.set_xlabel(selected_feature)
                ax.set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        if st.checkbox("Show Correlation Matrix"):
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                correlation_matrix = data[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Matrix')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for correlation matrix.")

        if st.button("Next: Go to Feature Selection"):
            st.session_state.step = 2
            st.rerun()

    # Step 2: Feature Selection
    elif st.session_state.step == 2:
        st.subheader("Step 2: Feature Selection")
        
        # 사이드바에서 features와 target 선택
        st.sidebar.header("Select Features & Target")
        
        # 이전 선택값을 기본값으로 사용
        selected_features = st.sidebar.multiselect(
            "Select Features", 
            options=data.columns.tolist(),
            default=st.session_state.features if st.session_state.features else []
        )
        
        # target 선택 (이전 선택값을 기본값으로 사용)
        target_options = data.columns.tolist()
        default_target_index = 0
        if st.session_state.target and st.session_state.target in target_options:
            default_target_index = target_options.index(st.session_state.target)
        
        selected_target = st.sidebar.selectbox(
            "Select Target Variable", 
            options=target_options,
            index=default_target_index
        )
        
        # 세션 상태 업데이트
        st.session_state.features = selected_features
        st.session_state.target = selected_target
        
        # 선택된 변수들 표시
        if selected_features:
            st.write("**Selected Features:**")
            feature_info = []
            for feature in selected_features:
                feature_info.append({
                    'Feature': feature,
                    'Type': str(data[feature].dtype),
                    'Unique Values': data[feature].nunique(),
                    'Missing Values': data[feature].isnull().sum()
                })
            st.dataframe(pd.DataFrame(feature_info))
        
        if selected_target:
            st.write(f"**Selected Target:** {selected_target}")
            st.write("**Target Distribution:**")
            target_dist = data[selected_target].value_counts()
            st.dataframe(target_dist.to_frame('Count'))
        
        # 유효성 검사
        validation_passed = True
        if not selected_features:
            st.error("❌ Please select at least one feature.")
            validation_passed = False
        
        if not selected_target:
            st.error("❌ Please select a target variable.")
            validation_passed = False
        
        if selected_target in selected_features:
            st.error("❌ Target variable cannot be included in features.")
            validation_passed = False
        
        # 네비게이션 버튼들
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to EDA"):
                st.session_state.step = 1
                st.rerun()
        
        with col2:
            if validation_passed:
                if st.button("Next: Model Evaluation →"):
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.button("Next: Model Evaluation →", disabled=True)

    # Step 3: Model Evaluation
    elif st.session_state.step == 3:
        st.subheader("Step 3: Model Evaluation")
        
        # 저장된 features와 target 사용
        features = st.session_state.features
        target = st.session_state.target
        
        # 선택된 변수들 확인
        st.info(f"**Features:** {', '.join(features)}")
        st.info(f"**Target:** {target}")
        
        try:
            # 전처리 옵션 설정
            st.sidebar.header("Preprocessing Options")
            
            # 데이터 크기 체크
            data_size = data.shape[0] * data.shape[1]
            if data_size > 100000:  # 10만 셀 이상
                st.sidebar.warning("⚠️ Large dataset detected")
                use_sampling = st.sidebar.checkbox("Use data sampling (faster)", value=True)
                sample_size = st.sidebar.slider("Sample size", 1000, min(50000, data.shape[0]), 10000)
            else:
                use_sampling = False
                sample_size = None
            
            # 전처리 방식 선택
            preprocessing_mode = st.sidebar.selectbox(
                "Preprocessing Mode",
                ["Fast (Numeric only)", "Standard (with encoding)", "Skip preprocessing"],
                index=1
            )
            
            # 데이터 샘플링
            if use_sampling:
                data_sample = data.sample(n=sample_size, random_state=42)
                st.info(f"🔄 Using {sample_size:,} samples out of {data.shape[0]:,} for faster processing")
            else:
                data_sample = data
            
            # 전처리 실행
            if preprocessing_mode == "Skip preprocessing":
                # 전처리 없이 바로 사용 (수치형만)
                numeric_features = [f for f in features if data_sample[f].dtype in ['int64', 'float64']]
                if not numeric_features:
                    st.error("❌ No numeric features found. Please select 'Standard' preprocessing or upload numeric data.")
                    st.stop()
                
                X = data_sample[numeric_features].fillna(data_sample[numeric_features].mean())
                y = data_sample[target]
                st.info(f"ℹ️ Using {len(numeric_features)} numeric features only")
                
            elif preprocessing_mode == "Fast (Numeric only)":
                # 수치형 변수만 사용, 간단한 결측값 처리
                numeric_features = [f for f in features if data_sample[f].dtype in ['int64', 'float64']]
                categorical_features = [f for f in features if f not in numeric_features]
                
                if categorical_features:
                    st.warning(f"⚠️ Excluding {len(categorical_features)} categorical features: {categorical_features}")
                
                if not numeric_features:
                    st.error("❌ No numeric features found for fast mode.")
                    st.stop()
                
                X = data_sample[numeric_features].fillna(data_sample[numeric_features].mean())
                y = data_sample[target]
                
            else:  # Standard preprocessing
                # 범주형 변수 인코딩 (제한적으로)
                categorical_features = [f for f in features if data_sample[f].dtype in ['object', 'category']]
                numeric_features = [f for f in features if f not in categorical_features]
                
                # 고카디널리티 범주형 변수 필터링
                high_cardinality = []
                for col in categorical_features:
                    if data_sample[col].nunique() > 20:
                        high_cardinality.append(col)
                
                if high_cardinality:
                    st.warning(f"⚠️ Excluding high cardinality features (>20 categories): {high_cardinality}")
                    categorical_features = [f for f in categorical_features if f not in high_cardinality]
                
                # 데이터 전처리
                X_parts = []
                
                if numeric_features:
                    X_numeric = data_sample[numeric_features].fillna(data_sample[numeric_features].mean())
                    X_parts.append(X_numeric)
                
                if categorical_features:
                    with st.spinner("Encoding categorical variables..."):
                        X_categorical = pd.get_dummies(data_sample[categorical_features], dummy_na=True)
                        X_parts.append(X_categorical)
                
                X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame()
                y = data_sample[target]
            
            # 결측값 최종 처리
            if X.isnull().any().any():
                missing_count = X.isnull().sum().sum()
                st.warning(f"⚠️ Filling {missing_count} remaining missing values")
                X = X.fillna(0)  # 빠른 처리를 위해 0으로 채움
            
            # 타겟 변수 처리
            if y.isnull().any():
                missing_target = y.isnull().sum()
                st.warning(f"Filled {missing_target} missing values in target with mode.")
                y = y.fillna(y.mode()[0])

            # 타겟 변수 인코딩
            from sklearn.preprocessing import LabelEncoder
            le = None
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                st.info(f"Target classes: {list(le.classes_)}")
            else:
                y_encoded = y
                unique_values = sorted(y.unique())
                st.info(f"Target values: {unique_values}")
            
            # 최종 데이터 정보
            st.success(f"✅ Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features")

            # 모델 설정
            st.sidebar.header("Model Configuration")
            test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            st.write(f"**Training set:** {X_train.shape[0]} samples")
            st.write(f"**Test set:** {X_test.shape[0]} samples")

            # 모델 선택
            st.sidebar.header("Select Models to Compare")
            model_choices = st.sidebar.multiselect(
                "Models", 
                ["Random Forest", "Logistic Regression", "SVM", "LightGBM", "XGBoost"], 
                default=["Random Forest", "Logistic Regression"]
            )
            
            # 하이퍼파라미터 튜닝 옵션
            st.sidebar.header("Hyperparameter Tuning")
            enable_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False)
            
            if enable_tuning:
                tuning_strategy = st.sidebar.selectbox(
                    "Tuning Strategy", 
                    ["Auto Best (Recommended)", "Manual Selection", "Quick Test"]
                )
                
                if tuning_strategy == "Auto Best (Recommended)":
                    st.sidebar.info("🚀 Using optimized parameter ranges for best performance")
                    tuning_method = "RandomSearch"  # 더 효율적
                    cv_folds = 5
                    
                    # 자동으로 최적화된 파라미터 그리드 설정
                    param_grids = {}
                    
                    if "Random Forest" in model_choices:
                        param_grids["Random Forest"] = {
                            'n_estimators': [100, 200, 300, 500],
                            'max_depth': [None, 10, 20, 30, 50],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                            'max_features': ['sqrt', 'log2', None],
                            'random_state': [42]
                        }
                    
                    if "Logistic Regression" in model_choices:
                        param_grids["Logistic Regression"] = {
                            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                            'penalty': ['l1', 'l2', 'elasticnet', None],
                            'max_iter': [1000, 2000],
                            'random_state': [42]
                        }
                    
                    if "SVM" in model_choices:
                        param_grids["SVM"] = {
                            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                            'degree': [2, 3, 4, 5],  # poly kernel용
                            'probability': [True],
                            'random_state': [42]
                        }
                    
                    if "LightGBM" in model_choices:
                        param_grids["LightGBM"] = {
                            'n_estimators': [100, 200, 300, 500],
                            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                            'max_depth': [-1, 5, 10, 15, 20],
                            'num_leaves': [31, 50, 100, 200],
                            'subsample': [0.8, 0.9, 1.0],
                            'colsample_bytree': [0.8, 0.9, 1.0],
                            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
                            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
                            'random_state': [42],
                            'verbose': [-1]
                        }
                    
                    if "XGBoost" in model_choices:
                        param_grids["XGBoost"] = {
                            'n_estimators': [100, 200, 300, 500],
                            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                            'max_depth': [3, 4, 5, 6, 8, 10],
                            'subsample': [0.8, 0.9, 1.0],
                            'colsample_bytree': [0.8, 0.9, 1.0],
                            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
                            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
                            'gamma': [0.0, 0.1, 0.2, 0.5],
                            'random_state': [42],
                            'use_label_encoder': [False],
                            'eval_metric': ['logloss']
                        }
                
                elif tuning_strategy == "Quick Test":
                    st.sidebar.info("⚡ Using minimal parameters for quick testing")
                    tuning_method = "GridSearch"
                    cv_folds = 3
                    
                    # 빠른 테스트용 작은 그리드
                    param_grids = {}
                    
                    if "Random Forest" in model_choices:
                        param_grids["Random Forest"] = {
                            'n_estimators': [100, 200],
                            'max_depth': [None, 10],
                            'random_state': [42]
                        }
                    
                    if "Logistic Regression" in model_choices:
                        param_grids["Logistic Regression"] = {
                            'C': [0.1, 1.0, 10.0],
                            'solver': ['liblinear', 'lbfgs'],
                            'random_state': [42]
                        }
                    
                    if "SVM" in model_choices:
                        param_grids["SVM"] = {
                            'C': [1.0, 10.0],
                            'kernel': ['rbf', 'linear'],
                            'probability': [True],
                            'random_state': [42]
                        }
                    
                    if "LightGBM" in model_choices:
                        param_grids["LightGBM"] = {
                            'n_estimators': [100, 200],
                            'learning_rate': [0.1, 0.2],
                            'random_state': [42],
                            'verbose': [-1]
                        }
                    
                    if "XGBoost" in model_choices:
                        param_grids["XGBoost"] = {
                            'n_estimators': [100, 200],
                            'learning_rate': [0.1, 0.2],
                            'random_state': [42],
                            'use_label_encoder': [False],
                            'eval_metric': ['logloss']
                        }
                
                else:  # Manual Selection
                    tuning_method = st.sidebar.selectbox("Tuning Method", ["GridSearch", "RandomSearch"])
                    cv_folds = st.sidebar.slider("Cross Validation Folds", 3, 10, 5)
                    
                    # 기존 수동 선택 파라미터들...
                    param_grids = {}
                    
                    if "Random Forest" in model_choices:
                        with st.sidebar.expander("Random Forest Parameters"):
                            rf_n_estimators = st.multiselect("n_estimators", [50, 100, 200, 300], default=[100, 200])
                            rf_max_depth = st.multiselect("max_depth", [None, 10, 20, 30], default=[None, 20])
                            rf_min_samples_split = st.multiselect("min_samples_split", [2, 5, 10], default=[2, 5])
                            
                            param_grids["Random Forest"] = {
                                'n_estimators': rf_n_estimators,
                                'max_depth': rf_max_depth,
                                'min_samples_split': rf_min_samples_split,
                                'random_state': [42]
                            }
                    
                    if "Logistic Regression" in model_choices:
                        with st.sidebar.expander("Logistic Regression Parameters"):
                            lr_C = st.multiselect("C (Regularization)", [0.01, 0.1, 1.0, 10.0, 100.0], default=[0.1, 1.0, 10.0])
                            lr_solver = st.multiselect("solver", ['liblinear', 'lbfgs'], default=['liblinear'])
                            
                            param_grids["Logistic Regression"] = {
                                'C': lr_C,
                                'solver': lr_solver,
                                'max_iter': [1000],
                                'random_state': [42]
                            }
                    
                    if "SVM" in model_choices:
                        with st.sidebar.expander("SVM Parameters"):
                            svm_C = st.multiselect("C", [0.1, 1.0, 10.0, 100.0], default=[1.0, 10.0])
                            svm_kernel = st.multiselect("kernel", ['rbf', 'linear', 'poly'], default=['rbf', 'linear'])
                            svm_gamma = st.multiselect("gamma", ['scale', 'auto'], default=['scale'])
                            
                            param_grids["SVM"] = {
                                'C': svm_C,
                                'kernel': svm_kernel,
                                'gamma': svm_gamma,
                                'probability': [True],
                                'random_state': [42]
                            }
                    
                    if "LightGBM" in model_choices:
                        with st.sidebar.expander("LightGBM Parameters"):
                            lgb_n_estimators = st.multiselect("n_estimators", [100, 200, 300], default=[100, 200])
                            lgb_learning_rate = st.multiselect("learning_rate", [0.01, 0.1, 0.2], default=[0.1, 0.2])
                            lgb_max_depth = st.multiselect("max_depth", [-1, 10, 20], default=[-1, 10])
                            
                            param_grids["LightGBM"] = {
                                'n_estimators': lgb_n_estimators,
                                'learning_rate': lgb_learning_rate,
                                'max_depth': lgb_max_depth,
                                'random_state': [42],
                                'verbose': [-1]
                            }
                    
                    if "XGBoost" in model_choices:
                        with st.sidebar.expander("XGBoost Parameters"):
                            xgb_n_estimators = st.multiselect("n_estimators", [100, 200, 300], default=[100, 200])
                            xgb_learning_rate = st.multiselect("learning_rate", [0.01, 0.1, 0.2], default=[0.1, 0.2])
                            xgb_max_depth = st.multiselect("max_depth", [3, 6, 9], default=[3, 6])
                            
                            param_grids["XGBoost"] = {
                                'n_estimators': xgb_n_estimators,
                                'learning_rate': xgb_learning_rate,
                                'max_depth': xgb_max_depth,
                                'random_state': [42],
                                'use_label_encoder': [False],
                                'eval_metric': ['logloss']
                            }
                
                # 튜닝 예상 시간 표시
                if param_grids:
                    total_combinations = sum([len(list(itertools.product(*params.values()))) for params in param_grids.values()])
                    estimated_time = total_combinations * cv_folds * 0.1  # 대략적인 추정
                    
                    if tuning_method == "RandomSearch":
                        estimated_time = estimated_time * 0.3  # RandomSearch는 더 빠름
                        n_iter_per_model = min(50, total_combinations // len(param_grids)) if param_grids else 10
                        st.sidebar.info(f"⏱️ Estimated time: {estimated_time:.1f}s\n({n_iter_per_model} iterations per model)")
                    else:
                        st.sidebar.info(f"⏱️ Estimated time: {estimated_time:.1f}s\n({total_combinations} total combinations)")
                
                # Auto Best 모드에서 추가 설정
                if tuning_strategy == "Auto Best (Recommended)":
                    with st.sidebar.expander("⚙️ Advanced Auto Settings"):
                        auto_n_iter = st.slider("Max iterations per model", 20, 100, 50)
                        auto_scoring = st.selectbox("Optimization metric", 
                                                   ["accuracy", "f1", "precision", "recall", "roc_auc"], 
                                                   index=0)
                        early_stopping = st.checkbox("Early stopping (faster)", value=True)

            if model_choices:
                model_dict = {
                    "Random Forest": RandomForestClassifier(),
                    "Logistic Regression": LogisticRegression(),
                    "SVM": SVC(),
                    "LightGBM": LGBMClassifier(),
                    "XGBoost": XGBClassifier()
                }

                results = []
                best_params_results = []
                fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
                fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

                report_html = "<h2>Model Evaluation Report</h2>"

                # 모델 훈련 및 평가
                progress_bar = st.progress(0)
                total_models = len(model_choices)
                trained_models = {}  # 훈련된 모델들 저장
                
                for idx, model_name in enumerate(model_choices):
                    st.subheader(f"🔄 Training {model_name}...")
                    
                    base_model = model_dict[model_name]
                    
                    # 하이퍼파라미터 튜닝 수행
                    if enable_tuning and model_name in param_grids:
                        with st.spinner(f"🔍 Auto-tuning {model_name} for best performance..."):
                            if tuning_strategy == "Auto Best (Recommended)":
                                # RandomizedSearchCV 사용 (더 효율적)
                                from sklearn.model_selection import RandomizedSearchCV
                                
                                # 파라미터 그리드에서 호환되지 않는 조합 처리
                                filtered_params = param_grids[model_name].copy()
                                
                                # Logistic Regression penalty-solver 호환성 처리
                                if model_name == "Logistic Regression":
                                    # penalty와 solver 호환성 확인
                                    compatible_combinations = []
                                    for penalty in filtered_params.get('penalty', ['l2']):
                                        for solver in filtered_params.get('solver', ['lbfgs']):
                                            if (penalty == 'l1' and solver in ['liblinear', 'saga']) or \
                                               (penalty == 'l2' and solver in ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']) or \
                                               (penalty == 'elasticnet' and solver == 'saga') or \
                                               (penalty is None and solver in ['lbfgs', 'newton-cg', 'sag', 'saga']):
                                                compatible_combinations.append((penalty, solver))
                                    
                                    if compatible_combinations:
                                        penalties, solvers = zip(*compatible_combinations)
                                        filtered_params['penalty'] = list(set(penalties))
                                        filtered_params['solver'] = list(set(solvers))
                                
                                grid_search = RandomizedSearchCV(
                                    base_model,
                                    filtered_params,
                                    cv=cv_folds,
                                    scoring=auto_scoring if tuning_strategy == "Auto Best (Recommended)" else 'accuracy',
                                    n_iter=auto_n_iter if tuning_strategy == "Auto Best (Recommended)" else 30,
                                    n_jobs=-1,
                                    random_state=42
                                )
                            else:
                                # 기존 방식
                                if tuning_method == "GridSearch":
                                    from sklearn.model_selection import GridSearchCV
                                    grid_search = GridSearchCV(
                                        base_model, 
                                        param_grids[model_name], 
                                        cv=cv_folds, 
                                        scoring='accuracy',
                                        n_jobs=-1
                                    )
                                else:  # RandomSearch
                                    from sklearn.model_selection import RandomizedSearchCV
                                    grid_search = RandomizedSearchCV(
                                        base_model, 
                                        param_grids[model_name], 
                                        cv=cv_folds, 
                                        scoring='accuracy',
                                        n_iter=20,
                                        n_jobs=-1,
                                        random_state=42
                                    )
                            
                            # 시간 측정
                            import time
                            start_time = time.time()
                            
                            grid_search.fit(X_train, y_train)
                            
                            end_time = time.time()
                            tuning_time = end_time - start_time
                            
                            best_model = grid_search.best_estimator_
                            best_params = grid_search.best_params_
                            cv_score = grid_search.best_score_
                            
                            # 최적 파라미터 저장
                            best_params_results.append({
                                'Model': model_name,
                                'Best_Params': best_params,
                                'CV_Score': f"{cv_score:.4f}",
                                'Tuning_Time': f"{tuning_time:.1f}s"
                            })
                            
                            # 결과 표시
                            if tuning_strategy == "Auto Best (Recommended)":
                                st.success(f"🎯 Auto-tuned {model_name}: {cv_score:.4f} ({tuning_time:.1f}s)")
                            else:
                                st.success(f"✅ Best CV Score: {cv_score:.4f} ({tuning_time:.1f}s)")
                            
                            with st.expander(f"🏆 Optimal Parameters for {model_name}"):
                                for param, value in best_params.items():
                                    st.write(f"**{param}**: `{value}`")
                                
                                # 성능 향상 정보
                                if tuning_strategy == "Auto Best (Recommended)":
                                    st.info(f"🚀 Tested {auto_n_iter} parameter combinations")
                                    if hasattr(grid_search, 'n_splits_'):
                                        st.info(f"📊 {cv_folds}-fold cross-validation completed")
                    
                    else:
                        # 기본 파라미터로 훈련
                        if model_name == "Random Forest":
                            best_model = RandomForestClassifier(random_state=42)
                        elif model_name == "Logistic Regression":
                            best_model = LogisticRegression(max_iter=1000, random_state=42)
                        elif model_name == "SVM":
                            best_model = SVC(probability=True, random_state=42)
                        elif model_name == "LightGBM":
                            best_model = LGBMClassifier(random_state=42, verbose=-1)
                        elif model_name == "XGBoost":
                            best_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                        
                        best_model.fit(X_train, y_train)
                        best_params_results.append({
                            'Model': model_name,
                            'Best_Params': 'Default parameters used',
                            'CV_Score': 'N/A',
                            'Tuning_Time': 'N/A'
                        })
                    
                    # 훈련된 모델 저장
                    trained_models[model_name] = best_model
                    
                    # 예측 수행
                    y_pred = best_model.predict(X_test)
                    
                    # 이진 분류인지 다중 분류인지 확인
                    n_classes = len(set(y_encoded))
                    if n_classes == 2:
                        y_prob = best_model.predict_proba(X_test)[:, 1]
                    else:
                        # 다중 분류의 경우 ROC curve는 복잡하므로 일단 스킵
                        y_prob = None

                    acc = accuracy_score(y_test, y_pred)
                    results.append({"Model": model_name, "Accuracy": f"{acc:.4f}"})

                    report_html += f"<h3>{model_name} Classification Report</h3>"
                    if enable_tuning and model_name in param_grids:
                        report_html += f"<p><strong>Best Parameters:</strong> {best_params}</p>"
                        report_html += f"<p><strong>CV Score:</strong> {cv_score:.4f}</p>"
                    report_html += f"<pre>{classification_report(y_test, y_pred)}</pre>"

                    # ROC Curve (이진 분류만)
                    if y_prob is not None and n_classes == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

                        precision, recall, _ = precision_recall_curve(y_test, y_prob)
                        ax_pr.plot(recall, precision, label=f'{model_name}')
                    
                    # 진행률 업데이트
                    progress_bar.progress((idx + 1) / total_models)
                
                progress_bar.empty()

                # 최적 파라미터 결과 표시
                if enable_tuning:
                    st.subheader("🎯 Best Hyperparameters")
                    params_df = pd.DataFrame(best_params_results)
                    
                    for _, row in params_df.iterrows():
                        with st.expander(f"📊 {row['Model']} - CV Score: {row['CV_Score']}"):
                            if isinstance(row['Best_Params'], dict):
                                for param, value in row['Best_Params'].items():
                                    st.write(f"**{param}**: `{value}`")
                            else:
                                st.write(row['Best_Params'])

                # 결과 표시
                result_df = pd.DataFrame(results)
                st.subheader("📈 Model Comparison Table")
                st.dataframe(result_df, use_container_width=True)

                # 정확도 비교 차트
                st.subheader("Model Accuracy Comparison")
                fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
                result_df_numeric = result_df.copy()
                result_df_numeric['Accuracy'] = result_df_numeric['Accuracy'].astype(float)
                sns.barplot(data=result_df_numeric, x="Model", y="Accuracy", ax=ax_acc)
                ax_acc.set_ylim(0, 1)
                ax_acc.set_title("Model Accuracy Comparison")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_acc)

                # ROC Curve (이진 분류만)
                if n_classes == 2 and any(y_prob is not None for model_name in model_choices):
                    st.subheader("ROC Curve Comparison")
                    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8)
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.set_title('ROC Curve Comparison')
                    ax_roc.legend()
                    ax_roc.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_roc)

                    st.subheader("Precision-Recall Curve Comparison")
                    ax_pr.set_xlabel('Recall')
                    ax_pr.set_ylabel('Precision')
                    ax_pr.set_title('Precision-Recall Curve Comparison')
                    ax_pr.legend()
                    ax_pr.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_pr)

                # 결과 데이터 다운로드
                st.subheader("📊 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # 모델 성능 결과 CSV 다운로드
                    results_csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📈 Download Model Results (CSV)",
                        data=results_csv,
                        file_name=f"model_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # 하이퍼파라미터 결과 다운로드 (튜닝 활성화된 경우)
                    if enable_tuning and best_params_results:
                        # 파라미터 결과를 CSV 친화적으로 변환
                        params_for_csv = []
                        for item in best_params_results:
                            if isinstance(item['Best_Params'], dict):
                                params_str = '; '.join([f"{k}={v}" for k, v in item['Best_Params'].items()])
                            else:
                                params_str = str(item['Best_Params'])
                            
                            params_for_csv.append({
                                'Model': item['Model'],
                                'CV_Score': item['CV_Score'],
                                'Best_Parameters': params_str
                            })
                        
                        params_df = pd.DataFrame(params_for_csv)
                        params_csv = params_df.to_csv(index=False)
                        
                        st.download_button(
                            label="🎯 Download Best Parameters (CSV)",
                            data=params_csv,
                            file_name=f"best_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.button("🎯 Best Parameters (CSV)", disabled=True, help="Enable hyperparameter tuning to download parameters")
                
                with col3:
                    # 전처리된 데이터셋 다운로드
                    processed_data = pd.concat([X, pd.Series(y_encoded, name=target, index=X.index)], axis=1)
                    processed_csv = processed_data.to_csv(index=False)
                    
                    st.download_button(
                        label="🔧 Download Processed Data (CSV)",
                        data=processed_csv,
                        file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the preprocessed dataset used for training"
                    )
                
                # 예측 결과 다운로드 (테스트셋 + 예측값)
                if model_choices:
                    st.subheader("🔮 Test Set Predictions")
                    
                    # 마지막으로 훈련된 모델로 예측 결과 생성
                    test_predictions = pd.DataFrame(X_test)
                    test_predictions[f'{target}_actual'] = y_test
                    
                    # 각 모델의 예측 결과 추가
                    for model_name in model_choices:
                        # 이미 훈련된 모델 사용
                        model = trained_models[model_name]
                        
                        pred = model.predict(X_test)
                        test_predictions[f'{model_name}_prediction'] = pred
                        
                        # 확률값도 추가 (이진 분류인 경우)
                        if len(set(y_encoded)) == 2:
                            prob = model.predict_proba(X_test)[:, 1]
                            test_predictions[f'{model_name}_probability'] = prob
                    
                    # 예측 결과 미리보기
                    st.dataframe(test_predictions.head(10), use_container_width=True)
                    
                    # 예측 결과 다운로드 버튼
                    predictions_csv = test_predictions.to_csv(index=False)
                    st.download_button(
                        label="🔮 Download Test Predictions (CSV)",
                        data=predictions_csv,
                        file_name=f"test_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download test set with actual values and model predictions"
                    )
                
                # PDF 리포트
                st.subheader("📄 Report Download")
                
                if PDF_AVAILABLE:
                    try:
                        pdf_bytes = BytesIO()
                        pisa.CreatePDF(BytesIO(report_html.encode('utf-8')), dest=pdf_bytes)
                        pdf_base64 = base64.b64encode(pdf_bytes.getvalue()).decode('utf-8')
                        href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="model_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pdf">📄 Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"PDF generation failed: {str(e)}")
                        # PDF 실패 시 HTML 리포트 제공
                        st.download_button(
                            label="📝 Download HTML Report",
                            data=report_html,
                            file_name=f"model_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                else:
                    st.warning("PDF generation not available in this environment")
                    # HTML 리포트 제공
                    st.download_button(
                        label="📝 Download HTML Report",
                        data=report_html,
                        file_name=f"model_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )

            else:
                st.warning("Please select at least one model to compare.")
            
            # 네비게이션 버튼들
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Feature Selection"):
                    st.session_state.step = 2
                    st.rerun()
            
            with col2:
                if st.button("🔄 Restart App"):
                    reset_app()
                    st.rerun()

        except Exception as e:
            st.error(f"An error occurred during model evaluation: {str(e)}")
            st.info("Please go back and check your feature selection.")
            if st.button("← Back to Feature Selection"):
                st.session_state.step = 2
                st.rerun()

else:
    st.info("👆 Please upload a CSV file to proceed.")
    st.markdown("""
    ### How to use this app:
    1. **Upload Data**: Upload your CSV file using the sidebar
    2. **Explore Data**: View distributions and correlations
    3. **Select Features**: Choose input features and target variable
    4. **Evaluate Models**: Compare different ML models and download results
    """)
