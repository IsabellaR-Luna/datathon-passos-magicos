# test_preprocessing.py
"""
Testes para o módulo de pré-processamento.
"""

import pytest
import numpy as np
import pandas as pd

from app.services.clustering.core import (
    DataPreprocessor,
    PreprocessingError,
    ValidationResult
)


class TestDataPreprocessor:
    """Testes para a classe DataPreprocessor."""
    
    def test_init(self):
        """Testa inicialização do preprocessor."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor.scaler is not None
        assert preprocessor._is_fitted == False
        assert preprocessor._training_medians is None
    
    def test_feature_columns(self):
        """Testa se as colunas de features estão definidas."""
        assert DataPreprocessor.FEATURE_COLUMNS == [
            'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'Defas'
        ]
    
    def test_expected_ranges(self):
        """Testa se os ranges esperados estão definidos."""
        ranges = DataPreprocessor.EXPECTED_RANGES
        
        assert 'IAA' in ranges
        assert ranges['IAA'] == (0, 10)
        assert ranges['Defas'] == (-5, 5)
    
    def test_validate_input_valid(self, sample_df):
        """Testa validação com dados válidos."""
        preprocessor = DataPreprocessor()
        result = preprocessor.validate_input(sample_df)
        
        assert isinstance(result, ValidationResult)
        assert result.valid == True
        assert len(result.errors) == 0
    
    def test_validate_input_missing_columns(self):
        """Testa validação com colunas faltando."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({'IAA': [1, 2, 3]})  # Faltam outras colunas
        
        result = preprocessor.validate_input(df)
        
        assert result.valid == False
        assert len(result.errors) > 0
        assert 'Colunas faltando' in result.errors[0]
    
    def test_validate_input_with_nulls(self, sample_df_with_nulls):
        """Testa validação com valores nulos (deve gerar warnings)."""
        preprocessor = DataPreprocessor()
        result = preprocessor.validate_input(sample_df_with_nulls)
        
        assert result.valid == True  # Nulos geram warnings, não erros
        assert len(result.warnings) > 0
    
    def test_fit_transform_shape(self, sample_df):
        """Testa se fit_transform retorna shape correto."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_df)
        
        assert isinstance(X, np.ndarray)
        assert X.shape == (5, 7)  # 5 amostras, 7 features
    
    def test_fit_transform_normalized(self, sample_df):
        """Testa se dados são normalizados (média ~0, std ~1)."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_df)
        
        # Média próxima de 0
        assert np.abs(X.mean()) < 0.5
        # Std próximo de 1
        assert np.abs(X.std() - 1) < 0.5
    
    def test_fit_transform_sets_fitted(self, sample_df):
        """Testa se fit_transform marca como fitted."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor._is_fitted == False
        
        preprocessor.fit_transform(sample_df)
        
        assert preprocessor._is_fitted == True
    
    def test_fit_transform_saves_medians(self, sample_df):
        """Testa se fit_transform salva as medianas."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_df)
        
        assert preprocessor._training_medians is not None
        assert 'IAA' in preprocessor._training_medians
        assert 'IEG' in preprocessor._training_medians
    
    def test_fit_transform_handles_nulls(self, sample_df_with_nulls):
        """Testa se fit_transform trata valores nulos."""
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_df_with_nulls)
        
        # Não deve ter NaN no resultado
        assert not np.isnan(X).any()
    
    def test_transform_without_fit_raises_error(self, sample_df):
        """Testa se transform sem fit levanta erro."""
        preprocessor = DataPreprocessor()
        
        with pytest.raises(PreprocessingError):
            preprocessor.transform(sample_df)
    
    def test_transform_after_fit(self, sample_df):
        """Testa transform após fit."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_df)
        
        # Deve funcionar sem erro
        X = preprocessor.transform(sample_df)
        
        assert isinstance(X, np.ndarray)
        assert X.shape == (5, 7)
    
    def test_transform_single(self, sample_df, sample_features):
        """Testa transform de um único registro."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_df)
        
        X = preprocessor.transform_single(sample_features)
        
        assert isinstance(X, np.ndarray)
        assert X.shape == (1, 7)
    
    def test_transform_single_without_fit_raises_error(self, sample_features):
        """Testa transform_single sem fit levanta erro."""
        preprocessor = DataPreprocessor()
        
        with pytest.raises(PreprocessingError):
            preprocessor.transform_single(sample_features)
    
    def test_get_state(self, sample_df):
        """Testa serialização do estado."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_df)
        
        state = preprocessor.get_state()
        
        assert 'scaler_mean' in state
        assert 'scaler_scale' in state
        assert 'training_medians' in state
        assert 'is_fitted' in state
        assert state['is_fitted'] == True
    
    def test_load_state(self, sample_df):
        """Testa carregamento do estado."""
        preprocessor1 = DataPreprocessor()
        preprocessor1.fit_transform(sample_df)
        state = preprocessor1.get_state()
        
        preprocessor2 = DataPreprocessor()
        preprocessor2.load_state(state)
        
        assert preprocessor2._is_fitted == True
        assert preprocessor2._training_medians == preprocessor1._training_medians
    
    def test_fit_transform_invalid_data_raises_error(self):
        """Testa se dados inválidos levantam erro."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({'X': [1, 2, 3]})  # Colunas erradas
        
        with pytest.raises(PreprocessingError):
            preprocessor.fit_transform(df)
