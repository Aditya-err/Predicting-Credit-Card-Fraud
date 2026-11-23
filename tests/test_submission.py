import os
import pytest
import numpy as np
import pandas as pd
from src.submission import generate_submission, verify_submission

def test_generate_and_verify_submission(tmp_path):
    test_ids = pd.Series([100, 101, 102])
    test_preds = np.array([0.05, 0.95, 0.50])
    
    sub_path = str(tmp_path / "test_sub.csv")
    sample_path = str(tmp_path / "sample_sub.csv")
    
    pd.DataFrame({"id": [100, 101, 102], "is_fraud": [0.0, 0.0, 0.0]}).to_csv(sample_path, index=False)
    
    generate_submission(test_ids, test_preds, sub_path)
    is_valid, checks = verify_submission(sub_path, sample_path)
    
    assert is_valid is True
    assert checks["row_count_matches"] is True
    assert checks["no_nulls"] is True
    assert checks["values_within_0_and_1"] is True

def test_verify_submission_detects_out_of_bounds(tmp_path):
    sub_path = str(tmp_path / "bad_sub.csv")
    sample_path = str(tmp_path / "sample_sub.csv")
    
    pd.DataFrame({"id": [100, 101], "is_fraud": [0.0, 0.0]}).to_csv(sample_path, index=False)
    pd.DataFrame({"id": [100, 101], "is_fraud": [-0.1, 1.2]}).to_csv(sub_path, index=False)
    
    is_valid, checks = verify_submission(sub_path, sample_path)
    assert is_valid is False
    assert checks["values_within_0_and_1"] is False
