import pandas as pd
from scipy import stats

def validate(synthetic_data, real_data=None):
    """
    Validate the generated synthetic data by comparing it to real data or performing statistical tests.

    :param synthetic_data: DataFrame of synthetic agricultural data.
    :param real_data: DataFrame of real agricultural data to compare against (optional).
    :return: Dictionary containing validation report with summary statistics and statistical test results.
    """
    validation_report = {
        'summary_statistics': synthetic_data.describe(),  # Summary statistics for synthetic data
        'distribution_tests': {},  # Will hold KS-test or other distribution tests
        'correlation_matrix': synthetic_data.corr()  # Correlation matrix for synthetic data
    }

    # If real data is provided, perform distribution comparison (KS test)
    if real_data is not None:
        ks_test_results = {}
        common_columns = set(real_data.columns).intersection(set(synthetic_data.columns))

        for column in common_columns:
            real_column = real_data[column].dropna()
            synthetic_column = synthetic_data[column].dropna()

            if len(real_column) > 0 and len(synthetic_column) > 0:
                ks_statistic, p_value = stats.ks_2samp(real_column, synthetic_column)
                ks_test_results[column] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value
                }

        validation_report['distribution_tests'] = ks_test_results

    return validation_report
