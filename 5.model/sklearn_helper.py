"""
Script: sklearn_helper.py
============================
Performs inference on test data using a trained sklearn ensemble model.

This script reads the sklearn ensemble model from a pickle file, performs inference on the test set, and saves the
prediction results to a csv file.

Steps:
    Load the ensemble model from a pickle file.
    Perform inference on the test set.
    Calculate metrics (RMSE and MAE) for the model's performance on the test set.
    Load the original test data.
    Prepare the prediction results, including actuals, predictions, and the differences between them.
    Merge these results with the original test data.
    Save the merged data to a csv file.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importances(clf, X_train, y_train=None,
                             top_n=10, figsize=(6, 6), print_table=False, title="Feature Importances", plot=True):
        '''
        plot feature importances of a tree-based sklearn estimator

        Note: X_train and y_train are pandas DataFrames

        Note: Scikit-plot is a lovely package but I sometimes have issues
                  1. flexibility/extendibility
                  2. complicated models/datasets
              But for many situations Scikit-plot is the way to go
              see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

        Parameters
        ----------
            clf         (sklearn estimator) if not fitted, this routine will fit it

            X_train     (pandas DataFrame)

            y_train     (pandas DataFrame)  optional
                                            required only if clf has not already been fitted

            top_n       (int)               Plot the top_n most-important features
                                            Default: 10

            figsize     ((int,int))         The physical size of the plot
                                            Default: (8,8)

            print_table (boolean)           If True, print out the table of feature importances
                                            Default: False

        Returns
        -------
            the pandas dataframe with the features and their importance

        Author
        ------
            George Fisher
        '''

        __name__ = "plot_feature_importances"

        from xgboost.core import XGBoostError
        from lightgbm.sklearn import LightGBMError

        try:
            if not hasattr(clf, 'feature_importances_'):
                clf.fit(X_train.values, y_train.values.ravel())

                if not hasattr(clf, 'feature_importances_'):
                    raise AttributeError("{} does not have feature_importances_ attribute".
                                         format(clf.__class__.__name__))

        except (XGBoostError, LightGBMError, ValueError):
            clf.fit(X_train.values, y_train.values.ravel())


        feat_imp = pd.DataFrame({'importance': clf.feature_importances_})
        feat_imp['feature'] = X_train.columns
        feat_imp.sort_values(by='importance', ascending=False, inplace=True)
        feat_imp = feat_imp.iloc[:top_n]

        feat_imp.sort_values(by='importance', inplace=True)
        feat_imp = feat_imp.set_index('feature', drop=True)

        if plot == True:
            feat_imp.plot.barh(title=title, figsize=(10, 6))  # Increase the figure size
            plt.xlabel('Feature Importance Score')
            plt.yticks(rotation=45, ha='right')  # Adjust the rotation and horizontal alignment
            plt.tight_layout()  # Adjust the layout to fit labels
            plt.show()

        # add these lines just before 'if print_table:'
        feat_imp_export = feat_imp.copy()  # create a copy to prevent modifying the original df
        feat_imp_export.reset_index(inplace=True)  # reset index to get the feature column back
        feat_imp_export = feat_imp_export.sort_values(by='importance', ascending=False)  # sort by importance in descending order

        if print_table:
            from IPython.display import display
            print("Top {} features in descending order of importance".format(top_n))
            display(feat_imp_export)

            # save df to csv, save best_models to pickle
            cwd = os.getcwd()
            model_folder = os.path.join(cwd, 'models')

            feat_imp_export.to_csv(os.path.join(model_folder, 'skearn_feature_importances.csv'), index=False)

        return feat_imp

def adjusted_r2(r2, n, p):
    """Function to calculate adjusted R-squared"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)