
import shap
import matplotlib.pyplot as plt
import scipy.sparse as sp

def explain_xgb_model(model, X_sample, feature_names=None):
    if sp.issparse(X_sample):
        X_sample = X_sample.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
    plt.show()

    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar")
    plt.show()
