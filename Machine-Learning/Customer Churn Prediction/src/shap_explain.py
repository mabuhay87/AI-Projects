
import shap
import matplotlib.pyplot as plt

def explain_with_shap(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample)
    plt.show()
