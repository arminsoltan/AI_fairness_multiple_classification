import numpy as np
from scipy.optimize import linprog

from aif360.algorithms import Transformer
from aif360.metrics import ClassificationMetric, utils


class EqualizedOddsPostProcessing(Transformer):
    def __init__(self, unprivileged_groups, privileged_groups, seed=None):
        super(EqualizedOddsPostProcessing, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            seed=seed
        )

        self.seed = seed
        self.model_params = None
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups

    def fit(self, dataset_true, dataset_pred):
        metric = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=self.unprivileged_groups,
                                      privileged_groups=self.privileged_groups)

        sbr = metric.base_rate(privileged=True)
        obr = metric.base_rate(privileged=False)

        fpr0 = metric.false_positive_rate(privileged=True)
        fpr1 = metric.false_positive_rate(privileged=False)
        fnr0 = metric.false_negative_rate(privileged=True)
        fnr1 = metric.false_negative_rate(privileged=False)
        tpr0 = metric.true_positive_rate(privileged=True)
        tpr1 = metric.true_positive_rate(privileged=False)
        tnr0 = metric.true_negative_rate(privileged=True)
        tnr1 = metric.true_negative_rate(privileged=False)

        print("fpr0= {}, fpr1= {}, fnr0= {}, fnr1={}, tpr0={}, tpr1={}, tnr0={}, tnr1={}".format(fpr0, fpr1, fnr0, fnr1, tpr0, tpr1, tnr0, tnr1))
        # Coefficients of the linear objective function to be minimized
        c = np.array([fpr0 - tpr0, tnr0 - fnr0, fpr1 - tpr1, tnr1 - fnr1])

        # A_ub - 2-D array which, when matrix-multiplied by x, gives the values
        # of the upper-bound inequality constraints at x
        # b_ub - 1-D array of values representing the upper-bound of each
        # inequality constraint (row) in A_ub.
        # Just to keep these between zero and one
        A_ub = np.array([[1, 0, 0, 0],
                         [-1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, -1]], dtype=np.float64)
        b_ub = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)

        # Create boolean conditioning vectors for protected groups
        cond_vec_priv = utils.compute_boolean_conditioning_vector(
            dataset_pred.protected_attributes,
            dataset_pred.protected_attribute_names,
            self.privileged_groups)
        cond_vec_unpriv = utils.compute_boolean_conditioning_vector(
            dataset_pred.protected_attributes,
            dataset_pred.protected_attribute_names,
            self.unprivileged_groups)

        sconst = np.ravel(
            dataset_pred.labels[cond_vec_priv] == dataset_pred.favorable_label)
        sflip = np.ravel(
            dataset_pred.labels[cond_vec_priv] == dataset_pred.unfavorable_label)
        oconst = np.ravel(
            dataset_pred.labels[cond_vec_unpriv] == dataset_pred.favorable_label)
        oflip = np.ravel(
            dataset_pred.labels[cond_vec_unpriv] == dataset_pred.unfavorable_label)

        y_true = dataset_true.labels.ravel()

        sm_tn = np.logical_and(sflip,
                               y_true[cond_vec_priv] == dataset_true.unfavorable_label,
                               dtype=np.float64)
        sm_fn = np.logical_and(sflip,
                               y_true[cond_vec_priv] == dataset_true.favorable_label,
                               dtype=np.float64)
        sm_fp = np.logical_and(sconst,
                               y_true[cond_vec_priv] == dataset_true.unfavorable_label,
                               dtype=np.float64)
        sm_tp = np.logical_and(sconst,
                               y_true[cond_vec_priv] == dataset_true.favorable_label,
                               dtype=np.float64)

        om_tn = np.logical_and(oflip,
                               y_true[cond_vec_unpriv] == dataset_true.unfavorable_label,
                               dtype=np.float64)
        om_fn = np.logical_and(oflip,
                               y_true[cond_vec_unpriv] == dataset_true.favorable_label,
                               dtype=np.float64)
        om_fp = np.logical_and(oconst,
                               y_true[cond_vec_unpriv] == dataset_true.unfavorable_label,
                               dtype=np.float64)
        om_tp = np.logical_and(oconst,
                               y_true[cond_vec_unpriv] == dataset_true.favorable_label,
                               dtype=np.float64)

        # A_eq - 2-D array which, when matrix-multiplied by x,
        # gives the values of the equality constraints at x
        # b_eq - 1-D array of values representing the RHS of each equality
        # constraint (row) in A_eq.
        # Used to impose equality of odds constraint
        A_eq = [[(np.mean(sconst * sm_tp) - np.mean(sflip * sm_tp)) / sbr,
                 (np.mean(sflip * sm_fn) - np.mean(sconst * sm_fn)) / sbr,
                 (np.mean(oflip * om_tp) - np.mean(oconst * om_tp)) / obr,
                 (np.mean(oconst * om_fn) - np.mean(oflip * om_fn)) / obr],
                [(np.mean(sconst * sm_fp) - np.mean(sflip * sm_fp)) / (1 - sbr),
                 (np.mean(sflip * sm_tn) - np.mean(sconst * sm_tn)) / (1 - sbr),
                 (np.mean(oflip * om_fp) - np.mean(oconst * om_fp)) / (1 - obr),
                 (np.mean(oconst * om_tn) - np.mean(oflip * om_tn)) / (1 - obr)]]

        b_eq = [(np.mean(oflip * om_tp) + np.mean(oconst * om_fn)) / obr
                - (np.mean(sflip * sm_tp) + np.mean(sconst * sm_fn)) / sbr,
                (np.mean(oflip * om_fp) + np.mean(oconst * om_tn)) / (1 - obr)
                - (np.mean(sflip * sm_fp) + np.mean(sconst * sm_tn)) / (1 - sbr)]

        # Linear program
        self.model_params = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

        return self

    def predict(self, dataset):
        """Perturb the predicted labels to obtain new labels that satisfy
        equalized odds constraints.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
            dataset (BinaryLabelDataset): Transformed dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Get the model parameters output from fit
        sp2p, sn2p, op2p, on2p = self.model_params.x

        # Create boolean conditioning vectors for protected groups
        cond_vec_priv = utils.compute_boolean_conditioning_vector(
            dataset.protected_attributes, dataset.protected_attribute_names,
            self.privileged_groups)
        cond_vec_unpriv = utils.compute_boolean_conditioning_vector(
            dataset.protected_attributes, dataset.protected_attribute_names,
            self.unprivileged_groups)

        # Randomly flip labels according to the probabilities in model_params
        self_fair_pred = dataset.labels[cond_vec_priv].copy()
        self_pp_indices, _ = np.nonzero(
            dataset.labels[cond_vec_priv] == dataset.favorable_label)
        self_pn_indices, _ = np.nonzero(
            dataset.labels[cond_vec_priv] == dataset.unfavorable_label)
        np.random.shuffle(self_pp_indices)
        np.random.shuffle(self_pn_indices)

        n2p_indices = self_pn_indices[:int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = dataset.favorable_label
        p2n_indices = self_pp_indices[:int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = dataset.unfavorable_label

        othr_fair_pred = dataset.labels[cond_vec_unpriv].copy()
        othr_pp_indices, _ = np.nonzero(
            dataset.labels[cond_vec_unpriv] == dataset.favorable_label)
        othr_pn_indices, _ = np.nonzero(
            dataset.labels[cond_vec_unpriv] == dataset.unfavorable_label)
        np.random.shuffle(othr_pp_indices)
        np.random.shuffle(othr_pn_indices)

        n2p_indices = othr_pn_indices[:int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = dataset.favorable_label
        p2n_indices = othr_pp_indices[:int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = dataset.unfavorable_label

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy()

        new_labels = np.zeros_like(dataset.labels, dtype=np.float64)
        new_labels[cond_vec_priv] = self_fair_pred
        new_labels[cond_vec_unpriv] = othr_fair_pred

        dataset_new.labels = new_labels

        return dataset_new

    def fit_predict(self, dataset_true, dataset_pred):
        """fit and predict methods sequentially."""
        return self.fit(dataset_true, dataset_pred).predict(dataset_pred)
