import numpy as np

import tensorflow as tf
from sklearn.ensemble import IsolationForest

class EarlyStoppingBeforeOverfit(tf.keras.callbacks.Callback):
    """
    """

    def __init__(self, patience=15, min_acc=0.95):
        super(EarlyStoppingBeforeOverfit, self).__init__()
        self.min_acc = min_acc
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

        
    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0
        self.wait = 0
        self.lowest_point = 1

        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("accuracy")
        if (current < self.min_acc) or self.wait == self.patience - 1:
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)
        else:
            if current <= self.lowest_point:
                self.stopped_epoch = epoch
                self.best_weights = self.model.get_weights()
                self.wait = 0
                self.lowest_point = current
            else:
                self.wait += 1

                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            
            
def sample_enrichment_IF(r_seed, target_data, sample_size):
    np.random.seed(r_seed)
    domain_max = target_data.max(axis=0)
    domain_min = target_data.min(axis=0)
    domain_dim = target_data.shape[1]

    sample_enri = np.random.random(size=(sample_size, domain_dim))
    
    domain_gap = (domain_max - domain_min) * 1.2
    domain_mean = (domain_max + domain_min) / 2
    
    for dim_idx in range(domain_dim):
        sample_enri[:, dim_idx] = sample_enri[:, dim_idx] * domain_gap[
            dim_idx] + domain_mean[dim_idx] - domain_gap[dim_idx] / 2
    
    clf = IsolationForest(random_state=r_seed, max_samples=0.9).fit(target_data)
    sample_coef = clf.score_samples(sample_enri)
    sample_coef -= sample_coef.min()
    sample_coef /= sample_coef.max()
    print(np.unique(sample_coef).shape)
    return sample_enri, np.squeeze(sample_coef)


class aosr_risk(tf.keras.losses.Loss):
    
    def __init__(self, model, x_q, x_w, z_p_X, outlier_ratio, k):
        super().__init__(name='pq_risk')
        self.model = model
        self.x_q = x_q
        self.x_w = x_w
        self.k = k
        self.z_p_X = z_p_X
        self.outlier_ratio = outlier_ratio
 
    def call(self, y_true, y_pred):
        
        Rs_all_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        y_t_pred = self.model(self.x_q)
        y_true_q = np.zeros(self.x_w.shape[0]) + self.k
        Rt_k_hat = tf.keras.losses.sparse_categorical_crossentropy(y_true_q, y_t_pred)
        Rt_k_hat = tf.math.multiply(tf.convert_to_tensor(self.x_w, dtype=tf.float32), Rt_k_hat)
        Rt_k_hat = tf.reduce_mean(Rt_k_hat)

        num_out = tf.math.argmax(self.model(self.z_p_X), axis=1)
        num_out = tf.reduce_sum(tf.cast(tf.equal(num_out, self.k), tf.int32))     
        num_out = tf.cast(num_out, tf.float32 )

        outlier = self.z_p_X.shape[0] * self.outlier_ratio
        if num_out == 0:
            num_out = 0.00001
        num_out = tf.stop_gradient(num_out)
        
        return Rs_all_hat + (outlier*1.0/(num_out*1.0))* Rt_k_hat
    
    